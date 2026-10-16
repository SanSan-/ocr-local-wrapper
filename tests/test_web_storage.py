from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from ocr_local import service
from ocr_local.model_profiles import resolve_options, settings_identity
from ocr_local.models import LoadedOcrModel, OcrOptions
from ocr_local.utils import output_cache
from ocr_local.web.errors import WebError
from ocr_local.web.limits import WebLimits
from ocr_local.web.storage import StorageBudget

LOG = logging.getLogger(__name__)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    image = uploads / "input.png"
    Image.new("RGB", (4, 4), "white").save(image)
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(output_cache, "OCR_CACHE_FILE", cache)
    options = resolve_options(OcrOptions(image, model_path=tmp_path / "missing-model"))
    limits = replace(WebLimits(), text_bytes=32, cache_entry_bytes=2048, cache_bytes=16384, result_bytes=32768)
    return StorageBudget(uploads, cache, limits), options


def digest_files(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def seed(options, text="Прежний текст", **kwargs):
    output_cache.update_output_cache(
        options.input_path, options.input_path.with_suffix(".txt"), text, LOG,
        settings=settings_identity(options), **kwargs,
    )


def fail_loader(*args):
    raise AssertionError("Модель не должна загружаться.")


def fake_ocr(monkeypatch, text):
    monkeypatch.setattr(service, "recognize_image_text", lambda *args, **kwargs: text)
    return LoadedOcrModel(None, None, "cpu", False)


def test_existing_files_and_full_cache_are_accounted(storage):
    budget, options = storage
    nested = budget.uploads_dir / "nested"
    nested.mkdir()
    (nested / "manual.TXT").write_text("Ручной текст", encoding="utf-8")
    (nested / "other.JPG").write_bytes(b"image")
    budget.cache_file.write_text('{"legacy": {"text": "old"}}', encoding="utf-8")
    usage = budget.usage()
    assert usage.upload_count == 2
    assert usage.upload_bytes == options.input_path.stat().st_size + 5
    assert usage.result_count == 1
    assert usage.result_bytes == (nested / "manual.TXT").stat().st_size + budget.cache_file.stat().st_size


@pytest.mark.parametrize("dimension", ["upload_bytes", "upload_count"])
def test_upload_boundary_dedup_and_denial_preserve_existing_files(storage, tmp_path, dimension):
    budget, options = storage
    content = b"validated-image"
    temporary = tmp_path / "upload.part"
    temporary.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    usage = budget.usage()
    value = usage.upload_bytes + len(content) if dimension == "upload_bytes" else usage.upload_count + 1
    budget.limits = replace(budget.limits, **{dimension: value})
    target = budget.publish_upload(temporary, digest, ".png")
    before = digest_files([options.input_path, target])
    assert budget.publish_upload(temporary, digest, ".png") == target
    with pytest.raises(WebError) as error:
        budget.publish_upload(temporary, "a" * 64, ".png")
    assert error.value.status_code == 507 and error.value.code == "storage_quota_exceeded"
    assert digest_files([options.input_path, target]) == before
    assert budget.usage().upload_count == 2


def test_parallel_upload_admission_uses_one_count_slot(storage, tmp_path):
    budget, _ = storage
    budget.limits = replace(budget.limits, upload_count=2)
    paths = [tmp_path / "first.part", tmp_path / "second.part"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index + 1]))
    start = threading.Barrier(2)

    def publish(index):
        start.wait()
        try:
            return budget.publish_upload(paths[index], str(index + 1) * 64, ".png")
        except WebError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, range(2)))
    assert sum(isinstance(item, Path) for item in outcomes) == 1
    assert outcomes.count("storage_quota_exceeded") == 1
    assert budget.usage().upload_count == 2


@pytest.mark.parametrize("operation", ["reserve", "publish"])
def test_preflight_io_error_returns_safe_storage_failure_without_writes(storage, monkeypatch, tmp_path, operation):
    budget, options = storage
    temporary = tmp_path / "upload.part"
    temporary.write_bytes(b"validated")
    before = digest_files([options.input_path, temporary])

    def fail_usage():
        raise OSError("Private path in metadata failure")

    monkeypatch.setattr(budget, "_usage", fail_usage)
    action, args = ((budget.reserve, (options,)) if operation == "reserve"
                    else (budget.publish_upload, (temporary, "a" * 64, ".png")))
    with pytest.raises(WebError) as error:
        action(*args)
    assert (error.value.code, error.value.status_code) == ("storage_unavailable", 507)
    assert "Private path" not in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    assert digest_files([options.input_path, temporary]) == before
    assert sorted(path.name for path in budget.uploads_dir.iterdir()) == [options.input_path.name]
    assert not budget.cache_file.exists()


def test_staged_upload_is_not_counted_in_permanent_quota(storage):
    budget, _ = storage
    staging = budget.uploads_dir / ".ocr-local-tmp"
    staging.mkdir()
    temporary = staging / "image-staged.png"
    temporary.write_bytes(b"validated")
    usage = budget.usage()
    assert usage.upload_count == 1
    budget.limits = replace(budget.limits, upload_bytes=usage.upload_bytes + temporary.stat().st_size,
                           upload_count=usage.upload_count + 1)
    target = budget.publish_upload(temporary, hashlib.sha256(temporary.read_bytes()).hexdigest(), ".png")
    assert target.is_file()
    assert budget.usage().upload_count == 2 and budget.usage().upload_bytes == budget.limits.upload_bytes


@pytest.mark.parametrize("dimension", ["result_bytes", "cache_bytes", "result_count"])
def test_preflight_reserves_maximum_growth_and_release_is_idempotent(storage, dimension):
    budget, options = storage
    expected = {"result_bytes": budget.limits.text_bytes + budget.limits.cache_entry_bytes,
                "cache_bytes": budget.limits.cache_entry_bytes, "result_count": 1}[dimension]
    budget.limits = replace(budget.limits, **{dimension: expected - 1})
    with pytest.raises(WebError, match="квоте"):
        budget.reserve(options)
    assert not budget.cache_file.exists() and not options.input_path.with_suffix(".txt").exists()
    budget.limits = replace(budget.limits, **{dimension: expected})
    reservation = budget.reserve(options)
    with pytest.raises(WebError, match="квоте"):
        budget.reserve(options)
    reservation.release()
    reservation.release()
    budget.reserve(options).release()


def test_parallel_result_admission_cannot_use_same_free_budget(storage):
    budget, options = storage
    budget.limits = replace(budget.limits, result_bytes=budget.limits.text_bytes + budget.limits.cache_entry_bytes)
    start = threading.Barrier(2)

    def reserve(_):
        start.wait()
        try:
            return budget.reserve(options)
        except WebError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(reserve, range(2)))
    assert outcomes.count("storage_quota_exceeded") == 1
    for outcome in outcomes:
        if not isinstance(outcome, str):
            outcome.release()


def test_cache_restore_reserves_only_actual_text_and_preserves_cache(storage):
    budget, options = storage
    seed(options, "Кеш", limit_reached=True)
    original_cache = digest_files([budget.cache_file])
    text_size = len("Кеш".encode("utf-8"))
    budget.limits = replace(budget.limits, result_bytes=budget.cache_file.stat().st_size + text_size,
                           cache_bytes=budget.cache_file.stat().st_size, result_count=1)
    reservation = budget.reserve(options)
    assert reservation.text_growth == text_size and reservation.cache_growth == 0 and reservation.count == 1
    try:
        result = service.run_ocr(options, loader=fail_loader, storage_policy=reservation)
    finally:
        reservation.release()
    assert result.cached and result.metrics.limit_reached
    assert result.output_path.read_text(encoding="utf-8") == "Кеш"
    assert not result.output_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert digest_files([budget.cache_file]) == original_cache


@pytest.mark.parametrize("dimension", ["result_bytes", "result_count"])
def test_cache_restore_denied_before_model_and_without_disk_changes(storage, dimension):
    budget, options = storage
    seed(options, "Кеш")
    before = digest_files([options.input_path, budget.cache_file])
    value = budget.cache_file.stat().st_size + len("Кеш".encode("utf-8")) - 1 if dimension == "result_bytes" else 0
    budget.limits = replace(budget.limits, **{dimension: value})
    with pytest.raises(WebError) as error:
        budget.reserve(options)
    assert error.value.code == "storage_quota_exceeded" and error.value.status_code == 507
    assert not options.input_path.with_suffix(".txt").exists()
    assert digest_files([options.input_path, budget.cache_file]) == before


def test_cache_without_write_and_existing_txt_read_remain_available_over_total_quota(storage):
    budget, options = storage
    seed(options, "Кеш")
    output = options.input_path.with_suffix(".txt")
    output.write_text("Моя правка", encoding="utf-8")
    before = digest_files([output, budget.cache_file])
    budget.limits = replace(budget.limits, result_bytes=1, result_count=0)
    reservation = budget.reserve(options)
    try:
        result = service.run_ocr(options, loader=fail_loader, storage_policy=reservation)
    finally:
        reservation.release()
    assert result.text == "Кеш"
    assert budget.read_text(output) == "Моя правка"
    assert digest_files([output, budget.cache_file]) == before
    force_options = replace(options, force=True)
    with pytest.raises(WebError, match="квоте"):
        budget.reserve(force_options)


def test_force_reserves_only_positive_growth_and_does_not_repeat_file_count(storage):
    budget, options = storage
    seed(options, "old")
    output = options.input_path.with_suffix(".txt")
    output.write_text("old", encoding="utf-8")
    canonical_size = len(json.dumps(budget.load_cache(), ensure_ascii=False, indent=2).encode("utf-8"))
    budget.limits = replace(budget.limits, cache_entry_bytes=canonical_size, result_count=1,
                           result_bytes=budget.limits.text_bytes + max(canonical_size, budget.cache_file.stat().st_size))
    reservation = budget.reserve(replace(options, force=True))
    assert reservation.text_growth == budget.limits.text_bytes - 3
    assert reservation.cache_growth == max(0, canonical_size - budget.cache_file.stat().st_size)
    assert reservation.count == 0
    reservation.release()


@pytest.mark.parametrize("cached", [False, True])
def test_oversized_text_preserves_previous_data(storage, monkeypatch, cached):
    budget, options = storage
    seed(options, "Прежний")
    output = options.input_path.with_suffix(".txt")
    output.write_text("Ручной TXT", encoding="utf-8")
    before = digest_files([options.input_path, output, budget.cache_file])
    if cached:
        budget.limits = replace(budget.limits, text_bytes=1)
        with pytest.raises(WebError) as error:
            budget.reserve(options)
    else:
        options = replace(options, force=True)
        reservation = budget.reserve(options)
        loaded = fake_ocr(monkeypatch, "я" * (budget.limits.text_bytes // 2 + 1))
        try:
            with pytest.raises(WebError) as error:
                service.run_ocr(options, loaded=loaded, storage_policy=reservation)
        finally:
            reservation.release()
    assert error.value.code == "result_too_large"
    assert digest_files([options.input_path, output, budget.cache_file]) == before


@pytest.mark.parametrize("cached", [False, True])
def test_public_result_checked_before_first_write(storage, monkeypatch, cached):
    budget, options = storage
    if cached:
        seed(options, "Кеш")
    before = digest_files([options.input_path, budget.cache_file])

    def reject_result(result):
        assert result.cached == cached
        raise WebError("result_too_large", 413, "Публичный итог превышает допустимый размер.")

    reservation = budget.reserve(options, validate_result=reject_result)
    loaded = fake_ocr(monkeypatch, "Новый")
    try:
        with pytest.raises(WebError) as error:
            service.run_ocr(options, loaded=loaded, storage_policy=reservation)
    finally:
        reservation.release()
    assert error.value.code == "result_too_large"
    assert not options.input_path.with_suffix(".txt").exists()
    assert digest_files([options.input_path, budget.cache_file]) == before


@pytest.mark.parametrize("force", [False, True])
def test_exact_serialized_json_checked_before_first_txt_write(storage, monkeypatch, force):
    budget, options = storage
    output = options.input_path.with_suffix(".txt")
    if force:
        seed(options, "Прежний")
        output.write_text("Ручной TXT", encoding="utf-8")
        options = replace(options, force=True)
    before = digest_files([options.input_path, output, budget.cache_file])
    # Небольшой текст с экранированием помещается в TXT, но запись со всеми метаданными — нет.
    text = "\x00" * budget.limits.text_bytes
    encoded = output_cache.serialize_cache_update(budget.load_cache(), options.input_path, output,
                                                 text, settings=settings_identity(options))
    budget.limits = replace(budget.limits, cache_entry_bytes=len(encoded.encode("utf-8")) - 1)
    reservation = budget.reserve(options)
    try:
        loaded = fake_ocr(monkeypatch, text)
        with pytest.raises(WebError) as error:
            service.run_ocr(options, loaded=loaded, storage_policy=reservation)
    finally:
        reservation.release()
    assert error.value.code == "storage_quota_exceeded"
    assert output.exists() == force and budget.cache_file.exists() == force
    assert digest_files([options.input_path, output, budget.cache_file]) == before


def test_success_preserves_old_cache_records_and_uses_exact_utf8_bytes(storage, monkeypatch):
    budget, options = storage
    output_cache.update_output_cache(options.input_path, options.input_path.with_suffix(".txt"), "legacy", LOG)
    old_cache = budget.load_cache()
    text = "Строка\n\x00"
    expected = output_cache.serialize_cache_update(old_cache, options.input_path, options.input_path.with_suffix(".txt"),
                                                  text, settings=settings_identity(options))
    base_size = len(json.dumps(old_cache, ensure_ascii=False, indent=2).encode("utf-8"))
    budget.limits = replace(budget.limits, cache_entry_bytes=len(expected.encode("utf-8")) - base_size)
    reservation = budget.reserve(options)
    try:
        result = service.run_ocr(options, loaded=fake_ocr(monkeypatch, text), storage_policy=reservation)
    finally:
        reservation.release()
    assert result.output_path.read_bytes() == text.encode("utf-8")
    assert budget.cache_file.read_bytes() == expected.encode("utf-8")
    assert output_cache.get_cached_text(budget.load_cache(), options.input_path) == "legacy"


@pytest.mark.parametrize("artifact", ["cache", "text"])
def test_old_large_file_is_rejected_without_opening_or_modifying_it(storage, monkeypatch, artifact):
    budget, options = storage
    path = budget.cache_file if artifact == "cache" else options.input_path.with_suffix(".txt")
    path.write_text("0123456789", encoding="utf-8")
    budget.limits = replace(budget.limits, cache_bytes=9, text_bytes=9)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self != path, "Сверхлимитный файл нельзя открывать для полного чтения."
        return original_open(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        action, args = ((budget.load_cache, ()) if artifact == "cache" else (budget.read_text, (path,)))
        with pytest.raises(WebError) as error:
            action(*args)
    assert error.value.code == "stored_data_too_large"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    if artifact == "cache":
        output = options.input_path.with_suffix(".txt")
        output.write_text("TXT", encoding="utf-8")
        assert budget.read_text(output) == "TXT"
        force_options = replace(options, force=True)
        with pytest.raises(WebError):
            budget.reserve(force_options)


def test_invalid_existing_cache_is_not_discarded_by_web(storage):
    budget, options = storage
    budget.cache_file.write_text('{"legacy":', encoding="utf-8")
    before = digest_files([budget.cache_file])
    force_options = replace(options, force=True)
    with pytest.raises(WebError) as error:
        budget.reserve(force_options)
    assert error.value.code == "cache_unavailable"
    assert digest_files([budget.cache_file]) == before


def test_growth_after_admission_rechecked_before_writing(storage, monkeypatch):
    budget, options = storage
    reservation = budget.reserve(options)
    manual = budget.uploads_dir / "manual.txt"
    manual.write_bytes(b"x" * budget.limits.result_bytes)
    before = digest_files([manual, options.input_path])
    try:
        loaded = fake_ocr(monkeypatch, "Новый")
        with pytest.raises(WebError, match="квоте"):
            service.run_ocr(options, loaded=loaded, storage_policy=reservation)
    finally:
        reservation.release()
    assert not options.input_path.with_suffix(".txt").exists() and not budget.cache_file.exists()
    assert digest_files([manual, options.input_path]) == before


def test_cli_without_policy_is_not_subject_to_web_text_limit(storage, monkeypatch):
    budget, options = storage
    text = "я" * (budget.limits.text_bytes + 1)
    result = service.run_ocr(options, loaded=fake_ocr(monkeypatch, text))
    assert result.output_path.read_text(encoding="utf-8") == text
    assert output_cache.get_cached_text(output_cache.load_cache_snapshot(LOG), options.input_path,
                                        settings=settings_identity(options)) == text

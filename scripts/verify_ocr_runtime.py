"""Проверка реальных моделей, повторного использования и смены на искусственных данных."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ocr_local.models import OcrOptions
from ocr_local.service import run_ocr
from ocr_local.utils import output_cache
from ocr_local.utils.model_utils import ModelPool, load_model_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_cache.OCR_CACHE_FILE = root / "cache.json"
    image_path = root / "cyrillic.png"
    image = Image.new("RGB", (800, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 106, 799, 372), fill="red")
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 92)
    draw.multiline_text((52, 146), "ПОРОШОК\nНЕ ВХОДИ", fill="white", font=font, spacing=0)
    if not image_path.exists():
        image.save(image_path)
    source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    loads, records = [], []

    def loader(path, quantization, cpu_fallback):
        loaded = load_model_components(path, quantization, cpu_fallback)
        loads.append({"path": str(path), "dtype": str(loaded.model.dtype), "device": str(loaded.device)})
        return loaded

    pool = ModelPool(loader)
    sequence = ["PaddleOCR-VL-1.6", "PaddleOCR-VL-1.6", "GLM-OCR", "PaddleOCR-VL-1.6"]
    for index, name in enumerate(sequence):
        options = OcrOptions(
            input_path=image_path, output_path=root / f"{index}-{name}.txt",
            model_path=args.models_root / name, max_new_tokens=128, force=True,
        )
        result = run_ocr(options, loader=pool.get)
        normalized = " ".join(result.text.upper().split())
        visual = normalized.translate(str.maketrans({"H": "Н", "E": "Е", "B": "В", "X": "Х", "O": "О", "P": "Р", "K": "К"}))
        assert "ПОРОШ" in visual and "ВХОД" in visual, result.text
        assert result.device == "cuda" and not result.cached and not result.metrics.limit_reached
        records.append({"model": name, "text": result.text, "settings": result.settings,
                        "metrics": asdict(result.metrics),
                        "exact_cyrillic_match": result.text == "ПОРОШОК\nНЕ ВХОДИ"})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    assert len(loads) == 3, loads
    assert source_hash == hashlib.sha256(image_path.read_bytes()).hexdigest()
    pool.clear()
    import torch
    report = {
        "runtime_status": "PASS", "quality_status": "PASS" if all(r["exact_cyrillic_match"] for r in records) else "partial",
        "loads": loads, "runs": records, "input_sha256": source_hash,
        "cuda_allocated_after_clear": torch.cuda.memory_allocated(),
        "scope": "Синтетический кириллический знак, повтор и смена обеих моделей; не общий корпус качества.",
    }
    report_path = root / "runtime.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT={report_path}", flush=True)


if __name__ == "__main__":
    main()

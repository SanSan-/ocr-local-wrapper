"""Сопоставимый локальный замер на искусственном изображении без рабочих данных."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXT = "LOCAL OCR TEST\nInvoice 12345\nDate 2026-09-20\nProduct: Notebook\nQuantity: 12\nPrice: 250.00\nTotal: 3000.00\nThank you!"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.baseline_source:
        sys.path.insert(0, str(args.baseline_source.resolve()))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    image_path = args.image_path.resolve() if args.image_path else root / "synthetic.png"
    if not image_path.exists():
        image = Image.new("RGB", (1000, 760), "white")
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 44)
        ImageDraw.Draw(image).multiline_text((50, 40), TEXT, fill="black", font=font, spacing=25)
        image.save(image_path)
    import torch
    import transformers
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    from ocr_local.models import OcrOptions
    from ocr_local.service import run_ocr
    from ocr_local.utils import output_cache
    from ocr_local.utils.model_utils import load_model_components
    output_cache.OCR_CACHE_FILE = root / f"{args.variant}-cache.json"
    torch.cuda.synchronize()
    started = time.perf_counter()
    loaded = load_model_components(args.model_path, False, False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"load_seconds={load_seconds}", flush=True)
    records = []
    for index in range(args.runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = run_ocr(
            OcrOptions(
                input_path=image_path, output_path=root / f"{args.variant}-{index}.txt",
                model_path=args.model_path, prompt="OCR:", max_new_tokens=160,
                quantization_enabled=False, force=True,
            ),
            loaded=loaded,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = {"run": index, "elapsed_seconds": elapsed, "text": result.text}
        if hasattr(result, "metrics"):
            from dataclasses import asdict
            record["metrics"] = asdict(result.metrics)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    data = {
        "variant": args.variant, "model_path": str(args.model_path),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(), "dtype": str(loaded.model.dtype),
        "model_use_cache": getattr(loaded.model.config.get_text_config(), "use_cache", None),
        "generation_use_cache": loaded.model.generation_config.use_cache,
        "load_seconds": load_seconds, "runs": records,
        "max_new_tokens": 160, "prompt": "OCR:",
        "input_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "expected_text": TEXT,
    }
    report = root / f"{args.variant}.json"
    if report.exists():
        raise FileExistsError(report)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT={report}", flush=True)


if __name__ == "__main__":
    main()

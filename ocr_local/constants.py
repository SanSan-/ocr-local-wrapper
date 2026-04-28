from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
CACHE_DIR = RESOURCES_DIR / "cache"
UPLOADS_DIR = RESOURCES_DIR / "uploads"
OCR_CACHE_FILE = CACHE_DIR / "ocr_output_cache.json"
DEFAULT_MODEL_PATH = Path(r".\models\GLM-OCR")
DEFAULT_PROMPT = "Text Recognition:"
DEFAULT_MAX_NEW_TOKENS = 8192
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

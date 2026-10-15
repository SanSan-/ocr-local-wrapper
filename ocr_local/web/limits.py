from __future__ import annotations

from dataclasses import dataclass

KIB = 1024
MIB = 1024 * KIB


@dataclass(frozen=True)
class WebLimits:
    """Единые пределы веба; тесты могут передать уменьшенные значения."""

    image_bytes: int = 25 * MIB
    total_body_bytes: int = 26 * MIB
    form_field_bytes: int = 32 * KIB
    json_bytes: int = 32 * KIB
    prompt_bytes: int = 8 * KIB
    image_pixels: int = 40_000_000
    upload_total_seconds: float = 60.0
    upload_idle_seconds: float = 10.0
    temp_bytes: int = 52 * MIB
    upload_bytes: int = 1024 * MIB
    upload_count: int = 256
    result_bytes: int = 256 * MIB
    result_count: int = 1024
    cache_bytes: int = 32 * MIB
    text_bytes: int = MIB
    terminal_bytes: int = 8 * MIB
    cache_entry_bytes: int = 8 * MIB
    log_count: int = 250
    log_line_bytes: int = 4 * KIB
    log_bytes: int = MIB
    job_count: int = 32
    job_ttl_seconds: float = 1800.0
    cleanup_seconds: float = 30.0
    subscriber_count: int = 4
    send_timeout_seconds: float = 10.0


DEFAULT_LIMITS = WebLimits()

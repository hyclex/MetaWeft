"""Shared retry defaults for launchers and standalone clients."""

import json
import math
import os
from pathlib import Path


def validate_retry(max_retries, retry_delay_seconds):
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    if (type(retry_delay_seconds) not in (int, float)
            or not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0):
        raise ValueError("retry_delay_seconds must be a finite nonnegative number")
    return {"max_retries": max_retries, "retry_delay_seconds": retry_delay_seconds}


def retry_defaults():
    values = json.loads(Path(__file__).with_name("retry.json").read_text())
    return validate_retry(**values)


CACHE_TTL_SECONDS = 1800


def cache_settings(cache_mode=None, cache_path=None, *, workspace=None):
    """Resolve once in launchers; captured workers receive these explicit values."""
    mode = cache_mode if cache_mode is not None else os.environ.get("METAWEFT_LLM_CACHE_MODE", "no_cache")
    if mode not in ("no_cache", "cache", "timed_cache"):
        raise ValueError("cache_mode must be no_cache, cache, or timed_cache")
    path = cache_path if cache_path is not None else os.environ.get("METAWEFT_LLM_CACHE_PATH")
    root = Path.cwd() if workspace is None else Path(workspace)
    path = Path(path).expanduser() if path is not None else root / "raw_logs/.cache/metaweft/llm.sqlite3"
    if not path.is_absolute():
        path = root / path
    return {"cache_mode": mode, "cache_path": str(path.resolve()), "cache_ttl_seconds": CACHE_TTL_SECONDS}

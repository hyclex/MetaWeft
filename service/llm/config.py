"""Shared retry defaults for launchers and standalone clients."""

import json
import math
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

from __future__ import annotations

import math
from typing import Any, Dict


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    rough = math.ceil(ascii_chars / 4.0) + non_ascii
    return max(1, rough)


def _get_attr(obj: Any, key: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    try:
        return int(value)
    except Exception:
        return default


def coerce_usage(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    input_tokens = _get_attr(usage, "input_tokens", 0)
    output_tokens = _get_attr(usage, "output_tokens", 0)
    if input_tokens == 0:
        input_tokens = _get_attr(usage, "prompt_tokens", 0)
    if output_tokens == 0:
        output_tokens = _get_attr(usage, "completion_tokens", 0)
    cached_input_tokens = 0
    details = None
    if isinstance(usage, dict):
        details = usage.get("input_tokens_details")
    else:
        details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached_input_tokens = _get_attr(details, "cached_tokens", 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cached_input_tokens": int(cached_input_tokens or 0),
    }

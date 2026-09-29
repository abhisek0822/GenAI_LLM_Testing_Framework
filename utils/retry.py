"""Small retry helper for transient API calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(operation: Callable[[], T], attempts: int = 3, delay_seconds: float = 0.2) -> T:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - intentionally generic utility
            last_error = exc
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error

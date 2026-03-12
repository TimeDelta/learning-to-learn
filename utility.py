from __future__ import annotations

import logging
import random
import string
import time
from contextlib import contextmanager
from typing import Iterator


def generate_random_string(length):
    characters = string.ascii_letters
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


class TimingScope:
    """Holds contextual metadata for log_timing callers."""

    __slots__ = ("details",)

    def __init__(self) -> None:
        self.details: str | None = None

    def set_details(self, details: str) -> None:
        self.details = details


@contextmanager
def log_timing(
    logger: logging.Logger, label: str, level: int = logging.INFO, *, log_on_start: bool = False
) -> Iterator[TimingScope]:
    """Context manager that logs how long the wrapped block takes to run."""

    scope = TimingScope()
    start = time.perf_counter()
    if log_on_start:
        logger.log(level, "Starting %s", label)
    try:
        yield scope
    finally:
        duration = time.perf_counter() - start
        suffix = f" ({scope.details})" if scope.details else ""
        logger.log(level, "%s completed in %.3fs%s", label, duration, suffix)


__all__ = ["generate_random_string", "log_timing", "TimingScope"]

from __future__ import annotations

import logging
import os
import random
import string
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # Resource is not available on Windows.
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    resource = None

try:  # Optional, so keep import local to avoid hard dependency.
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - only used when installed
    psutil = None

try:  # tracemalloc is part of the stdlib but may be disabled in embedded builds.
    import tracemalloc
except ImportError:  # pragma: no cover - extremely uncommon
    tracemalloc = None


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


class MemoryUsageTracker:
    """Lightweight helper that records RSS + tracemalloc hotspots when enabled."""

    __slots__ = ("logger", "enabled", "top_stats", "_ps", "_filters", "_cwd")

    def __init__(self, logger: logging.Logger, *, enabled: bool = False, top_stats: int = 8) -> None:
        self.logger = logger
        self.enabled = bool(enabled and tracemalloc is not None)
        self.top_stats = max(1, int(top_stats))
        self._ps = None
        self._filters = None
        self._cwd = str(Path.cwd())
        if not self.enabled:
            return
        if tracemalloc is not None and not tracemalloc.is_tracing():
            tracemalloc.start(50)
        if psutil is not None:
            try:
                self._ps = psutil.Process(os.getpid())
            except Exception:  # pragma: no cover - psutil edge cases
                self._ps = None
        if tracemalloc is not None:
            # Focus on repository files to avoid noise from site-packages.
            repo_filter = tracemalloc.Filter(True, f"{self._cwd}/*")
            stdlib_filter = tracemalloc.Filter(False, "*/site-packages/*")
            self._filters = (repo_filter, stdlib_filter)

    def _rss_bytes(self) -> int | None:
        if not self.enabled:
            return None
        if self._ps is not None:
            try:
                return int(self._ps.memory_info().rss)
            except Exception:  # pragma: no cover - psutil edge cases
                pass
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = getattr(usage, "ru_maxrss", 0)
        if rss <= 0:
            return None
        # BSD (macOS) reports bytes, Linux reports kilobytes.
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024

    def snapshot(self, label: str, *, top_stats: int | None = None) -> None:
        if not self.enabled or tracemalloc is None:
            return
        current, peak = tracemalloc.get_traced_memory()
        rss_bytes = self._rss_bytes()
        rss_mib = f"{rss_bytes / (1024 ** 2):.1f} MiB" if rss_bytes is not None else "n/a"
        self.logger.info(
            "[memory] %s rss=%s traced=%.1f/%.1f MiB",
            label,
            rss_mib,
            current / (1024**2),
            peak / (1024**2),
        )
        snapshot = tracemalloc.take_snapshot()
        if self._filters is not None:
            snapshot = snapshot.filter_traces(self._filters)
        stats = snapshot.statistics("lineno")
        limit = top_stats or self.top_stats
        for index, stat in enumerate(stats[:limit], start=1):
            frame = stat.traceback[0]
            size_mib = stat.size / (1024**2)
            self.logger.info(
                "[memory] #%d %s:%d %.2f MiB (%d refs)",
                index,
                frame.filename.replace(self._cwd + os.sep, ""),
                frame.lineno,
                size_mib,
                stat.count,
            )


__all__ = ["generate_random_string", "log_timing", "TimingScope", "MemoryUsageTracker"]

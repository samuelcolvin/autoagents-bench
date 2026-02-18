"""Shared performance monitoring helpers for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import resource
import sys
import threading
import time

import psutil


@dataclass(slots=True)
class PerformanceStats:
    cpu_usage_percent: float
    memory_peak_mb: float


class PerformanceMonitor:
    """Measure CPU usage (process) and peak RSS during a benchmark run."""

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._start_cpu: float | None = None
        self._peak_rss: int = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._start_cpu = _cpu_time_seconds()
        self._peak_rss = _current_rss_bytes()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_rss, daemon=True)
        self._thread.start()

    def stop(self) -> PerformanceStats:
        if self._start_time is None or self._start_cpu is None:
            raise RuntimeError("PerformanceMonitor.stop() called before start()")

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        wall_time = time.perf_counter() - self._start_time
        end_cpu = _cpu_time_seconds()
        cpu_time = max(0.0, end_cpu - self._start_cpu)
        cpu_usage_percent = (cpu_time / wall_time * 100.0) if wall_time > 0 else 0.0

        return PerformanceStats(
            cpu_usage_percent=cpu_usage_percent,
            memory_peak_mb=self._peak_rss / (1024 * 1024),
        )

    def _sample_rss(self) -> None:
        proc = psutil.Process()
        while not self._stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass
            time.sleep(0.05)


def _cpu_time_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _current_rss_bytes() -> int:
    try:
        return psutil.Process().memory_info().rss
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage * 1024)

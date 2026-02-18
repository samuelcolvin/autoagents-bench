"""Shared timing helpers for benchmarks."""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Tuple

# Context variable used by each async worker to tag its LLM calls.
llm_request_id: contextvars.ContextVar[int] = contextvars.ContextVar(
    "llm_request_id", default=-1
)


class TimingRecorder:
    """Thread-safe recorder for durations (in seconds)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: List[float] = []

    def reset(self) -> None:
        with self._lock:
            self._durations.clear()

    def record(self, duration: float) -> None:
        if duration < 0:
            return
        with self._lock:
            self._durations.append(duration)

    def snapshot(self) -> List[float]:
        with self._lock:
            return list(self._durations)


class LlmTimingRecorder:
    """Thread-safe recorder that tracks LLM call durations per request.

    Each call to ``record`` appends the duration to a flat list **and** to a
    per-request bucket keyed by the current ``llm_request_id`` context-var.
    ``snapshot_request_totals`` returns the total LLM time for each request
    that recorded at least one call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: List[float] = []
        self._per_request: Dict[int, float] = {}

    def reset(self) -> None:
        with self._lock:
            self._durations.clear()
            self._per_request.clear()

    def record(self, duration: float) -> None:
        if duration < 0:
            return
        rid = llm_request_id.get()
        with self._lock:
            self._durations.append(duration)
            self._per_request[rid] = self._per_request.get(rid, 0.0) + duration

    def snapshot(self) -> List[float]:
        with self._lock:
            return list(self._durations)

    def snapshot_request_totals(self) -> List[float]:
        """Return one total-LLM-duration per request that was recorded."""
        with self._lock:
            return list(self._per_request.values())


# ---------------------------------------------------------------------------
# OpenAI monkey-patching
# ---------------------------------------------------------------------------
_original_sync_create = None
_original_async_create = None


class _TimedAsyncStream:
    """Wraps an AsyncStream so timing is recorded when the stream is fully consumed."""

    def __init__(self, stream: Any, start: float, recorder: "LlmTimingRecorder") -> None:
        self._stream = stream
        self._start = start
        self._recorder = recorder
        self._done = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._stream:
                yield chunk
        finally:
            if not self._done:
                self._done = True
                self._recorder.record(time.perf_counter() - self._start)

    # Proxy attribute access so callers that inspect the stream object still work
    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aenter__(self) -> "_TimedAsyncStream":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if not self._done:
            self._done = True
            self._recorder.record(time.perf_counter() - self._start)
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)


def patch_openai_timing(recorder: LlmTimingRecorder) -> None:
    """Monkey-patch the OpenAI SDK so every chat-completion records its wall-clock duration.

    For non-streaming calls the duration is recorded when ``create`` returns.
    For streaming calls the duration is recorded when the stream is fully consumed,
    which is the correct measure of total LLM response time.
    """
    global _original_sync_create, _original_async_create

    import openai.resources.chat.completions as _sync_mod

    _original_sync_create = _sync_mod.Completions.create

    def timed_sync_create(self, *args, **kwargs):
        start = time.perf_counter()
        result = _original_sync_create(self, *args, **kwargs)
        # For non-streaming sync calls the result is a ChatCompletion — record now.
        # Streaming sync calls are not used by any framework in this benchmark.
        recorder.record(time.perf_counter() - start)
        return result

    _sync_mod.Completions.create = timed_sync_create  # type: ignore[assignment]

    import openai.resources.chat.completions as _async_mod

    _original_async_create = _async_mod.AsyncCompletions.create

    async def timed_async_create(self, *args, **kwargs):
        start = time.perf_counter()
        result = await _original_async_create(self, *args, **kwargs)
        # If streaming, wrap the returned AsyncStream so timing is recorded
        # when the last chunk is consumed rather than when the stream opens.
        if kwargs.get("stream", False):
            return _TimedAsyncStream(result, start, recorder)
        recorder.record(time.perf_counter() - start)
        return result

    _async_mod.AsyncCompletions.create = timed_async_create  # type: ignore[assignment]


def unpatch_openai_timing() -> None:
    """Restore original OpenAI SDK methods."""
    global _original_sync_create, _original_async_create

    if _original_sync_create is not None:
        import openai.resources.chat.completions as _sync_mod

        _sync_mod.Completions.create = _original_sync_create  # type: ignore[assignment]
        _original_sync_create = None

    if _original_async_create is not None:
        import openai.resources.chat.completions as _async_mod

        _async_mod.AsyncCompletions.create = _original_async_create  # type: ignore[assignment]
        _original_async_create = None

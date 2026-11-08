"""Opt-in, privacy-conscious request traces for the local chat API."""

from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

ENABLE_ENV = "VANI_ENABLE_OBSERVABILITY"
TRACE_PATH = Path(__file__).resolve().parent.parent / "data" / "observability" / "chat_traces.jsonl"
_WRITE_LOCK = threading.Lock()
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def _safe_model(value: str | None) -> str | None:
    model = (value or "").strip()
    return model if _SAFE_MODEL.fullmatch(model) else None


def _token_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    """Read only token counts explicitly supplied by the SDK response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    prompt = _token_value(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None)))
    completion = _token_value(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", None)))
    total = _token_value(getattr(usage, "total_tokens", None))
    return prompt, completion, total


class ChatTrace:
    """Mutable metadata for one request; never stores user or document content."""

    def __init__(self, model: str | None) -> None:
        self.enabled = os.getenv(ENABLE_ENV) == "1"
        self.started = time.perf_counter_ns()
        self.timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        self.request_id = str(uuid4())
        self.model = _safe_model(model)
        self.retrieval_duration_ms: float | None = None
        self.llm_duration_ms: float | None = None
        self.retrieval_path = "none"
        self.retrieved_chunk_count = 0
        self.citations_returned = 0
        self.used_fallback = False
        self.fallback_reason: str | None = None
        self.outcome = "success"
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None

    def record_retrieval(self, path: str, chunk_count: int, duration_ms: float) -> None:
        self.retrieval_path = path if path in {"vector", "first_page", "first_page_plus_vector", "none"} else "none"
        self.retrieved_chunk_count = max(0, int(chunk_count))
        self.retrieval_duration_ms = round(max(0.0, duration_ms), 3)

    def record_llm(self, duration_ms: float, response: Any = None) -> None:
        self.llm_duration_ms = round(max(0.0, duration_ms), 3)
        if response is not None:
            self.prompt_tokens, self.completion_tokens, self.total_tokens = token_usage(response)

    def mark_fallback(self, reason: str) -> None:
        self.used_fallback = True
        self.fallback_reason = reason
        self.outcome = "fallback"

    def mark_error(self, reason: str) -> None:
        self.fallback_reason = reason
        self.outcome = "error"

    def mark_success(self, citations_returned: int) -> None:
        self.citations_returned = max(0, int(citations_returned))

    def _record(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp,
            "request_id": self.request_id,
            "overall_duration_ms": round(max(0, time.perf_counter_ns() - self.started) / 1_000_000, 3),
            "retrieval_duration_ms": self.retrieval_duration_ms,
            "llm_duration_ms": self.llm_duration_ms,
            "retrieval_path": self.retrieval_path,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "citations_returned": self.citations_returned,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "outcome": self.outcome,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def write(self) -> None:
        if not self.enabled:
            return
        try:
            line = json.dumps(self._record(), ensure_ascii=True, separators=(",", ":")) + "\n"
            with _WRITE_LOCK:
                TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with TRACE_PATH.open("a", encoding="utf-8", newline="\n") as trace_file:
                    trace_file.write(line)
                    trace_file.flush()
        except Exception:
            # Tracing is best-effort. Avoid logging exception messages or request data.
            return


_CURRENT_TRACE: contextvars.ContextVar[ChatTrace | None] = contextvars.ContextVar(
    "vani_chat_trace", default=None
)


@contextmanager
def chat_request(model: str | None = None) -> Iterator[ChatTrace | None]:
    """Collect and persist one trace only when the opt-in flag is enabled."""
    trace = ChatTrace(model)
    if not trace.enabled:
        yield None
        return
    context_token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    except BaseException:
        trace.mark_error(trace.fallback_reason or "other")
        raise
    finally:
        _CURRENT_TRACE.reset(context_token)
        trace.write()


def _active() -> ChatTrace | None:
    trace = _CURRENT_TRACE.get()
    return trace if trace is not None and trace.enabled else None


def record_retrieval(path: str, chunk_count: int, duration_ms: float) -> None:
    trace = _active()
    if trace:
        trace.record_retrieval(path, chunk_count, duration_ms)


def record_llm(duration_ms: float, response: Any = None) -> None:
    trace = _active()
    if trace:
        trace.record_llm(duration_ms, response)


def mark_fallback(reason: str) -> None:
    trace = _active()
    if trace:
        trace.mark_fallback(reason)


def mark_error(reason: str) -> None:
    trace = _active()
    if trace:
        trace.mark_error(reason)


def mark_success(citations_returned: int) -> None:
    trace = _active()
    if trace:
        trace.mark_success(citations_returned)

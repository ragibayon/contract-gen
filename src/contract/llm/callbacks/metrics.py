from __future__ import annotations

import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from langchain_core.callbacks import BaseCallbackHandler

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int_from_paths(
    raw: Mapping[str, Any], paths: Iterable[tuple[str, ...]]
) -> Optional[int]:
    for path in paths:
        current: Any = raw
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        parsed = _to_int(current)
        if parsed is not None:
            return parsed
    return None


def _safe_retry_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = _to_int(value)
    if parsed is None:
        return default
    return max(1, parsed)


def _safe_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = _to_float(value)
    if parsed is None:
        return default
    return max(0.0, parsed)


def _is_retryable_llm_error(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in (
            "overloaded_error",
            "overloaded",
            "error code: 529",
            "'code': 529",
            '"code": 529',
            "too many requests",
            "rate limit",
            "ratelimit",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    )


def _append_raw(current: Any, incoming: Any) -> Any:
    if incoming is None:
        return current
    incoming = deepcopy(incoming)
    if current is None:
        return incoming
    if isinstance(current, list):
        current.append(incoming)
        return current
    return [deepcopy(current), incoming]


def normalize_token_usage(usage: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(usage, Mapping):
        return None

    raw = dict(usage)

    input_tokens = _to_int(raw.get("input_tokens"))
    output_tokens = _to_int(raw.get("output_tokens"))
    total_tokens = _to_int(raw.get("total_tokens"))

    if input_tokens is None:
        input_tokens = _to_int(raw.get("prompt_tokens"))
    if output_tokens is None:
        output_tokens = _to_int(raw.get("completion_tokens"))

    if input_tokens is None:
        input_tokens = _to_int(raw.get("prompt_eval_count"))
    if output_tokens is None:
        output_tokens = _to_int(raw.get("eval_count"))

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    reasoning_tokens = _first_int_from_paths(
        raw,
        (
            ("reasoning_tokens",),
            ("reasoning_output_tokens",),
            ("completion_tokens_details", "reasoning_tokens"),
            ("completion_tokens_details", "reasoning"),
            ("completion_token_details", "reasoning_tokens"),
            ("completion_token_details", "reasoning"),
            ("output_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning"),
            ("output_token_details", "reasoning_tokens"),
            ("output_token_details", "reasoning"),
        ),
    )

    cached_tokens = _first_int_from_paths(
        raw,
        (
            ("cached_tokens",),
            ("prompt_tokens_details", "cached_tokens"),
            ("prompt_token_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
            ("input_token_details", "cached_tokens"),
        ),
    )

    cache_read_input_tokens = _first_int_from_paths(
        raw,
        (
            ("cache_read_input_tokens",),
            ("cache_read_tokens",),
            ("input_tokens_details", "cache_read"),
            ("input_token_details", "cache_read"),
            ("input_tokens_details", "cache_read_input_tokens"),
            ("input_token_details", "cache_read_input_tokens"),
        ),
    )

    cache_creation_input_tokens = _first_int_from_paths(
        raw,
        (
            ("cache_creation_input_tokens",),
            ("cache_creation_tokens",),
            ("input_tokens_details", "cache_creation"),
            ("input_token_details", "cache_creation"),
            ("input_tokens_details", "cache_creation_input_tokens"),
            ("input_token_details", "cache_creation_input_tokens"),
        ),
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "raw": deepcopy(raw),
    }


def _normalized_usage_candidates(
    container: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    seen_ids: set[int] = set()

    for key in ("token_usage", "usage", "usage_metadata"):
        candidate = container.get(key)
        if isinstance(candidate, Mapping):
            seen_ids.add(id(candidate))
            normalized = normalize_token_usage(candidate)
            if normalized is not None:
                yield normalized

    if id(container) not in seen_ids:
        normalized = normalize_token_usage(container)
        if normalized is not None:
            yield normalized


class LLMRunMetrics(BaseCallbackHandler):
    """Collect latency and token usage for one or more LangChain calls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.latency: Optional[float] = None
        self.token_usage: Optional[Dict[str, Any]] = None
        self.model_name: Optional[str] = None
        self.call_count: int = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        self.call_count += 1

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        self.call_count += 1

    def _merge_token_usage(self, usage: Dict[str, Any]) -> None:
        if self.token_usage is None:
            self.token_usage = deepcopy(usage)
            self.token_usage["calls"] = 1
            return

        merged = deepcopy(self.token_usage)

        for key in TOKEN_FIELDS:
            current = merged.get(key)
            incoming = usage.get(key)
            if current is None:
                merged[key] = incoming
            elif incoming is None:
                continue
            else:
                merged[key] = current + incoming

        merged["raw"] = _append_raw(merged.get("raw"), usage.get("raw"))
        merged["calls"] = int(merged.get("calls", 1)) + 1
        self.token_usage = merged

    def _fill_missing_token_details(self, usage: Dict[str, Any]) -> None:
        if self.token_usage is None:
            self._merge_token_usage(usage)
            return

        merged = deepcopy(self.token_usage)

        for key in TOKEN_FIELDS:
            if merged.get(key) is None and usage.get(key) is not None:
                merged[key] = usage[key]

        merged["raw"] = _append_raw(merged.get("raw"), usage.get("raw"))
        self.token_usage = merged

    def _capture_usage_candidates(
        self,
        candidates: Iterable[Dict[str, Any] | None],
        *,
        secondary_for_same_call: bool,
    ) -> bool:
        captured = False

        for usage in candidates:
            if usage is None:
                continue

            if not captured and not secondary_for_same_call:
                self._merge_token_usage(usage)
            else:
                self._fill_missing_token_details(usage)

            captured = True

        return captured

    def capture_from_message(self, msg: Any) -> bool:
        if msg is None:
            return False

        captured = False

        usage_metadata = getattr(msg, "usage_metadata", None)
        normalized = normalize_token_usage(usage_metadata)
        if normalized is not None:
            captured = self._capture_usage_candidates(
                [normalized],
                secondary_for_same_call=False,
            )

        response_metadata = getattr(msg, "response_metadata", None)
        if isinstance(response_metadata, Mapping):
            if self.model_name is None:
                self.model_name = response_metadata.get(
                    "model_name"
                ) or response_metadata.get("model")

            captured_response = self._capture_usage_candidates(
                _normalized_usage_candidates(response_metadata),
                secondary_for_same_call=captured,
            )
            captured = captured or captured_response

        return captured

    def on_llm_end(self, response, **kwargs) -> None:
        self.end_time = time.time()
        if self.start_time is not None:
            self.latency = self.end_time - self.start_time

        captured_from_message = False
        try:
            generation = response.generations[0][0]
            captured_from_message = self.capture_from_message(generation.message)
        except Exception:
            pass

        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, Mapping):
            captured_llm_output = self._capture_usage_candidates(
                _normalized_usage_candidates(llm_output),
                secondary_for_same_call=captured_from_message,
            )
            captured_from_message = captured_from_message or captured_llm_output

            if self.model_name is None:
                self.model_name = llm_output.get("model_name") or llm_output.get(
                    "model"
                )

    def on_llm_error(self, error, **kwargs) -> None:
        self.end_time = time.time()
        if self.start_time is not None:
            self.latency = self.end_time - self.start_time

    def as_dict(self) -> Dict[str, Any]:
        return {
            "latency_seconds": (
                round(self.latency, 3) if self.latency is not None else None
            ),
            "token_usage": deepcopy(self.token_usage),
            "model_name": self.model_name,
            "call_count": self.call_count,
        }


def merge_llm_metrics(
    metric_items: Iterable[Mapping[str, Any] | None],
) -> Dict[str, Any]:
    attempts = [deepcopy(dict(item)) for item in metric_items if item]

    token_sums = {key: 0 for key in TOKEN_FIELDS}
    token_seen = {key: False for key in TOKEN_FIELDS}
    raw_items: list[Any] = []
    calls = 0
    saw_usage = False

    latency_seconds = 0.0
    call_count = 0
    model_names: list[str] = []

    for item in attempts:
        latency = _to_float(item.get("latency_seconds"))
        if latency is not None:
            latency_seconds += latency

        call_count += int(item.get("call_count") or 0)

        model_name = item.get("model_name")
        if model_name and model_name not in model_names:
            model_names.append(str(model_name))

        usage = item.get("token_usage")
        if not isinstance(usage, Mapping):
            continue

        saw_usage = True

        usage_calls = _to_int(usage.get("calls"))
        if usage_calls is not None:
            calls += usage_calls

        for key in TOKEN_FIELDS:
            value = _to_int(usage.get(key))
            if value is not None:
                token_sums[key] += value
                token_seen[key] = True

        raw = usage.get("raw")
        if isinstance(raw, list):
            raw_items.extend(deepcopy(raw))
        elif raw is not None:
            raw_items.append(deepcopy(raw))

    token_usage: Optional[Dict[str, Any]]
    if saw_usage:
        token_usage = {
            key: token_sums[key] if token_seen[key] else None for key in TOKEN_FIELDS
        }
        token_usage["calls"] = calls
        token_usage["raw"] = raw_items if raw_items else None
    else:
        token_usage = None

    return {
        "attempts": attempts,
        "latency_seconds": round(latency_seconds, 3),
        "call_count": call_count,
        "model_names": model_names,
        "token_usage": token_usage,
    }


def invoke_with_metrics(
    runnable: Any,
    payload: Any,
    metrics: LLMRunMetrics | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    metrics = metrics or LLMRunMetrics()
    invoke_config: Dict[str, Any] = dict(config or {})
    callbacks = list(invoke_config.get("callbacks", []))
    callbacks.append(metrics)
    invoke_config["callbacks"] = callbacks

    max_attempts = _safe_retry_env("LLM_RETRY_BACKOFF_MAX_ATTEMPTS", 3)
    base_delay_seconds = _safe_float_env("LLM_RETRY_BACKOFF_BASE_SECONDS", 1.0)
    max_delay_seconds = _safe_float_env("LLM_RETRY_BACKOFF_MAX_SECONDS", 30.0)
    jitter_ratio = min(1.0, _safe_float_env("LLM_RETRY_BACKOFF_JITTER_RATIO", 0.2))

    attempt = 0
    while True:
        try:
            response = runnable.invoke(payload, config=invoke_config, **kwargs)
            return response, metrics
        except Exception as exc:
            attempt += 1
            if attempt >= max_attempts or not _is_retryable_llm_error(exc):
                raise
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            if jitter_ratio > 0:
                delay += random.uniform(0.0, delay * jitter_ratio)
            time.sleep(delay)

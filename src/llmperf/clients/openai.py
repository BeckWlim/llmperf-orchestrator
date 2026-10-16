import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from llmperf.clients.base import LLMClient, LLMResponse
from llmperf.clients.models import RequestConfig
from llmperf.metrics import constants as common_metrics
from llmperf.metrics.usage import normalize_usage


class OpenAIStreamError(RuntimeError):
    def __init__(self, message: str, code: Any = -1):
        super().__init__(message)
        self.code = code


class StreamInactivityTimeout(TimeoutError):
    """Raised when a stream produces no text for the configured interval."""


# ``requests`` may wait for the requested byte count on non-chunked responses.
# One-byte reads ensure a complete SSE line is yielded as soon as its newline arrives,
# so the text-progress timer is based on application data rather than buffer filling.
SSE_ITER_CHUNK_SIZE = 1


def decode_sse_line(chunk: bytes) -> Dict[str, Any]:
    """Decode one OpenAI-compatible SSE line without assuming a text choice."""

    line = chunk.strip()
    if not line or line.startswith(b":") or not line.startswith(b"data:"):
        return {"kind": "ignore"}
    payload = line[len(b"data:") :].strip()
    if payload == b"[DONE]":
        return {"kind": "done"}
    if not payload:
        return {"kind": "ignore"}

    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("SSE data payload must be a JSON object")
    provider_error = document.get("error")
    if provider_error:
        if isinstance(provider_error, dict):
            message = str(provider_error.get("message") or provider_error)
            code = provider_error.get("code", -1)
        else:
            message = str(provider_error)
            code = -1
        raise OpenAIStreamError(message, code)

    usage = document.get("usage")
    event_usage = usage if isinstance(usage, dict) else None
    choices = document.get("choices") or []
    if not choices:
        event: Dict[str, Any] = {"kind": "metadata"}
        if event_usage is not None:
            event["usage"] = event_usage
        return event
    if not isinstance(choices, list) or not isinstance(choices[0], dict):
        raise ValueError("SSE choices must be a list of objects")
    delta = choices[0].get("delta") or {}
    if not isinstance(delta, dict):
        raise ValueError("SSE choice delta must be an object")

    segments = []
    for field in ("reasoning_content", "content"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            segments.append(value)
    if not segments:
        event = {"kind": "metadata"}
        if event_usage is not None:
            event["usage"] = event_usage
        return event
    event = {"kind": "text", "text": "".join(segments)}
    if event_usage is not None:
        event["usage"] = event_usage
    return event


def cache_metrics_from_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Project typed usage normalization into canonical cache metrics."""

    normalized = normalize_usage(usage)
    metrics = normalized.to_metrics()
    metrics.pop(common_metrics.NORMALIZED_USAGE, None)
    metrics.pop(common_metrics.RAW_USAGE, None)
    metrics.pop(common_metrics.PROVIDER_INPUT_TOKENS, None)
    metrics.pop(common_metrics.PROVIDER_OUTPUT_TOKENS, None)
    return metrics


SAFE_RESPONSE_HEADERS = {
    "request-id",
    "server-timing",
    "trace-id",
    "x-request-id",
    "x-trace-id",
}


def _safe_response_headers(headers: Any) -> Dict[str, str]:
    return {
        str(name).lower(): str(value)[:2048]
        for name, value in headers.items()
        if str(name).lower() in SAFE_RESPONSE_HEADERS
    }


class OpenAIChatCompletionsClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> LLMResponse:
        prompt_text, prompt_len = request_config.prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_text},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        if body.get("stream_options") is None:
            body.pop("stream_options", None)
        inter_chunk_latencies = []
        ttft = 0.0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        total_request_time = 0.0
        usage: Dict[str, Any] = {}
        request_metadata = dict(request_config.metadata or {})
        response_headers: Dict[str, str] = {}
        response_headers_time = None
        first_sse_time = None
        first_text_time = None
        last_text_time = None
        completion_time = None

        metrics: Dict[str, Any] = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        start_utc = datetime.now(timezone.utc).isoformat()
        completed_utc = None
        address = self.environment.get("OPENAI_API_BASE")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = self.environment.get("OPENAI_API_KEY", "").strip()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        request_url = address.rstrip("/") + "/chat/completions"
        stream_idle_timeout = float(request_config.timeout_seconds or 180)
        transport_read_timeout = stream_idle_timeout + max(
            1.0, stream_idle_timeout * 0.1
        )
        inactivity_expired = threading.Event()
        request_finished = threading.Event()
        progress_changed = threading.Event()
        progress_lock = threading.Lock()
        last_progress_time = [start_time]
        response_holder: list[Optional[requests.Response]] = [None]

        def watch_stream_progress() -> None:
            while not request_finished.is_set():
                progress_changed.clear()
                with progress_lock:
                    remaining = stream_idle_timeout - (
                        time.monotonic() - last_progress_time[0]
                    )
                if remaining <= 0:
                    inactivity_expired.set()
                    with progress_lock:
                        response = response_holder[0]
                    if response is not None:
                        try:
                            response.close()
                        except Exception:
                            pass
                    return
                progress_changed.wait(remaining)

        progress_watcher = threading.Thread(target=watch_stream_progress, daemon=True)
        progress_watcher.start()
        try:
            with requests.post(
                request_url,
                json=body,
                stream=True,
                timeout=(stream_idle_timeout, transport_read_timeout),
                headers=headers,
            ) as response:
                with progress_lock:
                    response_holder[0] = response
                if inactivity_expired.is_set():
                    raise StreamInactivityTimeout(
                        "stream produced no text for "
                        f"{stream_idle_timeout:g} seconds"
                    )
                response_headers_time = time.monotonic()
                response_headers = _safe_response_headers(response.headers)
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=SSE_ITER_CHUNK_SIZE):
                    received_at = time.monotonic()
                    if inactivity_expired.is_set():
                        raise StreamInactivityTimeout(
                            "stream produced no text for "
                            f"{stream_idle_timeout:g} seconds"
                        )
                    if chunk and first_sse_time is None:
                        first_sse_time = received_at
                    event = decode_sse_line(chunk)
                    usage.update(event.get("usage") or {})
                    if event["kind"] == "metadata":
                        continue
                    if event["kind"] == "ignore":
                        continue
                    if event["kind"] == "done":
                        break
                    text = event["text"]
                    with progress_lock:
                        last_progress_time[0] = received_at
                    progress_changed.set()
                    if last_text_time is None:
                        first_text_time = received_at
                        ttft = first_text_time - start_time
                    else:
                        inter_chunk_latencies.append(received_at - last_text_time)
                    last_text_time = received_at
                    generated_text += text

            if inactivity_expired.is_set():
                raise StreamInactivityTimeout(
                    "stream produced no text for " f"{stream_idle_timeout:g} seconds"
                )

            if not generated_text:
                raise ValueError("Stream completed without text content")

        except OpenAIStreamError as exc:
            error_response_code = exc.code
            if not error_msg:
                error_msg = str(exc)
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {exc}")
            print(error_response_code)
        except Exception as exc:
            if inactivity_expired.is_set() and not isinstance(
                exc, StreamInactivityTimeout
            ):
                effective_error = StreamInactivityTimeout(
                    "stream produced no text for " f"{stream_idle_timeout:g} seconds"
                )
            else:
                effective_error = exc
            if not error_msg:
                error_msg = f"{type(effective_error).__name__}: {effective_error}"
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {error_msg}")
            print(error_response_code)
        finally:
            request_finished.set()
            progress_changed.set()
            with progress_lock:
                response_holder[0] = None
            progress_watcher.join(timeout=0.1)
            completion_time = time.monotonic()
            completed_utc = datetime.now(timezone.utc).isoformat()
            total_request_time = completion_time - start_time

        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = None
        metrics[common_metrics.NUM_TOTAL_TOKENS] = None
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = None
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
        metrics[common_metrics.LOCAL_INPUT_TOKENS] = prompt_len
        metrics[common_metrics.REQUEST_METADATA] = request_metadata
        metrics[common_metrics.RESPONSE_HEADERS] = response_headers
        metrics[common_metrics.RESPONSE_HEADER_LAT] = (
            response_headers_time - start_time
            if response_headers_time is not None
            else None
        )
        metrics[common_metrics.FIRST_SSE_LAT] = (
            first_sse_time - start_time if first_sse_time is not None else None
        )
        metrics[common_metrics.INTER_SSE_CHUNK_LAT] = inter_chunk_latencies
        metrics[common_metrics.REQUEST_TIMING] = {
            "client_start_monotonic": start_time,
            "response_headers_monotonic": response_headers_time,
            "first_sse_monotonic": first_sse_time,
            "first_text_monotonic": first_text_time,
            "last_text_monotonic": last_text_time,
            "completed_monotonic": completion_time,
            "client_start_utc": start_utc,
            "completed_utc": completed_utc,
        }
        metrics[common_metrics.STREAM_TIMING_SEMANTICS] = {
            "inter_sse_chunk_latency": "time_between_text-bearing_sse_events",
            "ttft_in_decode_intervals": False,
            "timeout": "maximum_seconds_without_text-bearing_sse_event",
        }
        if usage:
            normalized_usage = normalize_usage(usage)
            metrics.update(normalized_usage.to_metrics())
            output_tokens = normalized_usage.provider_output_tokens
            if output_tokens is not None:
                metrics[common_metrics.NUM_OUTPUT_TOKENS] = output_tokens
                metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                    output_tokens / total_request_time if total_request_time > 0 else 0
                )
                metrics[common_metrics.NUM_TOTAL_TOKENS] = prompt_len + output_tokens
            if (
                output_tokens is not None
                and output_tokens > 1
                and first_text_time is not None
                and completion_time is not None
            ):
                metrics[common_metrics.TPOT] = (completion_time - first_text_time) / (
                    output_tokens - 1
                )

        return metrics, generated_text, request_config

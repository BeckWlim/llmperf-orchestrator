import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from llmperf.usage import normalize_usage


class OpenAIStreamError(RuntimeError):
    def __init__(self, message: str, code: Any = -1):
        super().__init__(message)
        self.code = code


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
        event = {"kind": "metadata"}
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
    """Backward-compatible metric projection of typed usage normalization."""

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

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        inter_chunk_latencies = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        usage = {}
        request_metadata = dict(request_config.metadata or {})
        response_headers = {}
        response_headers_time = None
        first_sse_time = None
        first_text_time = None
        last_text_time = None
        completion_time = None

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        start_utc = datetime.now(timezone.utc).isoformat()
        completed_utc = None
        address = os.environ.get("OPENAI_API_BASE")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")
        headers = {"Authorization": f"Bearer {key}"}
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        request_timeout = request_config.timeout_seconds or 180
        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=request_timeout,
                headers=headers,
            ) as response:
                response_headers_time = time.monotonic()
                response_headers = _safe_response_headers(response.headers)
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=None):
                    received_at = time.monotonic()
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
                    tokens_received += 1
                    if first_text_time is None:
                        first_text_time = received_at
                        ttft = first_text_time - start_time
                    else:
                        inter_chunk_latencies.append(received_at - last_text_time)
                    last_text_time = received_at
                    generated_text += text

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
            if not error_msg:
                error_msg = f"{type(exc).__name__}: {exc}"
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {error_msg}")
            print(error_response_code)
        finally:
            completion_time = time.monotonic()
            completed_utc = datetime.now(timezone.utc).isoformat()
            total_request_time = completion_time - start_time
            output_throughput = (
                tokens_received / total_request_time if total_request_time > 0 else 0
            )

        # Retained for result-schema compatibility. This is based on SSE chunks,
        # not provider token boundaries; new consumers must use the explicitly
        # named inter_sse_chunk_latency_s and TPOT fields below.
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(inter_chunk_latencies)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
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
            "completed_monotonic": completion_time,
            "client_start_utc": start_utc,
            "completed_utc": completed_utc,
        }
        metrics[common_metrics.STREAM_TIMING_SEMANTICS] = {
            "legacy_inter_token_latency": "deprecated_inter_chunk_average",
            "inter_sse_chunk_latency": "time_between_text-bearing_sse_events",
            "ttft_in_decode_intervals": False,
        }
        if usage:
            normalized_usage = normalize_usage(usage)
            metrics.update(normalized_usage.to_metrics())
            output_tokens = normalized_usage.provider_output_tokens
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

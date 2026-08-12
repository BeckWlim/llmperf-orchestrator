import json
import os
import time
from typing import Any, Dict

import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


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

    choices = document.get("choices") or []
    if not choices:
        event = {"kind": "metadata"}
        usage = document.get("usage")
        if isinstance(usage, dict):
            event["usage"] = usage
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
        return {"kind": "metadata"}
    return {"kind": "text", "text": "".join(segments)}


def cache_metrics_from_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize cache token counters from OpenAI-compatible providers."""

    hit_tokens = usage.get("prompt_cache_hit_tokens")
    miss_tokens = usage.get("prompt_cache_miss_tokens")

    if hit_tokens is None:
        details = usage.get("prompt_tokens_details")
        total_tokens = usage.get("prompt_tokens")
        if not isinstance(details, dict):
            details = usage.get("input_tokens_details")
            total_tokens = usage.get("input_tokens")
        if isinstance(details, dict):
            hit_tokens = details.get("cached_tokens")
            if hit_tokens is not None and total_tokens is not None:
                miss_tokens = max(0, total_tokens - hit_tokens)

    if hit_tokens is None:
        return {}
    if miss_tokens is None:
        total_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        if total_tokens is None:
            return {common_metrics.KV_CACHE_HIT_TOKENS: hit_tokens}
        miss_tokens = max(0, total_tokens - hit_tokens)

    cacheable_tokens = hit_tokens + miss_tokens
    return {
        common_metrics.KV_CACHE_HIT_TOKENS: hit_tokens,
        common_metrics.KV_CACHE_MISS_TOKENS: miss_tokens,
        common_metrics.KV_CACHE_HIT_RATE: (
            hit_tokens / cacheable_tokens if cacheable_tokens else 0
        ),
    }


@ray.remote
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
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        usage = {}

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
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
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=None):
                    event = decode_sse_line(chunk)
                    if event["kind"] == "metadata":
                        usage.update(event.get("usage") or {})
                        continue
                    if event["kind"] == "ignore":
                        continue
                    if event["kind"] == "done":
                        break
                    text = event["text"]
                    tokens_received += 1
                    if not ttft:
                        ttft = time.monotonic() - start_time
                        time_to_next_token.append(ttft)
                    else:
                        time_to_next_token.append(
                            time.monotonic() - most_recent_received_token_time
                        )
                    most_recent_received_token_time = time.monotonic()
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
            total_request_time = time.monotonic() - start_time
            output_throughput = (
                tokens_received / total_request_time if total_request_time > 0 else 0
            )

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
        metrics.update(cache_metrics_from_usage(usage))

        return metrics, generated_text, request_config

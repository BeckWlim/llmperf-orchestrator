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
        return {"kind": "metadata"}
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
                    if event["kind"] in {"ignore", "metadata"}:
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

        return metrics, generated_text, request_config

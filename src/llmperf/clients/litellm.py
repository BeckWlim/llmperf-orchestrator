import time
from collections.abc import Mapping
from typing import Any, Dict

from llmperf.clients.base import LLMClient, LLMResponse
from llmperf.clients.models import RequestConfig
from llmperf.metrics import constants as common_metrics
from llmperf.metrics.usage import normalize_usage


class LiteLLMClient(LLMClient):
    """Client for LiteLLM Completions API."""

    def llm_request(self, request_config: RequestConfig) -> LLMResponse:
        # litellm package isn't serializable, so we import it within the function
        # to maintain compatibility with ray.
        from litellm import completion

        prompt_text, prompt_len = request_config.prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_text},
        ]
        assert (
            request_config.llm_api is not None
        ), "the request config's llm_api must be set."
        if request_config.llm_api == "litellm":
            model = request_config.model
        else:
            model = request_config.llm_api + "/" + request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        credential_prefix = (
            "ANTHROPIC" if request_config.llm_api == "anthropic" else "OPENAI"
        )
        api_key = self.environment.get(
            "LLMPERF_REQUEST_API_KEY"
        ) or self.environment.get(f"{credential_prefix}_API_KEY")
        api_base = self.environment.get(
            "LLMPERF_REQUEST_API_BASE"
        ) or self.environment.get(f"{credential_prefix}_API_BASE")
        if api_key:
            body["api_key"] = api_key
        if api_base:
            body["api_base"] = api_base

        time_to_next_token = []
        ttft = 0.0
        error_response_code = -1
        generated_text = ""
        total_request_time = 0.0

        metrics: Dict[str, Any] = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        try:
            start_time = time.monotonic()
            most_recent_received_token_time = time.monotonic()

            response = completion(**body)
            for tok in response:
                if isinstance(tok, tuple):
                    raise TypeError(
                        "LiteLLM returned a non-stream tuple while stream=True"
                    )
                chunk_document = tok.model_dump()
                raw_usage = chunk_document.get("usage")
                if isinstance(raw_usage, Mapping):
                    metrics.update(normalize_usage(raw_usage).to_metrics())
                if tok.choices and tok.choices[0].delta:
                    delta = tok.choices[0].delta
                    if delta.get("content", None):
                        if ttft == 0:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        generated_text += delta["content"]
                        most_recent_received_token_time = time.monotonic()

            total_request_time = time.monotonic() - start_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            metrics[common_metrics.ERROR_CODE] = error_response_code

            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_SSE_CHUNK_LAT] = time_to_next_token
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        output_tokens = metrics.get(common_metrics.PROVIDER_OUTPUT_TOKENS)
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
            output_tokens / total_request_time
            if output_tokens is not None and total_request_time > 0
            else None
        )
        metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            prompt_len + output_tokens if output_tokens is not None else None
        )
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = output_tokens
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
        return metrics, generated_text, request_config

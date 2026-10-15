import io
import json
import os
import time
from typing import Any, Dict

from boto3.session import Session

from llmperf.ray_llm_client import LLMClient, LLMResponse
from llmperf.models import RequestConfig
from llmperf import common_metrics


class SageMakerClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> LLMResponse:
        if not self.environment.get("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID must be set.")
        if not self.environment.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set.")
        if not self.environment.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt_text, prompt_len = request_config.prompt

        model = request_config.model
        # Each concurrent call owns its SDK session; avoid the global default
        # session's lazy initialization from multiple actor threads.
        sm_runtime = Session(
            aws_access_key_id=self.environment.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self.environment.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=self.environment.get("AWS_SESSION_TOKEN"),
        ).client(
            "sagemaker-runtime", region_name=self.environment.get("AWS_REGION_NAME")
        )

        sampling_params = dict(request_config.sampling_params or {})

        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]
        messages = sampling_params.pop(
            "messages",
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt_text},
            ],
        )

        request_body = {
            "inputs": [messages],
            "parameters": {
                **sampling_params,
            },
        }

        time_to_next_token = []
        ttft = 0.0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        total_request_time = 0.0
        metrics: Dict[str, Any] = {}

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            response = sm_runtime.invoke_endpoint_with_response_stream(
                EndpointName=model,
                ContentType="application/json",
                Body=json.dumps(request_body),
                CustomAttributes="accept_eula=true",
            )

            event_stream = response["Body"]
            json_byte = b""
            first_chunk_at = start_time
            for line, first_chunk_at, _ in LineIterator(event_stream):
                json_byte += line
                time_to_next_token.append(
                    time.monotonic() - most_recent_received_token_time
                )
                most_recent_received_token_time = time.monotonic()
            ttft = first_chunk_at - start_time
            resp = json.loads(json_byte)
            total_request_time = time.monotonic() - start_time
            generated_text = resp[0]["generation"]["content"]

        except Exception as e:
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            error_msg = str(e)
            error_response_code = 500

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = None
        metrics[common_metrics.NUM_TOTAL_TOKENS] = None
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = None
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


class LineIterator:
    """
    A helper class for parsing the byte stream input.
    Reference: https://aws.amazon.com/blogs/machine-learning/elevating-the-generative-ai-experience-introducing-streaming-support-in-amazon-sagemaker-hosting/
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0
        self.ttft = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                if self.ttft == 0:
                    self.ttft = time.monotonic()
                self.read_pos += len(line)
                return line[:-1], self.ttft, time.monotonic()
            # kyle: dealing with last ']' for chat output
            if line and self.read_pos == self.buffer.getbuffer().nbytes - 1:
                self.read_pos += 1
                return line, self.ttft, time.monotonic()
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])

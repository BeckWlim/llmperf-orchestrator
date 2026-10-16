import json
import os
import time
from typing import Any, Dict

import requests

from llmperf.clients.base import LLMClient, LLMResponse
from llmperf.clients.models import RequestConfig
from llmperf.metrics import constants as common_metrics


class VertexAIClient(LLMClient):
    """Client for VertexAI API."""

    def llm_request(self, request_config: RequestConfig) -> LLMResponse:
        project_id = self.environment.get("GCLOUD_PROJECT_ID")
        region = self.environment.get("GCLOUD_REGION")
        endpoint_id = self.environment.get("VERTEXAI_ENDPOINT_ID")
        raw_access_token = self.environment.get("GCLOUD_ACCESS_TOKEN")
        if not project_id:
            raise ValueError("the environment variable GCLOUD_PROJECT_ID must be set.")
        if not region:
            raise ValueError("the environment variable GCLOUD_REGION must be set.")
        if not endpoint_id:
            raise ValueError(
                "the environment variable VERTEXAI_ENDPOINT_ID must be set."
            )
        if not raw_access_token:
            raise ValueError(
                "the environment variable GCLOUD_ACCESS_TOKEN must be set."
            )
        access_token = raw_access_token.strip()
        if not access_token:
            raise ValueError(
                "the environment variable GCLOUD_ACCESS_TOKEN must be set."
            )
        prompt_text, prompt_len = request_config.prompt

        ttft = 0.0
        generated_text = ""
        total_request_time = 0.0

        metrics: Dict[str, Any] = {}
        response_code = -1

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        try:
            # Define the URL for the request
            url = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"
            )

            # Define the headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            sampling_params = dict(request_config.sampling_params or {})
            if "max_new_tokens" in sampling_params:
                sampling_params["maxOutputTokens"] = sampling_params.pop(
                    "max_new_tokens"
                )

            # Define the data payload
            data = {
                "instances": [{"prompt": prompt_text}],
                "parameters": sampling_params,
            }

            # Make the POST request
            start_time = time.monotonic()
            response = requests.post(url, headers=headers, data=json.dumps(data))
            total_request_time = time.monotonic() - start_time
            response_code = response.status_code
            response.raise_for_status()
            # output from the endpoint is in the form:
            # {"predictions": ["Input: ... \nOutput:\n ..."]}
            generated_text = response.json()["predictions"][0].split("\nOutput:\n")[1]
            ttft = -1.0

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            metrics[common_metrics.ERROR_CODE] = response_code
            print(f"Warning Or Error: {e}")
            print(response_code)

        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        # This prediction schema has no token usage. The Runner may optionally
        # count output tokens locally after its timed execution phase.
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = None
        metrics[common_metrics.NUM_TOTAL_TOKENS] = None
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = None
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


if __name__ == "__main__":
    # Run these before hand:

    # gcloud auth application-default login
    # gcloud config set project YOUR_PROJECT_ID
    # export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
    # export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
    # export GCLOUD_REGION=YOUR_REGION
    # export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

    client = VertexAIClient()
    request_config = RequestConfig(
        prompt=("Give me ten interview questions for the role of program manager.", 10),
        model="gpt3",
        sampling_params={
            "temperature": 0.2,
            "max_new_tokens": 256,
            "top_k": 40,
            "top_p": 0.95,
        },
    )
    client.llm_request(request_config)

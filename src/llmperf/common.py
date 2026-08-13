from typing import List

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients.vertexai_client import VertexAIClient
from llmperf.ray_llm_client import LLMClient


SUPPORTED_APIS = ["openai", "anthropic", "litellm"]


def construct_clients(llm_api: str, num_clients: int) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [OpenAIChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api == "sagemaker":
        try:
            from llmperf.ray_clients.sagemaker_client import SageMakerClient
        except ModuleNotFoundError as exc:
            if exc.name == "boto3":
                raise RuntimeError(
                    "The SageMaker adapter requires the 'sagemaker' extra: "
                    "pip install 'LLMPerf[sagemaker]'"
                ) from exc
            raise
        clients = [SageMakerClient.remote() for _ in range(num_clients)]
    elif llm_api == "vertexai":
        clients = [VertexAIClient.remote() for _ in range(num_clients)]
    elif llm_api in SUPPORTED_APIS:
        try:
            __import__("litellm")
        except ModuleNotFoundError as exc:
            if exc.name != "litellm":
                raise
            raise RuntimeError(
                "This adapter requires the 'litellm' extra: "
                "pip install 'LLMPerf[litellm]'"
            ) from exc
        from llmperf.ray_clients.litellm_client import LiteLLMClient

        clients = [LiteLLMClient.remote() for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients

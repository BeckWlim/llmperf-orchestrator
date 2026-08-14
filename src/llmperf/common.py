from functools import lru_cache
import os
from typing import Any, List, Type

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients.vertexai_client import VertexAIClient
from llmperf.ray_llm_client import LLMClient


SUPPORTED_APIS = ["openai", "anthropic", "litellm"]
RAY_ACTOR_CPUS_ENV = "LLMPERF_WORKER_RAY_ACTOR_CPUS"


def _client_class(llm_api: str) -> Type[LLMClient]:
    if llm_api == "openai":
        return OpenAIChatCompletionsClient
    if llm_api == "sagemaker":
        try:
            from llmperf.ray_clients.sagemaker_client import SageMakerClient
        except ModuleNotFoundError as exc:
            if exc.name == "boto3":
                raise RuntimeError(
                    "The SageMaker adapter requires the 'sagemaker' extra: "
                    "pip install 'LLMPerf[sagemaker]'"
                ) from exc
            raise
        return SageMakerClient
    if llm_api == "vertexai":
        return VertexAIClient
    if llm_api in SUPPORTED_APIS:
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

        return LiteLLMClient
    raise ValueError(f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}")


@lru_cache(maxsize=None)
def _ray_client_class(llm_api: str, num_cpus: float) -> Any:
    import ray

    return ray.remote(
        num_cpus=num_cpus,
        # A client actor is the atomic request-execution unit seen by the
        # Scheduler. Keep method execution serial even if Ray's defaults or
        # surrounding launch code change.
        max_concurrency=1,
        max_restarts=0,
        max_task_retries=0,
    )(_client_class(llm_api))


def construct_clients(llm_api: str, num_clients: int) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    num_cpus = float(os.environ.get(RAY_ACTOR_CPUS_ENV, "1.0"))
    if num_cpus <= 0:
        raise ValueError(f"{RAY_ACTOR_CPUS_ENV} must be greater than zero")
    remote_class = _ray_client_class(llm_api, num_cpus)
    return [remote_class.remote() for _ in range(num_clients)]

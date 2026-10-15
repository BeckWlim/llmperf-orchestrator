from functools import lru_cache
import os
from typing import TYPE_CHECKING, List, Protocol, Type, TypeVar

if TYPE_CHECKING:
    from ray.actor import ActorProxy

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients.vertexai_client import VertexAIClient
from llmperf.ray_llm_client import LLMClient

SUPPORTED_APIS = ["openai", "anthropic", "litellm", "sagemaker", "vertexai"]
LITELLM_APIS = {"anthropic", "litellm"}
RAY_ACTOR_CPUS_ENV = "LLMPERF_WORKER_RAY_ACTOR_CPUS"
RAY_ACTOR_CONCURRENCY_ENV = "LLMPERF_WORKER_RAY_ACTOR_CONCURRENCY"


def actor_concurrency() -> int:
    """Return the bounded number of simultaneous requests per client actor."""

    concurrency = int(os.environ.get(RAY_ACTOR_CONCURRENCY_ENV, "1"))
    if concurrency < 1:
        raise ValueError(f"{RAY_ACTOR_CONCURRENCY_ENV} must be greater than zero")
    return concurrency


def request_actor_count(request_slots: int, requests_per_actor: int) -> int:
    if request_slots < 1 or requests_per_actor < 1:
        raise ValueError("Request slots and requests per actor must be positive")
    return (request_slots + requests_per_actor - 1) // requests_per_actor


ClientT = TypeVar("ClientT")


class _RemoteActorClass(Protocol[ClientT]):
    """Public capability used from a configured Ray actor class."""

    def remote(self) -> "ActorProxy[ClientT]": ...


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
    if llm_api in LITELLM_APIS:
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
def _ray_client_class(
    llm_api: str, num_cpus: float, max_concurrency: int = 1
) -> _RemoteActorClass[LLMClient]:
    import ray

    remote_class = ray.remote(_client_class(llm_api))
    return remote_class.options(
        num_cpus=num_cpus,
        # Synchronous adapters use Ray's bounded thread pool. Each invocation
        # owns its request/response state; actors remain exclusive to a Runner.
        max_concurrency=max_concurrency,
        max_restarts=0,
        max_task_retries=0,
    )


def construct_clients(llm_api: str, num_clients: int) -> "List[ActorProxy[LLMClient]]":
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
    remote_class = _ray_client_class(llm_api, num_cpus, actor_concurrency())
    return [remote_class.remote() for _ in range(num_clients)]

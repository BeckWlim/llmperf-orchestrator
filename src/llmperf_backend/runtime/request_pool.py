"""Scheduler-owned shared Ray actors with per-request configuration and fair capacity."""

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    TypedDict,
)

from llmperf.clients.factory import _client_class
from llmperf.clients.models import RequestConfig
from llmperf.clients.base import LLMResponse

if TYPE_CHECKING:
    from ray import ObjectRef


class RequestMethod(Protocol):
    def remote(
        self, request: RequestConfig, environment: Dict[str, str]
    ) -> Awaitable[LLMResponse]: ...


@runtime_checkable
class RequestActorHandle(Protocol):
    @property
    def execute(self) -> RequestMethod: ...


class RouterMethod(Protocol):
    def remote(
        self, runner_id: str, request: RequestConfig, environment: Dict[str, str]
    ) -> "ObjectRef[LLMResponse]": ...


@runtime_checkable
class RouterHandle(Protocol):
    @property
    def execute(self) -> RouterMethod: ...

    @property
    def status(self) -> "StatusMethod": ...

    @property
    def cancel(self) -> "CancelMethod": ...


class CancelMethod(Protocol):
    def remote(self, runner_id: str) -> "ObjectRef[None]": ...


class StatusMethod(Protocol):
    def remote(self) -> "ObjectRef[PoolStatus]": ...


class PoolStatus(TypedDict):
    actor_count: int
    request_capacity: int
    executing_requests: int
    available_slots: int
    queued_requests: int
    runners: Dict[str, Dict[str, int]]


def create_request_actor(num_cpus: float, max_concurrency: int) -> RequestActorHandle:
    import ray

    handle = (
        ray.remote(RequestActor)
        .options(
            num_cpus=num_cpus,
            max_concurrency=max_concurrency,
            max_restarts=0,
            max_task_retries=0,
        )
        .remote()
    )
    if not isinstance(handle, RequestActorHandle):
        raise TypeError("Ray actor does not expose request execution")
    return handle


def create_request_router(
    actors: List[RequestActorHandle],
    requests_per_actor: int,
    max_pending: int,
    max_requests: Optional[int] = None,
) -> RouterHandle:
    import ray

    handle = (
        ray.remote(RequestRouter)
        .options(num_cpus=0, max_concurrency=10000, max_restarts=0, max_task_retries=0)
        .remote(actors, requests_per_actor, max_pending, max_requests)
    )
    if not isinstance(handle, RouterHandle):
        raise TypeError("Ray actor does not expose request routing")
    return handle


class RequestActor:
    """One invocation prepares and sends one Worker while holding a shared slot."""

    def __init__(self) -> None:
        from llmperf_backend.runtime.worker_input import WorkerInputPreparer

        self.preparer = WorkerInputPreparer()

    def execute(
        self, request: RequestConfig, environment: Dict[str, str]
    ) -> LLMResponse:
        import time
        from datetime import datetime, timezone
        from llmperf.benchmark.throughput import normalize_request_metrics
        from llmperf.metrics import constants
        from llmperf_backend.runtime.worker import _prepared_tokenizer
        from llmperf.datasets.tokenizer import TOKENIZER_PATH, TOKENIZER_FAST

        started_at = time.monotonic()
        try:
            if (
                request.deadline_at is not None
                and datetime.now(timezone.utc) >= request.deadline_at
            ):
                raise TimeoutError("Worker deadline elapsed while waiting for a slot")
            prepared_request = self.preparer.prepare(request, environment)
        except Exception as exc:
            return (
                {
                    constants.ERROR_CODE: "worker_preparation_error",
                    constants.ERROR_MSG: str(exc),
                    constants.E2E_LAT: 0.0,
                    constants.TTFT: None,
                    constants.NUM_INPUT_TOKENS: None,
                    constants.NUM_OUTPUT_TOKENS: None,
                    constants.REQ_OUTPUT_THROUGHPUT: None,
                    constants.REQUEST_METADATA: dict(request.metadata or {}),
                },
                "",
                request,
            )
        elapsed_seconds = time.monotonic() - started_at
        remaining_seconds = (
            (request.deadline_at - datetime.now(timezone.utc)).total_seconds()
            if request.deadline_at is not None
            else (
                request.timeout_seconds - elapsed_seconds
                if request.timeout_seconds is not None
                else None
            )
        )
        if remaining_seconds is not None and remaining_seconds <= 0:
            return (
                {
                    constants.ERROR_CODE: "worker_preparation_timeout",
                    constants.ERROR_MSG: "Worker deadline elapsed before Provider submission",
                    constants.E2E_LAT: 0.0,
                    constants.TTFT: None,
                    constants.NUM_INPUT_TOKENS: prepared_request.prompt[1],
                    constants.NUM_OUTPUT_TOKENS: None,
                    constants.REQ_OUTPUT_THROUGHPUT: None,
                    constants.REQUEST_METADATA: dict(prepared_request.metadata or {}),
                },
                "",
                prepared_request,
            )
        execution_request = prepared_request.model_copy(
            update={"timeout_seconds": remaining_seconds}
        )
        client_type = _client_class(execution_request.llm_api or "openai")
        client = client_type(environment)
        try:
            metrics, generated_text, returned_request = client.llm_request(
                execution_request
            )
        except Exception as exc:
            metrics = {
                constants.ERROR_CODE: "worker_request_error",
                constants.ERROR_MSG: str(exc),
                constants.E2E_LAT: time.monotonic() - started_at - elapsed_seconds,
            }
            generated_text = ""
            returned_request = execution_request
        metrics[constants.REQUEST_METADATA] = dict(
            metrics.get(constants.REQUEST_METADATA) or execution_request.metadata or {}
        )
        metrics.setdefault(constants.ERROR_CODE, None)
        metrics.setdefault(constants.E2E_LAT, 0.0)
        metrics.setdefault(constants.TTFT, None)
        metrics.setdefault(constants.NUM_INPUT_TOKENS, execution_request.prompt[1])
        if "history_version" in (execution_request.metadata or {}):
            metrics[constants.LOCAL_INPUT_TOKENS] = None
        if (
            request.preparation is not None
            and request.preparation.output_token_count == "local"
        ):
            tokenizer = _prepared_tokenizer(
                environment[TOKENIZER_PATH],
                environment.get(TOKENIZER_FAST, "true").lower() == "true",
            )
            normalize_request_metrics(
                metrics, generated_text, lambda value: len(tokenizer.encode(value))
            )
        else:
            normalize_request_metrics(metrics, generated_text)
        return metrics, generated_text, returned_request


@dataclass(eq=False)
class WaitingRequest:
    runner_id: str


class RequestRouter:
    """Async admission and round-robin fairness across logical Runners.

    Actors are synchronous threaded executors. This actor only awaits ObjectRefs;
    it never tokenizes, calls a Provider, or opens a database connection.
    """

    def __init__(
        self,
        actors: List[RequestActorHandle],
        requests_per_actor: int,
        max_pending: int,
        max_requests: Optional[int] = None,
    ):
        if not actors or requests_per_actor < 1 or max_pending < 1:
            raise ValueError("Request pool requires positive bounded capacity")
        self.actors = actors
        self.limit = requests_per_actor
        self.max_pending = max_pending
        self.capacity = min(
            len(actors) * requests_per_actor,
            max_requests or len(actors) * requests_per_actor,
        )
        self.busy = [0 for _ in actors]
        self.queues: Dict[str, deque[WaitingRequest]] = {}
        self.order: deque[str] = deque()
        self.condition = asyncio.Condition()
        self.cancelled: set[str] = set()
        self.executing_by_runner: Dict[str, int] = {}

    def status(self) -> PoolStatus:
        executing = sum(self.busy)
        return {
            "actor_count": len(self.actors),
            "request_capacity": self.capacity,
            "executing_requests": executing,
            "available_slots": self.capacity - executing,
            "queued_requests": sum(len(queue) for queue in self.queues.values()),
            "runners": {
                runner_id: {
                    "queued": len(self.queues.get(runner_id, ())),
                    "running": self.executing_by_runner.get(runner_id, 0),
                }
                for runner_id in self.queues.keys() | self.executing_by_runner.keys()
            },
        }

    async def cancel(self, runner_id: str) -> None:
        async with self.condition:
            self.cancelled.add(runner_id)
            self.condition.notify_all()

    async def forget(self, runner_id: str) -> None:
        async with self.condition:
            self.cancelled.discard(runner_id)

    def _remove(self, waiter: WaitingRequest) -> None:
        queue = self.queues.get(waiter.runner_id)
        if queue is None or waiter not in queue:
            return
        queue.remove(waiter)
        if not queue:
            del self.queues[waiter.runner_id]
            self.order.remove(waiter.runner_id)

    async def execute(
        self, runner_id: str, request: RequestConfig, environment: Dict[str, str]
    ) -> LLMResponse:
        waiter = WaitingRequest(runner_id)
        actor_index: Optional[int] = None
        async with self.condition:
            if sum(len(queue) for queue in self.queues.values()) >= self.max_pending:
                raise RuntimeError("Request pool pending capacity exceeded")
            if runner_id not in self.queues:
                self.queues[runner_id] = deque()
                self.order.append(runner_id)
            self.queues[runner_id].append(waiter)
            try:
                while actor_index is None:
                    if runner_id in self.cancelled:
                        raise RuntimeError("Runner cancelled before request dispatch")
                    candidate = min(range(len(self.busy)), key=self.busy.__getitem__)
                    if (
                        self.order[0] == runner_id
                        and self.queues[runner_id][0] is waiter
                        and self.busy[candidate] < self.limit
                        and sum(self.busy) < self.capacity
                    ):
                        actor_index = candidate
                        self.busy[candidate] += 1
                        self.executing_by_runner[runner_id] = (
                            self.executing_by_runner.get(runner_id, 0) + 1
                        )
                        self.order.rotate(-1)
                    else:
                        await self.condition.wait()
            finally:
                self._remove(waiter)
                self.condition.notify_all()
        try:
            result_ref = self.actors[actor_index].execute.remote(request, environment)
            # Shield physical execution: caller cancellation must not free a slot
            # while the HTTP request is still running in its actor.
            result_future = asyncio.ensure_future(result_ref)
            try:
                return await asyncio.shield(result_future)
            except asyncio.CancelledError:
                await asyncio.shield(result_future)
                raise
        finally:
            async with self.condition:
                self.busy[actor_index] -= 1
                self.executing_by_runner[runner_id] -= 1
                if not self.executing_by_runner[runner_id]:
                    del self.executing_by_runner[runner_id]
                self.condition.notify_all()


class SharedRequestExecutor:
    """Synchronous bridge used by a benchmark's request-launching threads."""

    def __init__(
        self, router: RouterHandle, runner_id: str, environment: Mapping[str, str]
    ):
        self.router = router
        self.runner_id = runner_id
        self.environment = dict(environment)

    def __call__(self, request: RequestConfig) -> LLMResponse:
        import ray

        return ray.get(
            self.router.execute.remote(self.runner_id, request, self.environment)
        )

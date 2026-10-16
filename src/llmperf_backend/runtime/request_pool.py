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
    def remote(self) -> "ObjectRef[Dict[str, int]]": ...


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
    """Each invocation owns its adapter and credentials; no environment mutation."""

    def execute(
        self, request: RequestConfig, environment: Dict[str, str]
    ) -> LLMResponse:
        client_type = _client_class(request.llm_api or "openai")
        client = client_type(environment)
        return client.llm_request(request)


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

    def status(self) -> Dict[str, int]:
        executing = sum(self.busy)
        return {
            "actor_count": len(self.actors),
            "request_capacity": self.capacity,
            "executing_requests": executing,
            "available_slots": self.capacity - executing,
            "queued_requests": sum(len(queue) for queue in self.queues.values()),
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

"""Scheduler-owned bounded Worker dispatch and atomic result publication."""

import asyncio
from datetime import datetime, timezone
import time
from typing import Dict, Mapping, Optional, Protocol

from sqlalchemy.exc import SQLAlchemyError

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.base import LLMResponse
from llmperf.metrics import constants
from llmperf_backend.runtime.messages import (
    MessageStore,
    MessageReconciler,
    RunnerMessage,
    RunnerMessageBus,
)
from llmperf_backend.runtime.request_pool import RouterHandle
from llmperf_backend.runtime.runner import WorkerResult


class DispatchRepository(MessageStore, Protocol):
    async def reserve_worker(
        self, runner_id: str, step: RequestStep, window: Optional[int]
    ) -> bool: ...
    async def commit_worker(
        self, runner_id: str, result: WorkerResult
    ) -> Optional[RunnerMessage]: ...


class WorkerDispatcher:
    def __init__(
        self, router: RouterHandle, repository: DispatchRepository, capacity: int
    ):
        if capacity < 1:
            raise ValueError("Scheduler admission capacity must be positive")
        self.router = router
        self.repository = repository
        self.reconciler = MessageReconciler(repository)
        self._capacity = asyncio.Semaphore(capacity)
        self.buses: Dict[str, RunnerMessageBus] = {}
        self.environments: Dict[str, Dict[str, str]] = {}
        self.active: Dict[tuple[str, str], asyncio.Task[None]] = {}
        self.cancelled: set[str] = set()
        self.errors: Dict[str, BaseException] = {}
        self.waiting: set[tuple[str, str]] = set()
        self.runner_runtimes: Dict[str, tuple[str, ...]] = {}
        self.runtime_limits: Dict[str, int] = {}
        self.runtime_gates: Dict[str, asyncio.Semaphore] = {}

    def register(
        self,
        runner_id: str,
        environment: Mapping[str, str],
        runtime_limits: Optional[Mapping[str, int]] = None,
    ) -> RunnerMessageBus:
        for runtime_id, limit in (runtime_limits or {}).items():
            if limit < 1:
                raise ValueError("Runtime Worker limit must be positive")
            if self.runtime_limits.get(runtime_id, limit) != limit:
                raise ValueError("Runtime Worker limit changed during execution")
            self.runtime_limits[runtime_id] = limit
            self.runtime_gates.setdefault(runtime_id, asyncio.Semaphore(limit))
        self.runner_runtimes[runner_id] = tuple(runtime_limits or {})
        bus = RunnerMessageBus(runner_id, self.repository, reconciler=self.reconciler)
        self.buses[runner_id] = bus
        self.environments[runner_id] = dict(environment)
        return bus

    async def submit(
        self, runner_id: str, step: RequestStep, window: Optional[int]
    ) -> None:
        waiting_key = (runner_id, step.step_id)
        self.waiting.add(waiting_key)
        runtime_gates = [
            self.runtime_gates[runtime_id]
            for runtime_id in self.runner_runtimes.get(runner_id, ())
        ]
        acquired_gates: list[asyncio.Semaphore] = []
        capacity_acquired = False
        handed_off = False
        try:
            # Child-to-parent quota waits never occupy global admission capacity.
            for gate in runtime_gates:
                await gate.acquire()
                acquired_gates.append(gate)
            await self._capacity.acquire()
            capacity_acquired = True
            if runner_id in self.cancelled:
                raise asyncio.CancelledError
            if not await self.repository.reserve_worker(runner_id, step, window):
                raise RuntimeError(
                    "Worker reservation rejected; automatic replay is disabled"
                )
            key = (runner_id, step.step_id)
            self.active[key] = asyncio.create_task(
                self._execute(runner_id, step),
                name=f"worker:{runner_id}:{step.step_id}",
            )

            def release_capacity(completed: asyncio.Task[None]) -> None:
                self.active.pop(key, None)
                self._capacity.release()
                for gate in reversed(acquired_gates):
                    gate.release()

            # A callback also releases permits if cancelled before _execute starts.
            self.active[key].add_done_callback(release_capacity)
            handed_off = True
        finally:
            self.waiting.discard(waiting_key)
            if not handed_off:
                if capacity_acquired:
                    self._capacity.release()
                for gate in reversed(acquired_gates):
                    gate.release()

    async def _execute(self, runner_id: str, step: RequestStep) -> None:
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        try:
            try:
                response_future = asyncio.ensure_future(
                    self.router.execute.remote(
                        runner_id, step.request, self.environments[runner_id]
                    )
                )
                try:
                    response = await asyncio.shield(response_future)
                except asyncio.CancelledError:
                    # Keep runtime permits until physical execution releases its slot.
                    await asyncio.shield(response_future)
                    raise
            except Exception as exc:
                response = (
                    {
                        constants.ERROR_CODE: "worker_request_error",
                        constants.ERROR_MSG: str(exc),
                        constants.E2E_LAT: time.monotonic() - started_monotonic,
                        constants.TTFT: None,
                        constants.NUM_INPUT_TOKENS: step.request.prompt[1],
                    },
                    "",
                    step.request,
                )
            metrics, generated_text, returned_request = response
            metadata = dict(
                metrics.get(constants.REQUEST_METADATA) or step.request.metadata or {}
            )
            metadata["worker_id"] = step.step_id
            metrics[constants.REQUEST_METADATA] = metadata
            if "history_version" in metadata:
                # The prepared prompt count excludes reused input messages.
                # Full input accounting must use Provider usage, if available.
                metrics[constants.LOCAL_INPUT_TOKENS] = None
                metrics[constants.NUM_INPUT_TOKENS] = metrics.get(
                    constants.PROVIDER_INPUT_TOKENS
                )
            metrics.setdefault(
                constants.REQUEST_TIMING,
                {
                    "client_start_utc": started_at.isoformat(),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            result = WorkerResult(
                step.step_id,
                (metrics, generated_text, returned_request),
                datetime.now(timezone.utc),
            )
            while True:
                try:
                    committed_message = await self.repository.commit_worker(
                        runner_id, result
                    )
                    if committed_message is None:
                        raise RuntimeError("Worker result commit rejected")
                    break
                except SQLAlchemyError:
                    # Retain this completed result, never resend the request.
                    await asyncio.sleep(1.0)
            await self.buses[runner_id].deliver_committed(committed_message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.errors[runner_id] = exc
            await self.buses[runner_id].close()

    async def cancel(self, runner_id: str) -> None:
        self.cancelled.add(runner_id)
        await self.router.cancel.remote(runner_id)
        tasks = [task for (owner, _), task in self.active.items() if owner == runner_id]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def unregister(self, runner_id: str) -> None:
        if any(owner == runner_id for owner, _ in self.active):
            await self.cancel(runner_id)
        bus = self.buses.pop(runner_id, None)
        if bus is not None:
            await bus.close()
        self.environments.pop(runner_id, None)
        self.cancelled.discard(runner_id)
        self.errors.pop(runner_id, None)
        runtime_ids = self.runner_runtimes.pop(runner_id, ())
        for runtime_id in runtime_ids:
            if not any(
                runtime_id in owners for owners in self.runner_runtimes.values()
            ):
                self.runtime_gates.pop(runtime_id, None)
                self.runtime_limits.pop(runtime_id, None)

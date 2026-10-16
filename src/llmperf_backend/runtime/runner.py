"""Runner-owned async submission; Workers are single requests, never waiting loops."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Dict, List, Mapping, Optional, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.base import LLMResponse
from llmperf.metrics import constants
from llmperf_backend.runtime.messages import (
    RunnerMessage,
    RunnerMessageBus,
    WorkerCompletion,
    WorkerCancellation,
)
from llmperf_backend.runtime.triggers import TimerSource
from llmperf_backend.runtime.triggers import (
    Completion,
    DateTrigger,
    DependencyTrigger,
    Eligibility,
)


@dataclass(frozen=True)
class WorkerResult:
    step_id: str
    response: LLMResponse
    completed_at: datetime

    @property
    def succeeded(self) -> bool:
        return self.response[0].get(constants.ERROR_CODE) is None

    def completion(self) -> WorkerCompletion:
        return WorkerCompletion(
            step_id=self.step_id,
            metrics=self.response[0],
            generated_text=self.response[1],
            request=self.response[2],
        )


class WorkerScheduler(Protocol):
    async def submit(
        self, runner_id: str, step: RequestStep, window: Optional[int]
    ) -> None: ...
    async def cancel(self, runner_id: str) -> None: ...


class TaskLogic(ABC):
    """Behavior hooks never submit requests or mutate authoritative storage."""

    @abstractmethod
    def steps(self) -> Sequence[RequestStep]: ...

    @abstractmethod
    def eligibility(
        self, step: RequestStep, now: datetime, results: Mapping[str, WorkerResult]
    ) -> Eligibility: ...

    def request(
        self, step: RequestStep, results: Mapping[str, WorkerResult]
    ) -> RequestStep:
        return step

    def on_message(
        self, message: RunnerMessage, results: Mapping[str, WorkerResult]
    ) -> Sequence[RequestStep]:
        """Optional bounded dynamic expansion; creation IDs must be deterministic."""
        if message.topic not in {"worker.completed", "worker.cancelled", "timer.due"}:
            raise ValueError(f"Unsupported task message topic: {message.topic}")
        return ()

    def creation_closed(self, results: Mapping[str, WorkerResult]) -> bool:
        """Dynamic logic keeps this false while it owns future message-driven work."""
        return True


class WorkflowTaskLogic(TaskLogic):
    def __init__(self, steps: Sequence[RequestStep], created_at: datetime):
        self._steps = tuple(steps)
        self.created_at = created_at
        identifiers = {step.step_id for step in steps}
        if len(identifiers) != len(steps):
            raise ValueError("Worker step IDs must be unique within a Runner")
        visited: set[str] = set()
        # Upstream preparation supplies topologically ordered steps. This also
        # rejects unknown dependencies and cycles before any request can be sent.
        for step in steps:
            if not step.step_id or not set(step.dependencies) <= visited:
                raise ValueError("Worker dependencies must refer to earlier steps")
            if not set(step.message_sources) <= set(step.dependencies):
                raise ValueError("message sources must name a direct dependency")
            if not math.isfinite(step.delay_seconds) or step.delay_seconds < 0:
                raise ValueError("Worker delays must be finite and nonnegative")
            if step.due_at is not None and step.due_at.tzinfo is None:
                raise ValueError("Worker due time must include a UTC offset")
            visited.add(step.step_id)

    def steps(self) -> Sequence[RequestStep]:
        return self._steps

    def eligibility(
        self, step: RequestStep, now: datetime, results: Mapping[str, WorkerResult]
    ) -> Eligibility:
        completions = {
            identifier: Completion(result.succeeded, result.completed_at)
            for identifier, result in results.items()
        }
        dependency = DependencyTrigger(
            step.dependencies, self.created_at, step.delay_seconds
        ).evaluate(now, completions)
        if dependency.blocked or dependency.due_at is None:
            return dependency
        due_at = max(dependency.due_at, step.due_at or dependency.due_at)
        return DateTrigger(due_at).evaluate(now, completions)

    def request(
        self, step: RequestStep, results: Mapping[str, WorkerResult]
    ) -> RequestStep:
        if not step.message_sources:
            return step
        history = [
            message
            for source in step.message_sources
            for message in results[source].completion().messages()
        ]
        messages = [*history, {"role": "user", "content": step.request.prompt[0]}]
        parameters = dict(step.request.sampling_params or {})
        parameters["messages"] = messages
        metadata = dict(step.request.metadata or {})
        metadata["history_version"] = len(history)
        metadata["messages_hash"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
        )
        prepared_request = step.request.model_copy(
            update={"sampling_params": parameters, "metadata": metadata}, deep=True
        )
        return RequestStep(
            step.step_id,
            prepared_request,
            step.dependencies,
            step.delay_seconds,
            step.due_at,
            step.message_sources,
        )


class RunnerRuntime:
    """Listen to the common bus and submit eligible work with Scheduler backpressure."""

    def __init__(
        self,
        runner_id: str,
        logic: TaskLogic,
        scheduler: WorkerScheduler,
        bus: RunnerMessageBus,
        timeout_seconds: float,
        max_outstanding_workers: Optional[int] = None,
    ):
        if max_outstanding_workers is not None and max_outstanding_workers < 1:
            raise ValueError("max_outstanding_workers must be positive when specified")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Runner timeout must be finite and positive")
        self.runner_id = runner_id
        self.logic = logic
        self.scheduler = scheduler
        self.bus = bus
        self.window = max_outstanding_workers
        self.timeout_seconds = timeout_seconds
        self.results: Dict[str, WorkerResult] = {}
        self.cancelled_steps: Dict[str, WorkerCancellation] = {}
        self.outstanding: set[str] = set()
        self.timed_out = False

    async def _cancel_step(
        self, step: RequestStep, failed_dependencies: list[str]
    ) -> None:
        cancellation = WorkerCancellation(
            step_id=step.step_id,
            reason="dependency_failed" if failed_dependencies else "readiness_blocked",
            failed_dependencies=failed_dependencies,
        )
        await self.bus.publish(
            RunnerMessage(
                self.runner_id,
                "worker.cancelled",
                cancellation.model_dump(mode="json"),
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"llmperf:{self.runner_id}:{step.step_id}:cancelled",
                    )
                ),
                source=step.step_id,
            )
        )
        self.cancelled_steps[step.step_id] = cancellation

    async def run(self) -> List[LLMResponse]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        pending = {step.step_id: step for step in self.logic.steps()}
        known_steps = dict(pending)
        consumer = "task-progression"
        message_task = asyncio.create_task(self.bus.receive(consumer))
        submission_task: Optional[asyncio.Task[None]] = None
        timer_task: Optional[asyncio.Task[None]] = None
        timer_due: Optional[datetime] = None
        timers = TimerSource(self.bus)
        try:
            while (
                pending
                or self.outstanding
                or submission_task is not None
                or not self.logic.creation_closed(self.results)
            ):
                if message_task.done():
                    messages = message_task.result()
                    for message in messages:
                        if message.topic == "worker.completed":
                            completion = WorkerCompletion.model_validate(
                                message.payload
                            )
                            identifier = completion.step_id
                            if identifier not in known_steps:
                                raise ValueError(
                                    "Worker completion does not belong to task"
                                )
                            self.results.setdefault(
                                identifier,
                                WorkerResult(
                                    identifier,
                                    (
                                        completion.metrics,
                                        completion.generated_text,
                                        completion.request,
                                    ),
                                    message.created_at,
                                ),
                            )
                            self.outstanding.discard(identifier)
                        elif message.topic == "worker.cancelled":
                            cancellation = WorkerCancellation.model_validate(
                                message.payload
                            )
                            if cancellation.step_id not in known_steps:
                                raise ValueError(
                                    "Worker cancellation does not belong to Runner"
                                )
                            self.cancelled_steps.setdefault(
                                cancellation.step_id, cancellation
                            )
                            pending.pop(cancellation.step_id, None)
                        for new_step in self.logic.on_message(message, self.results):
                            if new_step.step_id in known_steps:
                                if known_steps[new_step.step_id] != new_step:
                                    raise ValueError(
                                        "Dynamic Worker ID reused with different inputs"
                                    )
                                continue
                            if not set(new_step.dependencies) <= known_steps.keys():
                                raise ValueError(
                                    "Dynamic Worker has unknown dependencies"
                                )
                            known_steps[new_step.step_id] = new_step
                            pending[new_step.step_id] = new_step
                        await self.bus.acknowledge(message, consumer)
                    message_task = asyncio.create_task(self.bus.receive(consumer))
                if submission_task is not None and submission_task.done():
                    submission_task.result()
                    submission_task = None
                if timer_task is not None and timer_task.done():
                    timer_task.result()
                    timer_task = None
                    timer_due = None
                now = datetime.now(timezone.utc)
                next_due: Optional[datetime] = None
                for identifier, step in tuple(pending.items()):
                    failed_dependencies = [
                        parent
                        for parent in step.dependencies
                        if parent in self.cancelled_steps
                        or (
                            parent in self.results
                            and not self.results[parent].succeeded
                        )
                    ]
                    if failed_dependencies:
                        await self._cancel_step(step, failed_dependencies)
                        del pending[identifier]
                        continue
                    # The Runner enforces dependency success even when custom task
                    # logic supplies its own readiness or request transformation.
                    if any(parent not in self.results for parent in step.dependencies):
                        continue
                    eligibility = self.logic.eligibility(step, now, self.results)
                    if eligibility.blocked:
                        await self._cancel_step(step, [])
                        del pending[identifier]
                    elif eligibility.ready:
                        capacity = (
                            self.window is None or len(self.outstanding) < self.window
                        )
                        if (
                            submission_task is None
                            and capacity
                            and loop.time() < deadline
                        ):
                            prepared_step = self.logic.request(step, self.results)
                            # One blocked submission per Runner, not an unbounded task
                            # per ready request. Continue listening while Scheduler waits.
                            self.outstanding.add(identifier)
                            submission_task = asyncio.create_task(
                                self.scheduler.submit(
                                    self.runner_id, prepared_step, self.window
                                )
                            )
                            del pending[identifier]
                    elif eligibility.due_at is not None:
                        next_due = min(
                            next_due or eligibility.due_at, eligibility.due_at
                        )
                if next_due is None and timer_task is not None:
                    timer_task.cancel()
                    await asyncio.gather(timer_task, return_exceptions=True)
                    timer_task = None
                    timer_due = None
                if next_due is not None and (timer_due is None or next_due < timer_due):
                    if timer_task is not None:
                        timer_task.cancel()
                        await asyncio.gather(timer_task, return_exceptions=True)
                    timer_due = next_due
                    timer_task = asyncio.create_task(
                        timers.fire("task-readiness", next_due)
                    )
                if (
                    not pending
                    and not self.outstanding
                    and submission_task is None
                    and self.logic.creation_closed(self.results)
                ):
                    break
                remaining_seconds = deadline - loop.time()
                if remaining_seconds <= 0:
                    self.timed_out = True
                    break
                if (
                    pending
                    and not self.outstanding
                    and submission_task is None
                    and timer_task is None
                ):
                    raise RuntimeError(
                        "Task has unresolved dependencies without an event source"
                    )
                waiters: set[asyncio.Task] = {message_task}
                if submission_task is not None:
                    waiters.add(submission_task)
                if timer_task is not None:
                    waiters.add(timer_task)
                await asyncio.wait(
                    waiters,
                    timeout=remaining_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            for listener in (message_task, submission_task, timer_task):
                if listener is not None:
                    listener.cancel()
            await asyncio.gather(message_task, return_exceptions=True)
            await asyncio.gather(
                *(task for task in (submission_task, timer_task) if task is not None),
                return_exceptions=True,
            )
            if self.outstanding:
                await self.scheduler.cancel(self.runner_id)
        return [result.response for result in self.results.values()]

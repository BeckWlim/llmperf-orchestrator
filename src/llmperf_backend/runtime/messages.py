"""Runner-layer durable messaging; notifications only accelerate durable reads."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from llmperf.clients.models import RequestConfig


class WorkerCompletion(BaseModel):
    """Durable message data used for both result aggregation and later request inputs."""

    model_config = ConfigDict(extra="forbid")
    step_id: str
    metrics: dict[str, Any]
    generated_text: str
    request: RequestConfig

    def messages(self) -> list[dict[str, Any]]:
        """Reuse the executed request's message structure and append its response."""
        parameters = self.request.sampling_params or {}
        previous_messages = parameters.get("messages")
        request_messages = (
            TypeAdapter(list[dict[str, Any]]).validate_python(previous_messages)
            if previous_messages is not None
            else [{"role": "user", "content": self.request.prompt[0]}]
        )
        return [
            *deepcopy(request_messages),
            {"role": "assistant", "content": self.generated_text},
        ]


class WorkerCancellation(BaseModel):
    """Runner decision to cancel a planned Worker before it is submitted."""

    model_config = ConfigDict(extra="forbid")
    step_id: str
    reason: str
    failed_dependencies: list[str]


@dataclass(frozen=True)
class RunnerMessage:
    runner_id: str
    topic: str
    payload: dict[str, JsonValue]
    id: str = field(default_factory=lambda: str(uuid4()))
    source: Optional[str] = None
    destination: str = "runner"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MessageStore(Protocol):
    async def publish_message(self, message: RunnerMessage) -> bool: ...
    async def pending_messages(
        self, runner_id: str, destination: str, consumer: str, limit: int
    ) -> Sequence[RunnerMessage]: ...
    async def acknowledge_message(
        self, runner_id: str, message_id: str, consumer: str
    ) -> None: ...


class RunnerMessageBus:
    """One Runner's common mailbox for outputs, timers, and other task events.

    Receive never acknowledges automatically. Consumers acknowledge only after their
    durable effects commit; effects must use the stable message ID for deduplication.
    A restart or a dropped wakeup replays unacknowledged messages from the store.
    """

    def __init__(
        self, runner_id: str, store: MessageStore, reconcile_seconds: float = 1.0
    ):
        if reconcile_seconds <= 0:
            raise ValueError("message reconciliation interval must be positive")
        self.runner_id = runner_id
        self.store = store
        self.reconcile_seconds = reconcile_seconds
        self._condition = asyncio.Condition()
        self._generation = 0
        self._closed = False

    async def notify(self) -> None:
        async with self._condition:
            self._generation += 1
            self._condition.notify_all()

    async def publish(self, message: RunnerMessage) -> bool:
        if message.runner_id != self.runner_id:
            raise ValueError("Message belongs to another Runner")
        if self._closed:
            raise RuntimeError("Runner message bus is closed")
        accepted = await self.store.publish_message(message)
        if accepted:
            await self.notify()
        return accepted

    async def receive(
        self, consumer: str, destination: str = "runner", limit: int = 64
    ) -> Sequence[RunnerMessage]:
        if limit < 1:
            raise ValueError("message receive limit must be positive")
        while not self._closed:
            observed_generation = self._generation
            messages = await self.store.pending_messages(
                self.runner_id, destination, consumer, limit
            )
            if messages:
                return messages
            async with self._condition:
                if self._closed:
                    break
                if observed_generation != self._generation:
                    continue
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._closed
                            or observed_generation != self._generation
                        ),
                        self.reconcile_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        raise asyncio.CancelledError("Runner message bus closed")

    async def acknowledge(self, message: RunnerMessage, consumer: str) -> None:
        if message.runner_id != self.runner_id:
            raise ValueError("Message belongs to another Runner")
        await self.store.acknowledge_message(self.runner_id, message.id, consumer)

    async def close(self) -> None:
        self._closed = True
        await self.notify()

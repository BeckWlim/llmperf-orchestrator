"""Committed event delivery with bounded buffers and durable reconciliation."""

import asyncio
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from llmperf.clients.models import RequestConfig

LOGGER = logging.getLogger(__name__)


def _log_value(value: JsonValue) -> JsonValue:
    """Exclude transport credentials and raw Provider bodies from durable logs."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_log_value(item) for item in value]
    if isinstance(value, dict):
        safe_document: dict[str, JsonValue] = {}
        for name, item in value.items():
            normalized_name = name.lower().replace("-", "_")
            if normalized_name in {
                "headers",
                "response_headers",
                "base_url",
                "api_base",
                "error_msg",
            } or any(
                fragment in normalized_name
                for fragment in (
                    "api_key",
                    "apikey",
                    "authorization",
                    "password",
                    "secret",
                    "credential",
                    "private_key",
                )
            ):
                continue
            safe_document[name] = _log_value(item)
        return safe_document
    return value


class WorkerCompletion(BaseModel):
    """Durable message data used for both result aggregation and later request inputs."""

    model_config = ConfigDict(extra="forbid")
    step_id: str
    metrics: dict[str, Any]
    generated_text: str
    request: RequestConfig

    def for_storage(self) -> "WorkerCompletion":
        document = TypeAdapter(dict[str, JsonValue]).validate_python(
            self.model_dump(mode="json")
        )
        return WorkerCompletion.model_validate(_log_value(document))

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
    async def pending_messages_batch(
        self, runner_ids: Sequence[str], destination: str, consumer: str, limit: int
    ) -> Sequence[RunnerMessage]: ...
    async def acknowledge_messages(
        self, runner_id: str, message_ids: Sequence[str], consumer: str
    ) -> None: ...


@dataclass
class _Subscription:
    messages: dict[str, RunnerMessage] = field(default_factory=dict)
    recovered_ids: set[str] = field(default_factory=set)
    revision: int = 0


class MessageReconciler:
    """One recovery loop for all active mailboxes, with bounded per-Runner reads."""

    def __init__(self, store: MessageStore, interval_seconds: float = 5.0):
        if interval_seconds <= 0:
            raise ValueError("message reconciliation interval must be positive")
        self.store = store
        self.interval_seconds = interval_seconds
        self.buses: dict[str, RunnerMessageBus] = {}
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None

    def register(self, bus: "RunnerMessageBus") -> None:
        self.buses[bus.runner_id] = bus
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="llmperf-mailboxes")

    def wake(self) -> None:
        self._wake.set()

    async def unregister(self, bus: "RunnerMessageBus") -> None:
        if self.buses.get(bus.runner_id) is bus:
            self.buses.pop(bus.runner_id)
        if not self.buses:
            await self.close()

    async def close(self) -> None:
        running_task = self._task
        self._task = None
        if running_task is not None:
            running_task.cancel()
            await asyncio.gather(running_task, return_exceptions=True)

    async def reconcile(self) -> None:
        groups: dict[tuple[str, str], list[tuple[RunnerMessageBus, int]]] = {}
        for bus in tuple(self.buses.values()):
            for key, subscription in tuple(bus._subscriptions.items()):
                if len(subscription.messages) < bus.capacity:
                    groups.setdefault(key, []).append((bus, subscription.revision))
        for (consumer, destination), subscribers in groups.items():
            for offset in range(0, len(subscribers), 64):
                batch = subscribers[offset : offset + 64]
                messages = await self.store.pending_messages_batch(
                    [bus.runner_id for bus, _ in batch], destination, consumer, 1
                )
                owners = {bus.runner_id: (bus, revision) for bus, revision in batch}
                for message in messages:
                    owner = owners.get(message.runner_id)
                    if owner is not None:
                        bus, revision = owner
                        await bus._recover(message, consumer, revision)

    async def _run(self) -> None:
        while asyncio.current_task() is self._task:
            self._wake.clear()
            try:
                await self.reconcile()
            except Exception:
                LOGGER.exception(
                    "Mailbox reconciliation failed; durable events retained"
                )
            try:
                await asyncio.wait_for(self._wake.wait(), self.interval_seconds)
            except asyncio.TimeoutError:
                pass


class RunnerMessageBus:
    """One Runner's common mailbox for outputs, timers, and other task events.

    Receive never acknowledges automatically. Consumers acknowledge only after their
    durable effects commit; effects must use the stable message ID for deduplication.
    A restart or a dropped wakeup replays unacknowledged messages from the store.
    """

    def __init__(
        self,
        runner_id: str,
        store: MessageStore,
        reconcile_seconds: float = 1.0,
        *,
        reconciler: Optional[MessageReconciler] = None,
        capacity: int = 64,
    ):
        if reconcile_seconds <= 0:
            raise ValueError("message reconciliation interval must be positive")
        if capacity < 1:
            raise ValueError("message buffer capacity must be positive")
        self.runner_id = runner_id
        self.store = store
        self.reconcile_seconds = reconcile_seconds
        self._condition = asyncio.Condition()
        self._generation = 0
        self._closed = False
        self.capacity = capacity
        self.reconciler = reconciler
        self._subscriptions: dict[tuple[str, str], _Subscription] = {}
        if reconciler is not None:
            reconciler.register(self)

    async def notify(self) -> None:
        if self.reconciler is not None:
            self.reconciler.wake()
        async with self._condition:
            self._generation += 1
            self._condition.notify_all()

    async def deliver_committed(self, message: RunnerMessage) -> None:
        """Called only after persistence succeeds; overflow is recovered from storage."""
        if message.runner_id != self.runner_id:
            raise ValueError("Message belongs to another Runner")
        async with self._condition:
            if self._closed:
                return
            for (_, destination), subscription in self._subscriptions.items():
                if destination == message.destination:
                    if len(subscription.messages) < self.capacity:
                        subscription.messages.setdefault(message.id, message)
            self._generation += 1
            self._condition.notify_all()

    async def _recover(
        self, message: RunnerMessage, consumer: str, revision: int
    ) -> None:
        async with self._condition:
            subscription = self._subscriptions.get((consumer, message.destination))
            # A receipt committed during the read makes that snapshot stale.
            if (
                not self._closed
                and subscription is not None
                and subscription.revision == revision
                and len(subscription.messages) < self.capacity
            ):
                if message.id not in subscription.messages:
                    subscription.messages[message.id] = message
                    subscription.recovered_ids.add(message.id)
                self._generation += 1
                self._condition.notify_all()

    async def publish(self, message: RunnerMessage) -> bool:
        if message.runner_id != self.runner_id:
            raise ValueError("Message belongs to another Runner")
        if self._closed:
            raise RuntimeError("Runner message bus is closed")
        accepted = await self.store.publish_message(message)
        if accepted:
            await self.deliver_committed(message)
        return accepted

    async def receive(
        self, consumer: str, destination: str = "runner", limit: int = 64
    ) -> Sequence[RunnerMessage]:
        if limit < 1:
            raise ValueError("message receive limit must be positive")
        key = (consumer, destination)
        if key not in self._subscriptions:
            self._subscriptions[key] = _Subscription()
            if self.reconciler is not None:
                self.reconciler.wake()
        subscription = self._subscriptions[key]
        while not self._closed:
            observed_generation = self._generation
            if subscription.messages:
                return list(subscription.messages.values())[:limit]
            if self.reconciler is None:
                revision = subscription.revision
                recovered_messages = await self.store.pending_messages(
                    self.runner_id, destination, consumer, min(limit, self.capacity)
                )
                for message in recovered_messages:
                    await self._recover(message, consumer, revision)
                if subscription.messages:
                    continue
            async with self._condition:
                if self._closed:
                    break
                if observed_generation != self._generation:
                    continue
                try:
                    waiter = self._condition.wait_for(
                        lambda: self._closed or observed_generation != self._generation
                    )
                    if self.reconciler is None:
                        await asyncio.wait_for(waiter, self.reconcile_seconds)
                    else:
                        await waiter
                except asyncio.TimeoutError:
                    pass
        raise asyncio.CancelledError("Runner message bus closed")

    async def acknowledge(self, message: RunnerMessage, consumer: str) -> None:
        await self.acknowledge_many([message], consumer)

    async def acknowledge_many(
        self, messages: Sequence[RunnerMessage], consumer: str
    ) -> None:
        if any(message.runner_id != self.runner_id for message in messages):
            raise ValueError("Message belongs to another Runner")
        if not messages:
            return
        await self.store.acknowledge_messages(
            self.runner_id, [message.id for message in messages], consumer
        )
        for message in messages:
            subscription = self._subscriptions.get((consumer, message.destination))
            if subscription is not None:
                subscription.revision += 1
                subscription.messages.pop(message.id, None)
                if message.id in subscription.recovered_ids:
                    subscription.recovered_ids.remove(message.id)
                    if self.reconciler is not None:
                        self.reconciler.wake()

    async def close(self) -> None:
        self._closed = True
        await self.notify()
        self._subscriptions.clear()
        if self.reconciler is not None:
            await self.reconciler.unregister(self)

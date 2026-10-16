"""Transient scheduling state reconstructed from durable metadata and checkpoints."""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Protocol

from llmperf_backend.compiler.rounds import ready_round_runners
from llmperf_backend.runtime.triggers import Completion, DependencyTrigger

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconcileHint:
    runner_id: Optional[str] = None


class ReconcileNotifier:
    """Bounded, coalescing wakeups; durable records repair missed notifications."""

    def __init__(self, capacity: int = 1024):
        self.queue: asyncio.Queue[ReconcileHint] = asyncio.Queue(capacity)
        self.pending: set[ReconcileHint] = set()

    def publish(self, event: ReconcileHint) -> None:
        if event in self.pending or self.queue.full():
            return
        self.pending.add(event)
        self.queue.put_nowait(event)

    async def receive(self, timeout_seconds: float) -> Optional[ReconcileHint]:
        try:
            event = await asyncio.wait_for(self.queue.get(), timeout_seconds)
        except asyncio.TimeoutError:
            return None
        self.pending.discard(event)
        return event


@dataclass(frozen=True)
class NodeSnapshot:
    node_id: str
    dependencies: tuple[str, ...]
    delay_seconds: float
    state: str
    completed_at: Optional[datetime]
    terminal_status: Optional[str]


def ready_nodes(
    nodes: List[NodeSnapshot], created_at: datetime, now: datetime
) -> Dict[str, datetime]:
    """Evaluate DAG readiness without a database, actor, or role-specific branch."""
    completions = {
        node.node_id: Completion(node.terminal_status == "succeeded", node.completed_at)
        for node in nodes
        if node.terminal_status is not None and node.completed_at is not None
    }
    ready: Dict[str, datetime] = {}
    for node in nodes:
        if node.state not in {"pending", "blocked"}:
            continue
        eligibility = DependencyTrigger(
            node.dependencies, created_at, node.delay_seconds
        ).evaluate(now, completions)
        if eligibility.ready and eligibility.due_at is not None:
            ready[node.node_id] = eligibility.due_at
    return ready


class RuntimeRepository(Protocol):
    async def runtime_snapshot(
        self, runner_id: Optional[str] = None
    ) -> Dict[str, Any]: ...
    async def activate_node(
        self, node_id: str, due_at: datetime
    ) -> Optional[Dict[str, Any]]: ...
    async def start_selected(
        self, runner_id: str, owner: str
    ) -> Optional[Dict[str, Any]]: ...


class RuntimeCoordinator:
    """Reconcile durable work into an in-memory fair ready queue.

    Database operations persist selected transitions; selection and dependencies live
    here. Wakeups shorten latency, while bounded periodic reconciliation repairs loss.
    """

    def __init__(
        self,
        repository: RuntimeRepository,
        interval_seconds: float = 1.0,
        ready_capacity: int = 1024,
    ):
        self.repository = repository
        self.interval_seconds = interval_seconds
        self.notifier = ReconcileNotifier()
        self.queues: Dict[str, deque[str]] = {}
        self.order: deque[str] = deque()
        self.admitted: set[str] = set()
        self.condition = asyncio.Condition()
        self.task: Optional[asyncio.Task[None]] = None
        self.last_error: Optional[str] = None
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.ready_capacity = ready_capacity

    async def start(self) -> None:
        await self.reconcile()
        self.task = asyncio.create_task(self._run(), name="llmperf-coordinator")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        self.queues.clear()
        self.order.clear()
        self.admitted.clear()
        self.instances.clear()

    async def reconcile(self, runner_id: Optional[str] = None) -> None:
        snapshot = await self.repository.runtime_snapshot(runner_id)
        runners = ready_round_runners(
            snapshot["runners"], snapshot.get("round_states", [])
        )
        now = datetime.now(timezone.utc)
        if runner_id is None:
            self.instances.clear()
        for instance in snapshot["instances"]:
            if instance["state"] in {"planned", "active"}:
                self.instances[instance["instance_id"]] = instance
            else:
                self.instances.pop(instance["instance_id"], None)
        for instance in self.instances.values():
            nodes = [NodeSnapshot(**node) for node in instance["nodes"]]
            for node_id, due_at in ready_nodes(
                nodes, instance["created_at"], now
            ).items():
                if len(runners) + len(self.admitted) >= self.ready_capacity:
                    break
                runner = await self.repository.activate_node(node_id, due_at)
                if runner is not None:
                    runners.append(runner)
                    for node in instance["nodes"]:
                        if node["node_id"] == node_id:
                            node["state"] = "emitted"
        async with self.condition:
            for runner in runners:
                if len(self.admitted) >= self.ready_capacity:
                    break
                candidate_runner_id = runner["runner_id"]
                if candidate_runner_id in self.admitted:
                    continue
                campaign_id = runner.get("campaign_id") or candidate_runner_id
                if campaign_id not in self.queues:
                    self.queues[campaign_id] = deque()
                    self.order.append(campaign_id)
                self.queues[campaign_id].append(candidate_runner_id)
                self.admitted.add(candidate_runner_id)
            self.condition.notify_all()

    async def next_runner(self, owner: str) -> Dict[str, Any]:
        while True:
            async with self.condition:
                await self.condition.wait_for(lambda: bool(self.order))
                campaign_id = self.order.popleft()
                runner_id = self.queues[campaign_id].popleft()
                if self.queues[campaign_id]:
                    self.order.append(campaign_id)
                else:
                    del self.queues[campaign_id]
            try:
                runner = await self.repository.start_selected(runner_id, owner)
            except BaseException:
                self.admitted.discard(runner_id)
                raise
            if runner is not None:
                return runner
            self.admitted.discard(runner_id)

    def release_runner(self, runner_id: str) -> None:
        self.admitted.discard(runner_id)
        self.notifier.publish(ReconcileHint(runner_id))

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_reconcile_at = loop.time() + self.interval_seconds
        while True:
            event = await self.notifier.receive(
                max(0.001, next_reconcile_at - loop.time())
            )
            try:
                if event is None or loop.time() >= next_reconcile_at:
                    await self.reconcile()
                    next_reconcile_at = loop.time() + self.interval_seconds
                else:
                    await self.reconcile(event.runner_id)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__
                LOGGER.exception("Runtime reconciliation failed; durable work retained")

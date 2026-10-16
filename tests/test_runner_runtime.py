"""Runner message-driven submission and dependency behavior without Provider calls."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from unittest.mock import AsyncMock, Mock

from pydantic import TypeAdapter, JsonValue

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.models import RequestConfig
from llmperf_backend.runtime.messages import RunnerMessage, RunnerMessageBus
from llmperf_backend.runtime.runner import RunnerRuntime, WorkflowTaskLogic
from llmperf_backend.runtime.runner import WorkerResult
from llmperf_backend.runtime.dispatch import WorkerDispatcher
from llmperf_backend.configuration.models import BenchmarkConfig
from llmperf_backend.runtime.triggers import TimerSource
from llmperf_backend.runtime.triggers import Completion, DependencyTrigger, Eligibility
from llmperf_backend.storage.messages import ConversationMessageRecord, MessageRecord


class MemoryMessages:
    """Test-only persistence double; production uses PostgreSQL."""

    def __init__(self):
        self.messages: dict[str, RunnerMessage] = {}
        self.receipts: set[tuple[str, str]] = set()

    async def publish_message(self, message: RunnerMessage) -> bool:
        self.messages.setdefault(message.id, message)
        return True

    async def pending_messages(
        self, runner_id: str, destination: str, consumer: str, limit: int
    ) -> Sequence[RunnerMessage]:
        return [
            message
            for message in self.messages.values()
            if message.runner_id == runner_id
            and message.destination == destination
            and (message.id, consumer) not in self.receipts
        ][:limit]

    async def acknowledge_message(
        self, runner_id: str, message_id: str, consumer: str
    ) -> None:
        assert self.messages[message_id].runner_id == runner_id
        self.receipts.add((message_id, consumer))


class FakeScheduler:
    def __init__(self, bus: RunnerMessageBus, capacity: int):
        self.bus = bus
        self.capacity = asyncio.Semaphore(capacity)
        self.tasks: list[asyncio.Task[None]] = []
        self.submitted: list[RequestStep] = []
        self.active = 0
        self.peak = 0
        self.cancelled = False
        self.fail: set[str] = set()
        self.hold: Optional[asyncio.Event] = None

    async def submit(
        self, runner_id: str, step: RequestStep, window: Optional[int]
    ) -> None:
        await self.capacity.acquire()
        self.submitted.append(step)
        self.active += 1
        self.peak = max(self.active, self.peak)

        async def execute() -> None:
            try:
                if self.hold is not None:
                    await self.hold.wait()
                await asyncio.sleep(0.002)
                request_document = step.request.model_dump(mode="json")
                await self.bus.publish(
                    RunnerMessage(
                        runner_id,
                        "worker.completed",
                        {
                            "step_id": step.step_id,
                            "metrics": (
                                {"error_code": "failed"}
                                if step.step_id in self.fail
                                else {}
                            ),
                            "generated_text": "answer:" + step.step_id,
                            "request": request_document,
                        },
                    )
                )
            finally:
                self.active -= 1
                self.capacity.release()

        self.tasks.append(asyncio.create_task(execute()))

    async def cancel(self, runner_id: str) -> None:
        self.cancelled = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def steps(count: int) -> list[RequestStep]:
    return [
        RequestStep(
            str(index), RequestConfig(model="test", prompt=(f"question:{index}", 2))
        )
        for index in range(count)
    ]


def test_scheduler_capacity():
    async def run() -> None:
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 3)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(steps(9), datetime.now(timezone.utc)),
            scheduler,
            bus,
            2,
        )
        responses = await asyncio.wait_for(runtime.run(), 3)
        assert len(responses) == 9
        assert scheduler.peak == 3
        assert len({step.step_id for step in scheduler.submitted}) == 9

    asyncio.run(run())


def test_optional_window():
    async def run() -> None:
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 8)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(steps(12), datetime.now(timezone.utc)),
            scheduler,
            bus,
            2,
            max_outstanding_workers=2,
        )
        assert len(await asyncio.wait_for(runtime.run(), 3)) == 12
        assert scheduler.peak == 2

    asyncio.run(run())


def test_history_timer():
    async def run() -> None:
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store)
        scheduler = FakeScheduler(bus, 8)
        initial = steps(2)
        followup = RequestStep(
            "1", initial[1].request, ("0",), 0.02, message_sources=("0",)
        )
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic([initial[0], followup], datetime.now(timezone.utc)),
            scheduler,
            bus,
            2,
        )
        assert len(await asyncio.wait_for(runtime.run(), 3)) == 2
        assert scheduler.peak == 1
        assert any(message.topic == "timer.due" for message in store.messages.values())
        assert scheduler.submitted[1].request.sampling_params == {
            "messages": [
                {"role": "user", "content": "question:0"},
                {"role": "assistant", "content": "answer:0"},
                {"role": "user", "content": "question:1"},
            ]
        }

    asyncio.run(run())


def test_dependency_failure():
    async def run() -> None:
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 8)
        scheduler.fail.add("0")
        initial = steps(3)
        dependent = RequestStep("1", initial[1].request, ("0",))
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(
                [initial[0], dependent, initial[2]], datetime.now(timezone.utc)
            ),
            scheduler,
            bus,
            2,
        )
        assert len(await asyncio.wait_for(runtime.run(), 3)) == 2
        assert set(runtime.cancelled_steps) == {"1"}
        assert {step.step_id for step in scheduler.submitted} == {"0", "2"}

    asyncio.run(run())


def test_runner_cancels_chain():
    class PermissiveLogic(WorkflowTaskLogic):
        def eligibility(self, step, now, results):
            # Dependency safety belongs to Runner, even with permissive task logic.
            return Eligibility(True)

    async def run() -> None:
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store)
        scheduler = FakeScheduler(bus, 2)
        scheduler.fail.add("0")
        initial = steps(5)
        child = RequestStep(
            "1", initial[1].request, ("0",), 3600, message_sources=("0",)
        )
        descendant = RequestStep("2", initial[2].request, ("1",))
        merge = RequestStep(
            "4", initial[4].request, ("2", "3"), message_sources=("2", "3")
        )
        runtime = RunnerRuntime(
            "runner",
            PermissiveLogic(
                [initial[0], child, descendant, initial[3], merge],
                datetime.now(timezone.utc),
            ),
            scheduler,
            bus,
            2,
        )
        assert len(await asyncio.wait_for(runtime.run(), 3)) == 2
        assert {step.step_id for step in scheduler.submitted} == {"0", "3"}
        assert set(runtime.cancelled_steps) == {"1", "2", "4"}
        cancellations = [
            message
            for message in store.messages.values()
            if message.topic == "worker.cancelled"
        ]
        assert len(cancellations) == 3
        assert cancellations[0].payload == {
            "step_id": "1",
            "reason": "dependency_failed",
            "failed_dependencies": ["0"],
        }
        assert runtime.cancelled_steps["4"].failed_dependencies == ["2"]
        assert not runtime.timed_out
        assert not any(
            message.topic == "timer.due" for message in store.messages.values()
        )

    asyncio.run(run())


def test_repeated_fanin():
    from llmperf_backend.compiler.requests import compile_requests

    async def run() -> None:
        benchmark = {
            "repeat": 2,
            "steps": [
                {"name": "left"},
                {"name": "right"},
                {
                    "name": "merge",
                    "depends_on": [
                        {"step": "right", "reuse_messages": True},
                        {"step": "left", "reuse_messages": True},
                    ],
                },
            ],
        }
        compiled = compile_requests(benchmark, steps(6))
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 4)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(compiled, datetime.now(timezone.utc)),
            scheduler,
            bus,
            2,
        )
        assert len(await asyncio.wait_for(runtime.run(), 3)) == 6
        by_id = {step.step_id: step for step in scheduler.submitted}
        for repetition in range(2):
            request = by_id[f"{repetition}.merge"].request
            assert request.sampling_params == {
                "messages": [
                    {"role": "user", "content": f"question:{repetition * 3 + 1}"},
                    {"role": "assistant", "content": f"answer:{repetition}.right"},
                    {"role": "user", "content": f"question:{repetition * 3}"},
                    {"role": "assistant", "content": f"answer:{repetition}.left"},
                    {"role": "user", "content": f"question:{repetition * 3 + 2}"},
                ]
            }

    asyncio.run(run())


def test_cancel_backpressure():
    async def run() -> None:
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 1)
        scheduler.hold = asyncio.Event()
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(steps(100), datetime.now(timezone.utc)),
            scheduler,
            bus,
            0.02,
        )
        assert await asyncio.wait_for(runtime.run(), 1) == []
        assert runtime.timed_out
        assert scheduler.cancelled
        assert len(scheduler.submitted) == 1

    asyncio.run(run())


def test_mailbox_recovery():
    async def run() -> None:
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store, reconcile_seconds=0.005)
        receiver = asyncio.create_task(bus.receive("consumer"))
        await asyncio.sleep(0)
        message = RunnerMessage("runner", "custom.input", {"value": 7})
        # Bypass notify to prove periodic durable reconciliation repairs loss.
        await store.publish_message(message)
        assert await asyncio.wait_for(receiver, 1) == [message]
        await bus.publish(message)
        assert await bus.receive("consumer") == [message]
        await bus.acknowledge(message, "consumer")
        assert not await store.pending_messages("runner", "runner", "consumer", 10)
        assert await bus.receive("other-consumer") == [message]
        await bus.close()

    asyncio.run(run())


def test_periodic_messages():
    async def run() -> None:
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store)
        source = TimerSource(bus)
        origin = datetime.now(timezone.utc)
        await source.periodic("sample", origin, 0.001, 3)
        await source.periodic("sample", origin, 0.001, 3)
        assert len(store.messages) == 3
        assert all(message.topic == "timer.due" for message in store.messages.values())

    asyncio.run(run())


def test_timer_activation():
    async def run() -> None:
        origin = datetime.now(timezone.utc)
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store)
        scheduler = FakeScheduler(bus, 1)
        initial = steps(1)[0]
        scheduled = RequestStep(
            "0",
            initial.request,
            delay_seconds=0.03,
            due_at=origin + timedelta(seconds=0.01),
        )
        runtime = RunnerRuntime(
            "runner", WorkflowTaskLogic([scheduled], origin), scheduler, bus, 1
        )
        assert len(await asyncio.wait_for(runtime.run(), 2)) == 1
        events = list(store.messages.values())
        assert [event.topic for event in events] == ["timer.due", "worker.completed"]
        assert (
            events[0].payload["due_at"]
            == (origin + timedelta(seconds=0.03)).isoformat()
        )
        assert len(scheduler.submitted) == 1

    asyncio.run(run())


def test_trigger_anchor():
    origin = datetime.now(timezone.utc)
    parent = Completion(True, origin + timedelta(seconds=4))
    trigger = DependencyTrigger(("parent",), origin, 3)
    assert not trigger.evaluate(origin + timedelta(seconds=6), {"parent": parent}).ready
    assert trigger.evaluate(origin + timedelta(seconds=7), {"parent": parent}).ready
    assert trigger.evaluate(origin, {"parent": Completion(False, origin)}).blocked
    assert issubclass(ConversationMessageRecord, MessageRecord)


def test_default_configuration():
    configuration = BenchmarkConfig.model_validate(
        {
            "provider": "test",
            "model": "test",
            "steps": [
                {"name": "first"},
                {
                    "name": "next",
                    "depends_on": [{"step": "first", "reuse_messages": True}],
                },
            ],
        }
    )
    assert configuration.concurrent_requests is None
    assert configuration.steps is not None
    assert configuration.steps[1].message_sources == ("first",)


class MemoryJournal(MemoryMessages):
    def __init__(self):
        super().__init__()
        self.reserved: set[tuple[str, str]] = set()
        self.results: dict[tuple[str, str], WorkerResult] = {}

    async def reserve_worker(
        self, runner_id: str, step: RequestStep, window: Optional[int]
    ) -> bool:
        key = (runner_id, step.step_id)
        if key in self.reserved:
            return False
        self.reserved.add(key)
        return True

    async def commit_worker(self, runner_id: str, result: WorkerResult) -> bool:
        self.results[(runner_id, result.step_id)] = result
        payload = TypeAdapter(dict[str, JsonValue]).validate_python(
            {
                "step_id": result.step_id,
                "metrics": result.response[0],
                "generated_text": result.response[1],
                "request": result.response[2].model_dump(mode="json"),
            }
        )
        return await self.publish_message(
            RunnerMessage(
                runner_id, "worker.completed", payload, created_at=result.completed_at
            )
        )


def test_dispatch_pipeline():
    async def run() -> None:
        router = Mock()

        async def request(runner_id, configuration, environment):
            await asyncio.sleep(0)
            return ({}, "answer", configuration)

        router.execute.remote = AsyncMock(side_effect=request)
        router.cancel.remote = AsyncMock()
        journal = MemoryJournal()
        dispatcher = WorkerDispatcher(router, journal, 2)
        bus = dispatcher.register("runner", {})
        prepared = steps(7)
        prepared[6] = RequestStep(
            "6", prepared[6].request, ("0",), message_sources=("0",)
        )
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(prepared, datetime.now(timezone.utc)),
            dispatcher,
            bus,
            3,
        )
        responses = await asyncio.wait_for(runtime.run(), 4)
        assert len(responses) == len(journal.reserved) == len(journal.results) == 7
        assert router.execute.remote.await_count == 7
        assert len(journal.receipts) == 7
        assert all(
            response[0]["request_metadata"]["worker_id"] for response in responses
        )
        reused_metrics = journal.results[("runner", "6")].response[0]
        assert reused_metrics["local_input_tokens"] is None
        assert reused_metrics["number_input_tokens"] is None
        await dispatcher.unregister("runner")

    asyncio.run(run())

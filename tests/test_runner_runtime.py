"""Runner message-driven submission and dependency behavior without Provider calls."""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from unittest.mock import AsyncMock, Mock
from concurrent.futures import ThreadPoolExecutor

from pydantic import TypeAdapter, JsonValue

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.models import RequestConfig
from llmperf_backend.runtime.messages import (
    MessageReconciler,
    RunnerMessage,
    RunnerMessageBus,
)
from llmperf_backend.runtime.runner import RunnerRuntime, WorkflowTaskLogic
from llmperf_backend.runtime.runner import RunnerExecutionLog, WorkerResult
from llmperf_backend.runtime.dispatch import WorkerDispatcher
from llmperf_backend.runtime.admission import RunnerAdmission
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

    async def acknowledge_messages(self, runner_id, message_ids, consumer):
        for message_id in message_ids:
            await self.acknowledge_message(runner_id, message_id, consumer)

    async def pending_messages_batch(self, runner_ids, destination, consumer, limit):
        messages = []
        for runner_id in runner_ids:
            messages.extend(
                await self.pending_messages(runner_id, destination, consumer, limit)
            )
        return messages


class FakeScheduler:
    async def create_workers(
        self, runner_id: str, steps: Sequence[RequestStep]
    ) -> None:
        pass

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
        execution_log = RunnerExecutionLog("runner", 4096)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(steps(100), datetime.now(timezone.utc)),
            scheduler,
            bus,
            0.02,
            execution_log=execution_log,
        )
        assert await asyncio.wait_for(runtime.run(), 1) == []
        assert runtime.timed_out
        assert scheduler.cancelled
        assert len(scheduler.submitted) == 1
        assert "runner.timed_out" in execution_log.text()
        assert 'worker.submitted worker="0"' in execution_log.text()
        assert 'worker.submitted worker="1"' not in execution_log.text()

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


def test_committed_delivery():
    async def exercise():
        store = MemoryMessages()
        reconciler = MessageReconciler(store, interval_seconds=3600)
        bus = RunnerMessageBus("runner", store, reconciler=reconciler)
        await reconciler.close()
        receiver = asyncio.create_task(bus.receive("consumer"))
        await asyncio.sleep(0)
        commit_allowed = asyncio.Event()
        publish_started = asyncio.Event()
        original_publish = store.publish_message

        async def publish(message):
            publish_started.set()
            await commit_allowed.wait()
            return await original_publish(message)

        store.publish_message = publish
        message = RunnerMessage("runner", "output", {"value": 1})
        publisher = asyncio.create_task(bus.publish(message))
        await publish_started.wait()
        assert not receiver.done()
        assert not store.messages
        commit_allowed.set()
        assert await publisher
        # Recovery is stopped: committed delivery itself must wake the receiver.
        assert await asyncio.wait_for(receiver, 1) == [message]
        await bus.acknowledge(message, "consumer")
        await bus.close()

    asyncio.run(exercise())


def test_mailbox_overflow():
    async def exercise():
        store = MemoryMessages()
        reconciler = MessageReconciler(store)
        bus = RunnerMessageBus("runner", store, reconciler=reconciler, capacity=1)
        await reconciler.close()
        receiver = asyncio.create_task(bus.receive("consumer"))
        await asyncio.sleep(0)
        messages = [
            RunnerMessage("runner", "output", {"index": index}) for index in range(3)
        ]
        for message in messages:
            await bus.publish(message)
        assert await asyncio.wait_for(receiver, 1) == [messages[0]]
        for message in messages:
            assert await bus.receive("consumer") == [message]
            await bus.acknowledge(message, "consumer")
            await reconciler.reconcile()
        assert len(store.receipts) == 3
        await bus.close()

    asyncio.run(exercise())


def test_recovery_ack_race():
    async def exercise():
        store = MemoryMessages()
        reconciler = MessageReconciler(store)
        bus = RunnerMessageBus("runner", store, reconciler=reconciler)
        await reconciler.close()
        receiver = asyncio.create_task(bus.receive("consumer"))
        await asyncio.sleep(0)
        message = RunnerMessage("runner", "output", {})
        await bus.publish(message)
        assert await receiver == [message]
        snapshot_read = asyncio.Event()
        finish_read = asyncio.Event()
        original_read = store.pending_messages_batch

        async def pending(runner_ids, destination, consumer, limit):
            snapshot = await original_read(runner_ids, destination, consumer, limit)
            snapshot_read.set()
            await finish_read.wait()
            return snapshot

        store.pending_messages_batch = pending
        recovery = asyncio.create_task(reconciler.reconcile())
        await snapshot_read.wait()
        await bus.acknowledge(message, "consumer")
        finish_read.set()
        await recovery
        next_receive = asyncio.create_task(bus.receive("consumer"))
        await asyncio.sleep(0)
        assert (
            not next_receive.done()
        ), "stale recovery must not resurrect an acknowledged event"
        await bus.close()
        with pytest.raises(asyncio.CancelledError):
            await next_receive

    asyncio.run(exercise())


def test_batched_mailbox_recovery():
    async def exercise():
        store = MemoryMessages()
        reconciler = MessageReconciler(store)
        buses = [
            RunnerMessageBus(str(index), store, reconciler=reconciler)
            for index in range(130)
        ]
        await reconciler.close()
        receivers = [asyncio.create_task(bus.receive("consumer")) for bus in buses]
        await asyncio.sleep(0)
        for bus in buses:
            await store.publish_message(RunnerMessage(bus.runner_id, "output", {}))
        read_batches = AsyncMock(wraps=store.pending_messages_batch)
        store.pending_messages_batch = read_batches
        await reconciler.reconcile()
        received = await asyncio.wait_for(asyncio.gather(*receivers), 1)
        assert all(len(messages) == 1 for messages in received)
        assert read_batches.await_count == 3
        assert max(len(call.args[0]) for call in read_batches.call_args_list) == 64
        for bus in buses:
            await bus.close()

    asyncio.run(exercise())


def test_failed_receipt_retry():
    async def exercise():
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store)
        message = RunnerMessage("runner", "output", {})
        await bus.publish(message)
        assert await bus.receive("consumer") == [message]
        original_acknowledge = store.acknowledge_messages
        store.acknowledge_messages = AsyncMock(side_effect=RuntimeError("unavailable"))
        with pytest.raises(RuntimeError, match="unavailable"):
            await bus.acknowledge(message, "consumer")
        assert await bus.receive("consumer") == [message]
        store.acknowledge_messages = original_acknowledge
        await bus.acknowledge(message, "consumer")
        await bus.close()

    asyncio.run(exercise())


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
    assert configuration.concurrent_workers is None
    assert configuration.steps is not None
    assert configuration.steps[1].message_sources == ("first",)


class MemoryJournal(MemoryMessages):
    async def create_workers(
        self, runner_id: str, steps: Sequence[RequestStep]
    ) -> None:
        pass

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

    async def commit_worker(
        self, runner_id: str, result: WorkerResult
    ) -> Optional[RunnerMessage]:
        self.results[(runner_id, result.step_id)] = result
        payload = TypeAdapter(dict[str, JsonValue]).validate_python(
            {
                "step_id": result.step_id,
                "metrics": result.response[0],
                "generated_text": result.response[1],
                "request": result.response[2].model_dump(mode="json"),
            }
        )
        message = RunnerMessage(
            runner_id, "worker.completed", payload, created_at=result.completed_at
        )
        await self.publish_message(message)
        return message


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


def test_admission_fairness():
    """A submitting Runner cannot jump ahead of already waiting Runners."""

    async def run():
        releases = asyncio.Queue()
        starts = asyncio.Queue()
        router = Mock()

        async def request(runner_id, configuration, environment):
            await starts.put(runner_id)
            await releases.get()
            return {}, "", configuration

        router.execute.remote = AsyncMock(side_effect=request)
        dispatcher = WorkerDispatcher(router, MemoryJournal(), 1)
        for owner in ("large", "small", "other"):
            dispatcher.register(owner, {})
        prepared = steps(2)
        await dispatcher.submit("large", prepared[0], None)
        assert await asyncio.wait_for(starts.get(), 1) == "large"
        submissions = []
        for owner in ("small", "other", "large"):
            submissions.append(
                asyncio.create_task(dispatcher.submit(owner, prepared[1], None))
            )
            await asyncio.sleep(0)
        assert len(dispatcher.waiting) == 3
        for owner in ("small", "other", "large"):
            await releases.put(None)
            assert await asyncio.wait_for(starts.get(), 1) == owner
        await releases.put(None)
        await asyncio.gather(*submissions)
        await asyncio.gather(*dispatcher.active.values())
        for owner in ("large", "small", "other"):
            await dispatcher.unregister(owner)

    asyncio.run(run())


def test_preparation_yields(monkeypatch):
    """Activated Runners create only recipes while waiting for shared capacity."""
    import llmperf_backend.runtime.preparation as preparation
    from llmperf.datasets.tokenizer import TOKENIZER_PATH

    def forbidden(*args, **kwargs):
        raise AssertionError("Backend must not prepare inputs")

    monkeypatch.setattr(preparation, "_prepared_tokenizer", forbidden)
    monkeypatch.setattr(preparation, "_calculation_options", forbidden)
    monkeypatch.setattr(
        preparation,
        "aggregate_workers",
        lambda benchmark, responses, *args: (
            {"results": {}},
            [response[0] for response in responses],
        ),
    )

    async def run():
        gate = asyncio.Event()
        router = Mock()

        async def request(runner_id, configuration, environment):
            assert configuration.preparation is not None
            assert configuration.prompt[0] == ""
            await gate.wait()
            return {}, "answer", configuration

        router.execute.remote = AsyncMock(side_effect=request)
        router.cancel.remote = AsyncMock()
        dispatcher = WorkerDispatcher(router, MemoryJournal(), 2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            runners = [
                preparation.RunnerPreparation(str(index), executor, dispatcher)
                for index in range(24)
            ]
            for runner in runners:
                runner.start(
                    {
                        "repeat": 1,
                        "model": "test",
                        "adapter": "openai",
                        "mean_input_tokens": 10,
                        "timeout": 10,
                    },
                    {TOKENIZER_PATH: "unused"},
                    {},
                    4096,
                )
            try:

                async def all_created():
                    while any(runner.created_workers != 1 for runner in runners):
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(all_created(), 5)
                await asyncio.sleep(0.02)
                assert len(dispatcher.active) == 2
                assert len(dispatcher.waiting) == 22
                assert router.execute.remote.await_count == 2
                gate.set()
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*(runner.result_async() for runner in runners)),
                        5,
                    )
                except asyncio.TimeoutError:
                    pytest.fail(
                        str(
                            {
                                "phases": [
                                    (runner.runner_id, runner.phase, runner.ready())
                                    for runner in runners
                                ],
                                "active": list(dispatcher.active),
                                "waiting": list(dispatcher.waiting),
                                "errors": dispatcher.errors,
                            }
                        )
                    )
                assert all(result["ok"] for result in results)
                for runner, result in zip(runners, results):
                    output = result["stdout"]
                    assert "runner.started" in output
                    assert "workers.created count=1" in output
                    assert "worker.submitted" in output
                    assert 'state="succeeded"' in output
                    assert "runner.finished succeeded=1 failed=0" in output
                    assert "answer" not in output
                    assert "unused" not in output
                    assert all(
                        f'runner="{runner.runner_id}"' in line
                        for line in output.splitlines()
                    )
                assert all(
                    result["summary"]["worker_creation"]["created"] == 1
                    for result in results
                )
                assert not dispatcher.waiting
                assert not dispatcher.active
            finally:
                gate.set()
                for runner in runners:
                    runner.cancel()
                    await dispatcher.unregister(runner.runner_id)

    asyncio.run(run())


def test_execution_log_bounds():
    execution_log = RunnerExecutionLog("runner", 1024)
    request = RequestConfig(model="test", prompt=("private prompt", 2))
    result = WorkerResult(
        "worker\n\x1b[31m",
        (
            {
                "end_to_end_latency_s": 0.25,
                "ttft_s": float("nan"),
                "number_input_tokens": 2,
                "number_output_tokens": 3,
                "error_code": 429,
                "error_msg": "private credentials",
                "response_headers": {"authorization": "secret"},
            },
            "private answer",
            request,
        ),
        datetime.now(timezone.utc),
    )
    for _ in range(50):
        execution_log.completion(result)
    output = execution_log.text()
    assert len(output.encode("utf-8")) <= 1024
    assert output.startswith("[Earlier Runner events omitted")
    assert 'state="failed"' in output
    assert "error_code=429" in output
    assert "end_to_end_latency_s=0.25" in output
    assert "ttft_s=null" in output
    assert "private" not in output
    assert "secret" not in output
    assert "\x1b" not in output
    assert "worker\\n\\u001b[31m" in output


def test_dependency_logs():
    async def run():
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 1)
        scheduler.fail.add("0")
        initial = steps(2)
        child = RequestStep("1", initial[1].request, ("0",))
        execution_log = RunnerExecutionLog("runner", 4096)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic([initial[0], child], datetime.now(timezone.utc)),
            scheduler,
            bus,
            2,
            execution_log=execution_log,
        )
        await asyncio.wait_for(runtime.run(), 3)
        output = execution_log.text()
        assert 'worker.completed worker="0" state="failed"' in output
        assert 'worker.cancelled worker="1" reason="dependency_failed"' in output
        assert output.count("worker.completed") == 1
        assert output.count("worker.cancelled") == 1

    asyncio.run(run())


def test_preparation_failure_logs(monkeypatch):
    import llmperf_backend.runtime.preparation as preparation

    def fail_aggregation(*args):
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(preparation, "aggregate_workers", fail_aggregation)

    async def run():
        router = Mock()

        async def request(runner_id, configuration, environment):
            return {}, "answer", configuration

        router.execute.remote = AsyncMock(side_effect=request)
        router.cancel.remote = AsyncMock()
        dispatcher = WorkerDispatcher(router, MemoryJournal(), 1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            runner = preparation.RunnerPreparation("runner", executor, dispatcher)
            runner.start(
                {
                    "repeat": 1,
                    "model": "test",
                    "adapter": "openai",
                    "mean_input_tokens": 10,
                    "timeout": 2,
                },
                {},
                {},
                4096,
            )
            try:
                result = await asyncio.wait_for(runner.result_async(), 3)
                assert not result["ok"]
                assert "worker.completed" in result["stdout"]
                assert 'runner.failed error_type="RuntimeError"' in result["stdout"]
                assert "aggregation failed" in result["stderr"]
            finally:
                runner.close()
                await dispatcher.unregister("runner")

    asyncio.run(run())


@pytest.mark.parametrize("fail_first", [False, True])
def test_worker_cooldown(fail_first):
    async def run():
        store = MemoryMessages()
        bus = RunnerMessageBus("runner", store)
        scheduler = FakeScheduler(bus, 8)
        starts: list[datetime] = []
        original_submit = scheduler.submit

        async def record_submit(runner_id, step, window):
            starts.append(datetime.now(timezone.utc))
            await original_submit(runner_id, step, window)

        scheduler.submit = record_submit
        if fail_first:
            scheduler.fail.add("0")
        prepared = steps(3)
        assert all(not step.dependencies for step in prepared)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(prepared, datetime.now(timezone.utc)),
            scheduler,
            bus,
            2,
            cooldown_seconds=0.02,
        )
        assert len(await asyncio.wait_for(runtime.run(), 3)) == 3
        completions = [
            message.created_at
            for message in store.messages.values()
            if message.topic == "worker.completed"
        ]
        assert scheduler.peak == 1
        assert all(
            (starts[index + 1] - completions[index]).total_seconds() >= 0.02
            for index in range(2)
        )
        assert len(scheduler.submitted) == 3  # Failed requests are not retried.
        assert not runtime.cancelled_steps

    asyncio.run(run())


def test_cooldown_timeout():
    async def run():
        bus = RunnerMessageBus("runner", MemoryMessages())
        scheduler = FakeScheduler(bus, 8)
        runtime = RunnerRuntime(
            "runner",
            WorkflowTaskLogic(steps(3), datetime.now(timezone.utc)),
            scheduler,
            bus,
            0.05,
            cooldown_seconds=3600,
        )
        assert len(await asyncio.wait_for(runtime.run(), 1)) == 1
        assert runtime.timed_out
        assert scheduler.active == 0
        assert len(scheduler.submitted) == 1

    asyncio.run(run())


def test_rolling_admission():
    async def run():
        admission = RunnerAdmission(1)
        starts: list[str] = []
        runtimes = []

        async def execute(owner):
            await admission.acquire(owner)
            bus = RunnerMessageBus(owner, MemoryMessages())
            scheduler = FakeScheduler(bus, 1)
            original_submit = scheduler.submit

            async def submit(runner_id, step, window):
                assert admission.active == {owner}
                starts.append(owner)
                await original_submit(runner_id, step, window)

            scheduler.submit = submit
            runtime = RunnerRuntime(
                owner,
                WorkflowTaskLogic(steps(2), datetime.now(timezone.utc)),
                scheduler,
                bus,
                2,
                cooldown_seconds=0.05,
                admission=admission,
            )
            runtimes.append(runtime)
            try:
                return await runtime.run()
            finally:
                admission.release(owner)

        results = await asyncio.wait_for(
            asyncio.gather(execute("a"), execute("b"), execute("c")), 3
        )
        assert starts[:3] == ["a", "b", "c"]
        assert all(len(result) == 2 for result in results)
        assert all(not runtime.timed_out for runtime in runtimes)
        assert not admission.active

    asyncio.run(run())


def test_runtime_admission():
    async def run():
        releases = {owner: asyncio.Event() for owner in ("a1", "a2", "b")}
        started: asyncio.Queue[str] = asyncio.Queue()
        router = Mock()

        async def request(runner_id, configuration, environment):
            await started.put(runner_id)
            await releases[runner_id].wait()
            return {}, "", configuration

        router.execute.remote = AsyncMock(side_effect=request)
        router.cancel.remote = AsyncMock()
        dispatcher = WorkerDispatcher(router, MemoryJournal(), 2)
        for owner in ("a1", "a2"):
            dispatcher.register(owner, {}, runtime_limits={"a": 1})
        dispatcher.register("b", {})
        initial = steps(1)[0]
        await dispatcher.submit("a1", initial, None)
        assert await asyncio.wait_for(started.get(), 1) == "a1"
        waiting = asyncio.create_task(dispatcher.submit("a2", initial, None))
        await asyncio.sleep(0)
        await asyncio.wait_for(dispatcher.submit("b", initial, None), 1)
        assert await asyncio.wait_for(started.get(), 1) == "b"
        assert not waiting.done()
        releases["a1"].set()
        assert await asyncio.wait_for(started.get(), 1) == "a2"
        await waiting
        releases["a2"].set()
        releases["b"].set()
        await asyncio.gather(*dispatcher.active.values())
        for owner in releases:
            await dispatcher.unregister(owner)
        assert not dispatcher.runtime_gates

    asyncio.run(run())


def test_cancelled_runtime_permit():
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        router = Mock()

        async def request(runner_id, configuration, environment):
            started.set()
            await release.wait()
            return {}, "", configuration

        router.execute.remote = AsyncMock(side_effect=request)
        router.cancel.remote = AsyncMock()
        dispatcher = WorkerDispatcher(router, MemoryJournal(), 2)
        for owner in ("first", "waiting", "next"):
            dispatcher.register(owner, {}, runtime_limits={"a": 1})
        initial = steps(1)[0]
        await dispatcher.submit("first", initial, None)
        await asyncio.wait_for(started.wait(), 1)
        waiting = asyncio.create_task(dispatcher.submit("waiting", initial, None))
        await asyncio.sleep(0)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        cancelling = asyncio.create_task(dispatcher.cancel("first"))
        next_submission = asyncio.create_task(dispatcher.submit("next", initial, None))
        await asyncio.sleep(0.01)
        assert not next_submission.done()
        assert not cancelling.done()  # Physical execution still owns its permit.
        release.set()
        await asyncio.wait_for(cancelling, 1)
        await asyncio.wait_for(next_submission, 1)
        await asyncio.gather(*dispatcher.active.values())
        for owner in ("first", "waiting", "next"):
            await dispatcher.unregister(owner)

    asyncio.run(run())

"""Runtime dependencies, bounded admission, cancellation, and conversation ordering."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict

import pytest

from llmperf.clients.models import RequestConfig
from llmperf.clients.base import LLMResponse
from llmperf_backend.runtime.request_pool import RequestRouter
from llmperf_backend.runtime.coordinator import (
    NodeSnapshot,
    ready_nodes,
    ReconcileNotifier,
    ReconcileHint,
)
from llmperf_backend.compiler.tasks import TaskRecipe, TaskCompiler


def test_dependency_delay():
    now = datetime.now(timezone.utc)
    nodes = [
        NodeSnapshot("fast", (), 0, "emitted", now, "succeeded"),
        NodeSnapshot("slow", (), 0, "emitted", None, None),
        NodeSnapshot("next", ("fast",), 2, "blocked", None, None),
        NodeSnapshot("join", ("fast", "slow"), 0, "blocked", None, None),
    ]
    assert ready_nodes(nodes, now, now) == {}
    assert ready_nodes(nodes, now, now + timedelta(seconds=2)) == {
        "next": now + timedelta(seconds=2)
    }


def test_bus_bound():
    async def exercise():
        bus = ReconcileNotifier(1)
        first = ReconcileHint("first")
        bus.publish(first)
        bus.publish(first)
        bus.publish(ReconcileHint("second"))
        assert bus.queue.qsize() == 1
        assert await bus.receive(1) == first
        assert not bus.pending

    asyncio.run(exercise())


def test_shared_capacity():
    async def exercise():
        gate = asyncio.Event()
        started: list[str] = []

        class RemoteMethod:
            async def remote(
                self, request: RequestConfig, environment: Dict[str, str]
            ) -> LLMResponse:
                started.append(environment["key"])
                await gate.wait()
                return {}, environment["key"], request

        class Actor:
            @property
            def execute(self) -> RemoteMethod:
                return RemoteMethod()

        router = RequestRouter([Actor()], 2, 16)
        request = RequestConfig(model="test", prompt=("hi", 1))
        tasks = [
            asyncio.create_task(
                router.execute(str(index), request, {"key": str(index)})
            )
            for index in range(4)
        ]
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(started) == 2
        assert router.status()["executing_requests"] == 2
        assert router.status()["queued_requests"] == 2
        assert router.status()["runners"] == {
            "0": {"queued": 0, "running": 1},
            "1": {"queued": 0, "running": 1},
            "2": {"queued": 1, "running": 0},
            "3": {"queued": 1, "running": 0},
        }
        await router.cancel("3")
        gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert [item[1] for item in results[:3] if isinstance(item, tuple)] == [
            "0",
            "1",
            "2",
        ]
        assert isinstance(results[3], RuntimeError)
        assert router.status()["executing_requests"] == 0
        assert router.status()["runners"] == {}

    asyncio.run(exercise())


def test_runner_queue_fairness():
    """A deep Runner backlog cannot overtake another ready Runner's turns."""

    async def exercise():
        releases = asyncio.Queue()
        starts = asyncio.Queue()

        class RemoteMethod:
            async def remote(
                self, request: RequestConfig, environment: Dict[str, str]
            ) -> LLMResponse:
                await starts.put(environment["runner"])
                await releases.get()
                return {}, "", request

        class Actor:
            @property
            def execute(self) -> RemoteMethod:
                return RemoteMethod()

        router = RequestRouter([Actor()], 1, 32)
        request = RequestConfig(model="test", prompt=("hi", 1))
        initial = asyncio.create_task(
            router.execute("initial", request, {"runner": "initial"})
        )
        assert await asyncio.wait_for(starts.get(), 1) == "initial"
        owners = ["large"] * 6 + ["small"] * 2 + ["other"] * 2
        pending = [
            asyncio.create_task(router.execute(owner, request, {"runner": owner}))
            for owner in owners
        ]
        try:

            async def all_queued():
                while router.status()["queued_requests"] != len(owners):
                    await asyncio.sleep(0)

            await asyncio.wait_for(all_queued(), 1)
            observed = []
            for _ in owners:
                await releases.put(None)
                observed.append(await asyncio.wait_for(starts.get(), 1))
            assert observed == [
                "large",
                "small",
                "other",
                "large",
                "small",
                "other",
                "large",
                "large",
                "large",
                "large",
            ]
            await releases.put(None)
            await asyncio.gather(initial, *pending)
        finally:
            for task in [initial, *pending]:
                if not task.done():
                    task.cancel()
            for _ in range(len(owners) + 1):
                releases.put_nowait(None)
            await asyncio.gather(initial, *pending, return_exceptions=True)

    asyncio.run(exercise())


def test_message_dependencies():
    base = {"name": "merge", "payloads": {"p": {"seed": "p"}}}
    definition = {
        **base,
        "workflow": [
            {
                "parallel": {
                    "name": "sources",
                    "branches": [
                        {"invoke": {"name": "left", "payload": "p"}},
                        {"invoke": {"name": "right", "payload": "p"}},
                    ],
                }
            },
            {
                "invoke": {
                    "name": "merge",
                    "payload": "p",
                    "depends_on": [
                        {"step": "sources.right", "reuse_messages": True},
                        {"step": "sources.left", "reuse_messages": True},
                    ],
                }
            },
        ],
    }
    TaskRecipe.model_validate(definition)
    assert TaskCompiler.estimate_token_budget(definition, 10, 2) == 60
    with pytest.raises(ValueError, match="earlier node"):
        TaskRecipe.model_validate(
            {
                **base,
                "workflow": [
                    {
                        "invoke": {
                            "name": "bad",
                            "payload": "p",
                            "depends_on": ["missing"],
                        }
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        TaskRecipe.model_validate(
            {
                **base,
                "workflow": [
                    {"invoke": {"name": "old", "payload": "p", "conversation": "chat"}}
                ],
            }
        )


def test_cancelled_capacity():
    async def exercise():
        gate = asyncio.Event()

        class RemoteMethod:
            async def remote(
                self, request: RequestConfig, environment: Dict[str, str]
            ) -> LLMResponse:
                await gate.wait()
                return {}, "done", request

        class Actor:
            @property
            def execute(self) -> RemoteMethod:
                return RemoteMethod()

        router = RequestRouter([Actor()], 1, 4)
        request = RequestConfig(model="test", prompt=("hello", 1))
        first = asyncio.create_task(router.execute("first", request, {}))
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert router.status()["executing_requests"] == 1
        second = asyncio.create_task(router.execute("second", request, {}))
        await asyncio.sleep(0)
        assert router.status()["queued_requests"] == 1
        gate.set()
        completed = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(completed[0], asyncio.CancelledError)
        assert isinstance(completed[1], tuple)
        assert router.status()["executing_requests"] == 0

    asyncio.run(exercise())


def test_history_identity(monkeypatch):
    from llmperf.metrics import constants as common_metrics
    from llmperf.benchmark import throughput as benchmark
    from llmperf.datasets.prompts import PromptDatasetSource

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text != "answer"
            return list(text)

    monkeypatch.setattr(benchmark, "get_tokenizer", Tokenizer)
    monkeypatch.setattr(
        benchmark,
        "prepare_prompt_requests",
        lambda **kwargs: ([("hello", 5)], [{"source": "test"}]),
    )
    observed = []
    for history in (
        [],
        [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "answer"},
        ],
    ):
        _, results = benchmark.get_token_throughput_latencies(
            model="test",
            mean_input_tokens=5,
            stddev_input_tokens=0,
            mean_output_tokens=1,
            stddev_output_tokens=0,
            max_num_completed_requests=1,
            prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
            task_context={"payload_seed": 1, "message_sources": ["previous"]},
            input_messages=history,
            request_executor=lambda request: (
                {
                    common_metrics.ERROR_CODE: None,
                    common_metrics.E2E_LAT: 1.0,
                    common_metrics.TTFT: 0.1,
                },
                "answer",
                request,
            ),
        )
        observed.append(results[0][common_metrics.REQUEST_METADATA])
        assert results[0][common_metrics.NUM_INPUT_TOKENS] is None
    assert observed[0]["prompt_hash"] == observed[1]["prompt_hash"]
    assert observed[0]["messages_hash"] != observed[1]["messages_hash"]
    assert [item["history_version"] for item in observed] == [0, 2]


def test_targeted_reconciliation(monkeypatch):
    from unittest.mock import AsyncMock
    from llmperf_backend.runtime import coordinator as runtime_coordinator

    now = datetime.now(timezone.utc)
    instances = [
        {
            "instance_id": name,
            "state": "active",
            "created_at": now,
            "nodes": [
                {
                    "node_id": name,
                    "dependencies": ("unfinished",),
                    "delay_seconds": 0,
                    "state": "blocked",
                    "completed_at": None,
                    "terminal_status": None,
                }
            ],
        }
        for name in ("background", "probe")
    ]
    visited = []
    original_ready = runtime_coordinator.ready_nodes

    def tracked_ready(nodes, created_at, observed_at):
        visited.append(nodes[0].node_id)
        return original_ready(nodes, created_at, observed_at)

    monkeypatch.setattr(runtime_coordinator, "ready_nodes", tracked_ready)

    async def exercise():
        repository = AsyncMock()
        repository.runtime_snapshot.side_effect = [
            {"runners": [], "instances": instances},
            {"runners": [], "instances": instances[:1]},
            {"runners": [], "instances": instances},
        ]
        coordinator = runtime_coordinator.RuntimeCoordinator(repository)
        await coordinator.reconcile()
        assert visited == ["background", "probe"]
        visited.clear()
        await coordinator.reconcile("completed-background-runner")
        assert visited == ["background"]
        visited.clear()
        await coordinator.reconcile()
        assert visited == ["background", "probe"]

    asyncio.run(exercise())


def test_incremental_staging():
    from unittest.mock import AsyncMock
    from llmperf_backend.runtime.coordinator import RuntimeCoordinator

    async def exercise():
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        repository = AsyncMock()
        repository.runtime_snapshot.return_value = {
            "runners": [],
            "instances": [
                {
                    "instance_id": "campaign",
                    "state": "planned",
                    "created_at": datetime.now(timezone.utc),
                    "nodes": [
                        {
                            "node_id": node_id,
                            "dependencies": (),
                            "delay_seconds": 0,
                            "state": "pending",
                            "completed_at": None,
                            "terminal_status": None,
                        }
                        for node_id in ("first", "second")
                    ],
                }
            ],
        }

        async def activate(node_id, due_at):
            if node_id == "second":
                second_started.set()
                await release_second.wait()
            return {"runner_id": node_id, "campaign_id": "campaign"}

        repository.activate_node.side_effect = activate
        coordinator = RuntimeCoordinator(repository, ready_capacity=4096)
        reconciliation = asyncio.create_task(coordinator.reconcile())
        try:
            await asyncio.wait_for(second_started.wait(), 1)
            assert list(coordinator.queues["campaign"]) == ["first"]
            await asyncio.wait_for(coordinator.wait_ready(), 1)
        finally:
            release_second.set()
            await reconciliation
        assert list(coordinator.queues["campaign"]) == ["first", "second"]

    asyncio.run(exercise())


def test_materialization_rotation():
    from unittest.mock import AsyncMock
    from llmperf_backend.runtime.coordinator import RuntimeCoordinator

    async def exercise():
        repository = AsyncMock()
        repository.runtime_snapshot.return_value = {
            "runners": [],
            "instances": [
                {
                    "instance_id": instance_id,
                    "state": "planned",
                    "created_at": datetime.now(timezone.utc),
                    "nodes": [
                        {
                            "node_id": f"{instance_id}-{index}",
                            "dependencies": (),
                            "delay_seconds": 0,
                            "state": "pending",
                            "completed_at": None,
                            "terminal_status": None,
                        }
                        for index in range(3)
                    ],
                }
                for instance_id in ("first", "second", "third")
            ],
        }

        async def activate(node_id, due_at):
            return {"runner_id": node_id, "campaign_id": node_id.split("-")[0]}

        repository.activate_node.side_effect = activate
        # A zero time slice deterministically permits one activation per pass.
        coordinator = RuntimeCoordinator(repository, interval_seconds=0)
        for _ in range(3):
            await coordinator.reconcile()
        assert [call.args[0] for call in repository.activate_node.await_args_list] == [
            "first-0",
            "second-0",
            "third-0",
        ]
        assert set(coordinator.queues) == {"first", "second", "third"}

    asyncio.run(exercise())


def test_late_campaign():
    from unittest.mock import AsyncMock
    from llmperf_backend.runtime.coordinator import RuntimeCoordinator

    async def exercise():
        repository = AsyncMock()
        instances = [
            {
                "instance_id": instance_id,
                "state": "planned",
                "created_at": datetime.now(timezone.utc),
                "nodes": [
                    {
                        "node_id": f"{instance_id}-{index}",
                        "dependencies": (),
                        "delay_seconds": 0,
                        "state": "pending",
                        "completed_at": None,
                        "terminal_status": None,
                    }
                    for index in range(4)
                ],
            }
            for instance_id in ("existing", "new")
        ]
        repository.runtime_snapshot.side_effect = [
            {"runners": [], "instances": instances[:1]},
            {"runners": [], "instances": instances},
        ]

        async def activate(node_id, due_at):
            return {"runner_id": node_id, "campaign_id": node_id.split("-")[0]}

        repository.activate_node.side_effect = activate
        coordinator = RuntimeCoordinator(repository, interval_seconds=0)
        await coordinator.reconcile()
        await coordinator.reconcile()
        assert [call.args[0] for call in repository.activate_node.await_args_list] == [
            "existing-0",
            "new-0",
        ]
        assert len(coordinator.queues["existing"]) == 1
        assert len(coordinator.queues["new"]) == 1

    asyncio.run(exercise())

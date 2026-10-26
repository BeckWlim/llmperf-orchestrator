"""Scheduler runtime ownership and Scheduler→Runner→Worker wiring tests."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import pytest
from typing import Any, Dict, Optional, Sequence, Tuple

from llmperf.metrics import constants as common_metrics
from llmperf.datasets.tokenizer import TOKENIZER_FAST, TOKENIZER_PATH
import llmperf_backend.runtime.scheduler as scheduler_module
from llmperf_backend.configuration.models import DatabaseConfig, SchedulerConfig
from llmperf_backend.artifacts.outbound import RAY_GRPC_ENABLE_HTTP_PROXY
from llmperf_backend.providers.registry import ProviderRegistry
from llmperf_backend.providers.files import ProviderFileProfile
from llmperf_backend.runtime.scheduler import (
    Scheduler,
    WORKER_RAY_ACTOR_CPUS,
)
from llmperf_backend.artifacts.resolver import DatasetResolution
from llmperf_backend.artifacts.resolver import TokenizerResolution
from llmperf_backend.runtime.worker import Worker, WORKER_DATASET_PATH


class UnusedRepository:
    """Complete Scheduler repository stub that rejects unexpected persistence calls."""

    async def reserve_worker(self, runner_id, step, window) -> bool:
        raise AssertionError("reserve_worker is not expected")

    async def commit_worker(self, runner_id, result) -> bool:
        raise AssertionError("commit_worker is not expected")

    async def publish_message(self, message) -> bool:
        raise AssertionError("publish_message is not expected")

    async def pending_messages(self, runner_id, destination, consumer, limit):
        raise AssertionError("pending_messages is not expected")

    async def acknowledge_message(self, runner_id, message_id, consumer) -> None:
        raise AssertionError("acknowledge_message is not expected")

    async def runtime_snapshot(self, runner_id: Optional[str] = None) -> Dict[str, Any]:
        raise AssertionError("runtime_snapshot is not expected")

    async def activate_node(self, node_id, due_at) -> Optional[Dict[str, Any]]:
        raise AssertionError("activate_node is not expected")

    async def start_selected(
        self, runner_id: str, owner: str
    ) -> Optional[Dict[str, Any]]:
        raise AssertionError("start_selected is not expected")

    async def acquire_runtime(self, owner: str) -> None:
        raise AssertionError("acquire_runtime is not expected")

    async def release_runtime(self) -> None:
        raise AssertionError("release_runtime is not expected")

    async def requeue_stale(self, stale_after_seconds: int) -> int:
        raise AssertionError("requeue_stale is not expected in this test")

    async def claim_next(self, scheduler_id: str) -> Optional[Dict[str, Any]]:
        raise AssertionError("claim_next is not expected in this test")

    async def heartbeat(self, runner_id: str) -> bool:
        raise AssertionError("heartbeat is not expected in this test")

    async def complete_runner(
        self,
        runner_id: str,
        summary: Dict[str, Any],
        request_metrics: Sequence[Dict[str, Any]],
        exit_code: int,
        stdout: str,
        stderr: str,
        terminal_status: str = "succeeded",
        error_message: Optional[str] = None,
    ) -> bool:
        raise AssertionError("complete_runner is not expected in this test")

    async def finish_runner(
        self,
        runner_id: str,
        status: str,
        message: str,
        exit_code: Optional[int],
        stdout: str,
        stderr: str,
    ) -> bool:
        raise AssertionError("finish_runner is not expected in this test")

    async def get_runner(self, runner_id: str) -> Optional[Dict[str, Any]]:
        raise AssertionError("get_runner is not expected in this test")

    async def requeue_runner(self, runner_id: str, message: str) -> None:
        raise AssertionError("requeue_runner is not expected in this test")


class FakeTokenizerCache:
    def __init__(self, path: Path):
        self.path = path
        self.spec = None

    async def resolve(self, spec):
        self.spec = spec
        return TokenizerResolution(
            source="huggingface",
            tokenizer_id=spec["id"],
            revision=spec["revision"],
            use_fast=spec["use_fast"],
            path=self.path,
            cached=True,
        )


class FakeDatasetCache:
    def __init__(self, path: Path):
        self.path = path
        self.spec = None

    async def resolve(self, spec):
        self.spec = spec
        return DatasetResolution(
            source="huggingface",
            dataset_id=spec["id"],
            filename=spec["filename"],
            revision=spec["revision"],
            adapter=spec["adapter"],
            path=self.path,
            cached=True,
        )


def test_worker_status(tmp_path: Path):
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )

    assert scheduler.status()["status"] == "stopped"
    assert scheduler.status()["worker_kind"] == "atomic_requests"
    assert scheduler.status()["active_workers"] == 0
    assert scheduler.status()["busy_slots"] == 0
    assert scheduler.status()["live_slots"] == 0


@pytest.mark.parametrize("runner_capacity", [2048, 4096])
def test_large_staging(runner_capacity: int):
    from datetime import datetime, timezone

    class StagingRepository(UnusedRepository):
        async def runtime_snapshot(self, runner_id=None):
            return {
                "runners": [
                    {"runner_id": f"background-{index}", "campaign_id": "background"}
                    for index in range(1536)
                ],
                "instances": [
                    {
                        "instance_id": "second-campaign",
                        "state": "planned",
                        "created_at": datetime.now(timezone.utc),
                        "nodes": [
                            {
                                "node_id": "second-request",
                                "dependencies": (),
                                "delay_seconds": 0,
                                "state": "pending",
                                "completed_at": None,
                                "terminal_status": None,
                            }
                        ],
                    }
                ],
            }

        async def activate_node(self, node_id, due_at):
            return {"runner_id": node_id, "campaign_id": "second-campaign"}

    async def exercise():
        scheduler = Scheduler(
            StagingRepository(),
            SchedulerConfig(max_concurrent_runners=runner_capacity),
            DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
            ProviderRegistry([], allow_empty=True),
        )
        await scheduler.coordinator.reconcile()
        assert len(scheduler.coordinator.queues["background"]) == 1536
        assert list(scheduler.coordinator.queues["second-campaign"]) == [
            "second-request"
        ]

    asyncio.run(exercise())


def test_tokenizer_injection(tmp_path: Path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    cache = FakeTokenizerCache(tokenizer_directory)
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
        cache,
    )
    runner = {
        "benchmark": {
            "provider": "test",
            "tokenizer": {
                "source": "huggingface",
                "id": "organization/tokenizer",
                "revision": "commit-1",
                "use_fast": False,
            },
        }
    }

    environment = asyncio.run(scheduler.worker_environment(runner, {"KEEP": "yes"}))

    assert environment["KEEP"] == "yes"
    assert environment[TOKENIZER_PATH] == str(tokenizer_directory)
    assert environment[TOKENIZER_FAST] == "false"
    assert cache.spec == runner["benchmark"]["tokenizer"]


def test_dataset_injection(tmp_path: Path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text("[]", encoding="utf-8")
    cache = FakeDatasetCache(dataset_path)
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
        dataset_cache=cache,
    )
    runner = {
        "benchmark": {
            "provider": "test",
            "dataset": {
                "source": "huggingface",
                "id": "organization/sharegpt",
                "filename": "sharegpt.json",
                "revision": "commit-1",
                "adapter": "sharegpt",
            },
        }
    }

    environment = asyncio.run(scheduler.worker_environment(runner, {"KEEP": "yes"}))

    assert environment["KEEP"] == "yes"
    assert environment[WORKER_DATASET_PATH] == str(dataset_path)
    assert cache.spec == runner["benchmark"]["dataset"]


def test_actor_environment(tmp_path: Path):
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(
            max_concurrent_runners=4,
            ray_address="ray://127.0.0.1:10001",
            ray_actor_max_concurrency=8,
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )
    runner = {
        "runner_id": "runner-1",
        "benchmark": {"provider": "test"},
    }

    environment = asyncio.run(scheduler.worker_environment(runner, {}))

    assert environment[WORKER_RAY_ACTOR_CPUS] == "1.0"
    assert environment["LLMPERF_WORKER_RAY_ACTOR_CONCURRENCY"] == "8"
    assert scheduler.status()["ray_actor_max_concurrency"] == 8
    assert scheduler.status()["ray_mode"] == "external"
    assert scheduler.status()["ray_address"] == "ray://127.0.0.1:10001"


def test_embedded_ray(tmp_path: Path, monkeypatch):
    calls = {}

    async def start_pool(self, ray):
        return None

    monkeypatch.setattr(Scheduler, "_start_request_pool", start_pool)
    monkeypatch.setenv(RAY_GRPC_ENABLE_HTTP_PROXY, "true")

    def initialize(**options):
        calls["init"] = options
        return SimpleNamespace(address_info={"address": "127.0.0.1:6379"})

    async def run_synchronously(function, *args, **kwargs):
        return function(*args, **kwargs)

    fake_ray = SimpleNamespace(
        init=initialize,
        remote=lambda **options: lambda function: calls.setdefault(
            "remote", (options, function)
        ),
        nodes=lambda: [{"Alive": True}],
        cluster_resources=lambda: {"CPU": 8.0, "object_store_memory": 100.0},
        available_resources=lambda: {"CPU": 8.0, "object_store_memory": 100.0},
        shutdown=lambda: calls.setdefault("shutdown", True),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(asyncio, "to_thread", run_synchronously)
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )

    async def exercise():
        scheduler._stop = asyncio.Event()
        await scheduler._start_ray_runtime()
        status = scheduler.status()
        monitor_task = scheduler._ray_monitor_task
        if monitor_task is None:
            raise RuntimeError("Ray monitor task was not started")
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        scheduler._ray_monitor_task = None
        await scheduler._stop_ray_runtime()
        scheduler._stop = None
        return status

    status = asyncio.run(exercise())

    assert calls["init"]["include_dashboard"] is False
    assert os.environ[RAY_GRPC_ENABLE_HTTP_PROXY] == "0"
    assert calls["init"]["num_cpus"] == 8
    assert calls["init"]["object_store_memory"] == 268_435_456
    assert status["ray_mode"] == "embedded"
    assert status["worker_kind"] == "atomic_requests"
    assert status["ray_runtime"]["status"] == "healthy"
    assert status["ray_runtime"]["object_store_available_ratio"] == 1.0
    assert status["ray_runtime"]["claim_blocked"] is False
    assert calls["remote"][0] == {"num_cpus": 0, "max_retries": 0}
    assert calls["shutdown"] is True


class ExecutionRepository(UnusedRepository):
    def __init__(self, cancel: bool = False):
        self.cancel = cancel
        self.completed: Optional[
            Tuple[
                str,
                Dict[str, Any],
                Sequence[Dict[str, Any]],
                int,
                str,
                str,
                str,
                Optional[str],
            ]
        ] = None
        self.finished: Optional[Tuple[str, str, str, Optional[int], str, str]] = None

    async def heartbeat(self, runner_id: str) -> bool:
        return self.cancel

    async def complete_runner(
        self,
        runner_id: str,
        summary: Dict[str, Any],
        request_metrics: Sequence[Dict[str, Any]],
        exit_code: int,
        stdout: str,
        stderr: str,
        terminal_status: str = "succeeded",
        error_message: Optional[str] = None,
    ) -> bool:
        self.completed = (
            runner_id,
            summary,
            request_metrics,
            exit_code,
            stdout,
            stderr,
            terminal_status,
            error_message,
        )
        return True

    async def finish_runner(
        self,
        runner_id: str,
        status: str,
        message: str,
        exit_code: Optional[int],
        stdout: str,
        stderr: str,
    ) -> bool:
        self.finished = (runner_id, status, message, exit_code, stdout, stderr)
        return True

    async def get_runner(self, runner_id: str) -> Optional[Dict[str, Any]]:
        return {"status": "running", "cancel_requested": self.cancel}


def _execution_runner():
    return {
        "runner_id": "runner-1",
        "campaign_id": "campaign-1",
        "metadata": {"suite": "worker-chain"},
        "benchmark": {
            "provider": "test",
            "model": "test-model",
            "adapter": "openai",
            "timeout": 10,
            "repeat": 1,
            "concurrent_workers": 1,
            "mean_input_tokens": 10,
            "stddev_input_tokens": 0,
            "mean_output_tokens": 1,
            "stddev_output_tokens": 0,
            "additional_sampling_params": {},
        },
    }


@pytest.mark.parametrize("batch", [False, True])
def test_worker_execution(tmp_path: Path, monkeypatch, batch):
    calls = {}

    class FakeWorker:
        def __init__(self, ray, remote, runner_id, actor_count, actor_cpus):
            calls["created"] = (runner_id, actor_count, actor_cpus)
            self.task_ref = None

        def start(
            self, benchmark, environment, runtime, log_limit, request_router=None
        ):
            calls["started"] = (benchmark, environment, runtime, log_limit)
            self.task_ref = object()

        def ready(self):
            return self.task_ref is not None

        async def result_async(self):
            await asyncio.sleep(0)
            return {
                "ok": True,
                "summary": {
                    "results": {
                        common_metrics.NUM_REQ_STARTED: 1,
                        common_metrics.NUM_COMPLETED_REQUESTS: 1,
                        common_metrics.NUM_ERRORS: 0,
                    },
                    "execution_runtime": {},
                },
                "requests": [{}],
                "stdout": "worker output",
                "stderr": "",
            }

        def task_id(self):
            return "task-1"

        def cancel(self, force=False):
            calls["cancelled"] = force

        def close(self):
            calls["closed"] = True

    repository = ExecutionRepository()
    scheduler = Scheduler(
        repository,
        SchedulerConfig(poll_interval_seconds=0.001),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )
    scheduler._ray_module = object()
    scheduler._worker_remote = object()
    monkeypatch.setattr(scheduler_module, "Worker", FakeWorker)
    if batch:
        # A non-null router enables the atomic path; batch lineage must still select Worker.
        monkeypatch.setattr(scheduler, "_request_router", SimpleNamespace())

    async def exercise():
        scheduler._stop = asyncio.Event()
        runner = _execution_runner()
        if batch:
            runner["benchmark"]["task_context"] = {"execution_kind": "batch"}
        await scheduler._execute(runner)
        scheduler._stop = None

    asyncio.run(exercise())

    assert calls["created"] == ("runner-1", 1, 1.0)
    assert calls["started"][2]["worker_kind"] == "ray_task"
    assert calls["started"][2]["resource_scheduling"] == "shared_request_pool"
    assert calls["started"][2]["campaign_id"] == "campaign-1"
    assert repository.completed is not None
    assert repository.completed[0] == "runner-1"
    assert repository.completed[1]["execution_runtime"]["worker_id"] == "task-1"
    assert repository.finished is None
    assert calls["closed"] is True
    assert scheduler.status()["active_workers"] == 0


def test_async_worker_wait():
    listener_started = threading.Event()
    release_listener = threading.Event()

    def get_result(_task_ref):
        listener_started.set()
        release_listener.wait(timeout=1)
        return {"ok": True}

    repository = ExecutionRepository(cancel=True)
    scheduler = Scheduler(
        repository,
        SchedulerConfig(poll_interval_seconds=0.001),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )
    worker = Worker(SimpleNamespace(get=get_result), object(), "runner-1", 1, 1.0)
    worker.task_ref = object()

    async def exercise():
        scheduler._stop = asyncio.Event()
        payload = await scheduler._wait_worker(worker, "runner-1")
        release_listener.set()
        await asyncio.sleep(0)
        scheduler._stop = None
        return payload

    payload = asyncio.run(exercise())

    assert payload is None
    assert listener_started.is_set()


def test_preworker_cancel(tmp_path: Path, monkeypatch):
    repository = ExecutionRepository(cancel=True)
    scheduler = Scheduler(
        repository,
        SchedulerConfig(
            poll_interval_seconds=0.001,
            artifact_resolution_timeout_seconds=1,
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )

    async def block_environment(runner, environment):
        await asyncio.Event().wait()

    monkeypatch.setattr(scheduler, "worker_environment", block_environment)

    asyncio.run(scheduler._execute(_execution_runner()))

    assert repository.finished is not None
    assert repository.finished[:3] == (
        "runner-1",
        "cancelled",
        "Benchmark cancelled before Worker start",
    )
    assert scheduler.status()["active_workers"] == 0


def test_artifact_timeout(tmp_path: Path, monkeypatch):
    repository = ExecutionRepository()
    scheduler = Scheduler(
        repository,
        SchedulerConfig(
            poll_interval_seconds=0.001,
            artifact_resolution_timeout_seconds=0.002,
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry(
            [
                ProviderFileProfile(base_url="http://127.0.0.1:8001/v1").resolve(
                    "test", {}
                )
            ]
        ),
    )

    async def block_environment(runner, environment):
        await asyncio.Event().wait()

    monkeypatch.setattr(scheduler, "worker_environment", block_environment)

    asyncio.run(scheduler._execute(_execution_runner()))

    assert repository.finished is not None
    assert repository.finished[0] == "runner-1"
    assert repository.finished[1] == "failed"
    assert "artifact resolution exceeded" in repository.finished[2]
    assert scheduler.status()["active_workers"] == 0

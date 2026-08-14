"""Scheduler runtime ownership and Scheduler→Runner→Worker wiring tests."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys

from llmperf import common_metrics
from llmperf.utils import TOKENIZER_FAST, TOKENIZER_PATH
import llmperf_backend.scheduler as scheduler_module
from llmperf_backend.models import DatabaseConfig, SchedulerConfig
from llmperf_backend.providers import ProviderRegistry
from llmperf_backend.scheduler import (
    Scheduler,
    WORKER_RAY_ACTOR_CPUS,
)
from llmperf_backend.tokenizers import TokenizerResolution
from llmperf_backend.datasets import DatasetResolution, WORKER_DATASET_PATH


class UnusedRepository:
    pass


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
            format=spec["format"],
            path=self.path,
            cached=True,
        )


def test_worker_status(tmp_path: Path):
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(
            working_directory=str(tmp_path),
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )

    assert scheduler.status()["status"] == "stopped"
    assert scheduler.status()["worker_kind"] == "ray_task"
    assert scheduler.status()["worker_module"] == "llmperf_backend.worker"
    assert scheduler.status()["active_workers"] == 0
    assert scheduler.status()["busy_slots"] == 0
    assert scheduler.status()["live_slots"] == 0


def test_tokenizer_injection(tmp_path: Path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    cache = FakeTokenizerCache(tokenizer_directory)
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(working_directory=str(tmp_path)),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
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
        SchedulerConfig(working_directory=str(tmp_path)),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
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
                "format": "sharegpt",
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
            working_directory=str(tmp_path),
            max_concurrent_runners=4,
            ray_address="ray://127.0.0.1:10001",
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )
    runner = {
        "runner_id": "runner-1",
        "benchmark": {"provider": "test"},
    }

    environment = asyncio.run(scheduler.worker_environment(runner, {}))

    assert environment[WORKER_RAY_ACTOR_CPUS] == "1.0"
    assert scheduler.status()["ray_mode"] == "external"
    assert scheduler.status()["ray_address"] == "ray://127.0.0.1:10001"


def test_embedded_ray(tmp_path: Path, monkeypatch):
    calls = {}

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
        SchedulerConfig(working_directory=str(tmp_path)),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )

    async def exercise():
        scheduler._stop = asyncio.Event()
        await scheduler._start_ray_runtime()
        status = scheduler.status()
        scheduler._ray_monitor_task.cancel()
        await asyncio.gather(scheduler._ray_monitor_task, return_exceptions=True)
        scheduler._ray_monitor_task = None
        await scheduler._stop_ray_runtime()
        scheduler._stop = None
        return status

    status = asyncio.run(exercise())

    assert calls["init"]["include_dashboard"] is False
    assert calls["init"]["num_cpus"] == 8
    assert calls["init"]["object_store_memory"] == 268_435_456
    assert status["ray_mode"] == "embedded"
    assert status["worker_kind"] == "ray_task"
    assert status["ray_runtime"]["status"] == "healthy"
    assert status["ray_runtime"]["object_store_available_ratio"] == 1.0
    assert status["ray_runtime"]["claim_blocked"] is False
    assert calls["remote"][0] == {"num_cpus": 0, "max_retries": 0}
    assert calls["shutdown"] is True


class ExecutionRepository:
    def __init__(self, cancel=False):
        self.cancel = cancel
        self.completed = None
        self.finished = None

    async def heartbeat(self, runner_id):
        return self.cancel

    async def complete_runner(self, *arguments, **options):
        self.completed = (arguments, options)
        return True

    async def finish_runner(self, *arguments):
        self.finished = arguments
        return True

    async def get_runner(self, runner_id):
        return {"status": "running", "cancel_requested": self.cancel}


def _execution_runner():
    return {
        "runner_id": "runner-1",
        "campaign_id": "campaign-1",
        "metadata": {"suite": "worker-chain"},
        "benchmark": {
            "provider": "test",
            "model": "test-model",
            "llm_api": "openai",
            "timeout_seconds": 10,
            "max_completed_requests": 1,
            "concurrent_requests": 1,
            "mean_input_tokens": 10,
            "stddev_input_tokens": 0,
            "mean_output_tokens": 1,
            "stddev_output_tokens": 0,
            "additional_sampling_params": {},
        },
    }


def test_worker_execution(tmp_path: Path, monkeypatch):
    calls = {}

    class FakeWorker:
        def __init__(self, ray, remote, runner_id, actor_count, actor_cpus):
            calls["created"] = (runner_id, actor_count, actor_cpus)
            self.task_ref = None

        def start(self, benchmark, environment, runtime, log_limit):
            calls["started"] = (benchmark, environment, runtime, log_limit)
            self.task_ref = object()

        def ready(self):
            return self.task_ref is not None

        def result(self):
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
        SchedulerConfig(working_directory=str(tmp_path), poll_interval_seconds=0.001),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )
    scheduler._ray_module = object()
    scheduler._worker_remote = object()
    monkeypatch.setattr(scheduler_module, "Worker", FakeWorker)

    async def exercise():
        scheduler._stop = asyncio.Event()
        await scheduler._execute(_execution_runner())
        scheduler._stop = None

    asyncio.run(exercise())

    assert calls["created"] == ("runner-1", 1, 1.0)
    assert calls["started"][2]["worker_kind"] == "ray_task"
    assert calls["started"][2]["resource_scheduling"] == "independent_actors"
    assert calls["started"][2]["campaign_id"] == "campaign-1"
    assert repository.completed[0][0] == "runner-1"
    assert repository.completed[0][1]["execution_runtime"]["worker_id"] == "task-1"
    assert repository.finished is None
    assert calls["closed"] is True
    assert scheduler.status()["active_workers"] == 0


def test_preworker_cancel(tmp_path: Path, monkeypatch):
    repository = ExecutionRepository(cancel=True)
    scheduler = Scheduler(
        repository,
        SchedulerConfig(
            working_directory=str(tmp_path),
            poll_interval_seconds=0.001,
            artifact_resolution_timeout_seconds=1,
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )

    async def block_environment(runner, environment):
        await asyncio.Event().wait()

    monkeypatch.setattr(scheduler, "worker_environment", block_environment)

    asyncio.run(scheduler._execute(_execution_runner()))

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
            working_directory=str(tmp_path),
            poll_interval_seconds=0.001,
            artifact_resolution_timeout_seconds=0.002,
        ),
        DatabaseConfig(url="postgresql+asyncpg:///unused_test"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )

    async def block_environment(runner, environment):
        await asyncio.Event().wait()

    monkeypatch.setattr(scheduler, "worker_environment", block_environment)

    asyncio.run(scheduler._execute(_execution_runner()))

    assert repository.finished[0] == "runner-1"
    assert repository.finished[1] == "failed"
    assert "artifact resolution exceeded" in repository.finished[2]
    assert scheduler.status()["active_workers"] == 0

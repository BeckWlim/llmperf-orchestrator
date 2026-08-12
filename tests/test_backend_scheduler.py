import asyncio
from pathlib import Path

from llmperf_backend.models import DatabaseConfig, SchedulerConfig
from llmperf_backend.providers import ProviderRegistry
from llmperf_backend.scheduler import Scheduler, WORKER_DATABASE_URL_ENV
from llmperf_backend.tokenizers import (
    TokenizerResolution,
    WORKER_TOKENIZER_PATH_ENV,
    WORKER_TOKENIZER_USE_FAST_ENV,
)
from llmperf_backend.datasets import DatasetResolution, WORKER_DATASET_PATH_ENV


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


def test_worker_command(tmp_path: Path):
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(
            working_directory=str(tmp_path),
        ),
        DatabaseConfig(url="sqlite+aiosqlite:///unused.db"),
        ProviderRegistry.from_environment(
            {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
        ),
    )

    command = scheduler.build_command("runner-1; touch unsafe")

    assert command[-2:] == ["--runner-id", "runner-1; touch unsafe"]
    assert command[1:3] == ["-m", "llmperf_backend.worker"]
    assert WORKER_DATABASE_URL_ENV not in " ".join(command)
    assert scheduler.status()["status"] == "stopped"
    assert scheduler.status()["active_slots"] == 0


def test_tokenizer_injection(tmp_path: Path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    cache = FakeTokenizerCache(tokenizer_directory)
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(working_directory=str(tmp_path)),
        DatabaseConfig(url="sqlite+aiosqlite:///unused.db"),
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
    assert environment[WORKER_TOKENIZER_PATH_ENV] == str(tokenizer_directory)
    assert environment[WORKER_TOKENIZER_USE_FAST_ENV] == "false"
    assert cache.spec == runner["benchmark"]["tokenizer"]


def test_dataset_injection(tmp_path: Path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text("[]", encoding="utf-8")
    cache = FakeDatasetCache(dataset_path)
    scheduler = Scheduler(
        UnusedRepository(),
        SchedulerConfig(working_directory=str(tmp_path)),
        DatabaseConfig(url="sqlite+aiosqlite:///unused.db"),
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
    assert environment[WORKER_DATASET_PATH_ENV] == str(dataset_path)
    assert cache.spec == runner["benchmark"]["dataset"]

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

import httpx

from llmperf_backend.app import create_app
from llmperf_backend.config import ConfigStore
from llmperf_backend.artifacts import DatasetResolution
from llmperf_backend.providers import ProviderRegistry
from llmperf_backend.artifacts import TokenizerResolution


CONFIG = """
version: "1.0.0"
environment: test
database:
  url: postgresql+asyncpg:///unused_test
  auto_create_schema: false
scheduler:
  enabled: false
planner:
  enabled: false
benchmark:
  provider: test
  model: test-model
"""


class FakeTokenizerCache:
    def __init__(self, path: Path):
        self.path = path

    async def resolve(self, spec: Mapping[str, Any]) -> TokenizerResolution:
        return TokenizerResolution(
            source="huggingface",
            tokenizer_id=str(spec["id"]),
            revision="a" * 40,
            use_fast=bool(spec.get("use_fast", True)),
            path=self.path,
            cached=True,
        )


class FakeDatasetCache:
    def __init__(self, path: Path):
        self.path = path

    async def resolve(self, spec: Mapping[str, Any]) -> DatasetResolution:
        return DatasetResolution(
            source="huggingface",
            dataset_id=str(spec["id"]),
            filename=str(spec["filename"]),
            revision="b" * 40,
            adapter=str(spec["adapter"]),
            path=self.path,
            cached=True,
        )


def test_artifact_api(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets-cache"))
    config_path = tmp_path / "backend.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    tokenizer_directory.joinpath("tokenizer.json").write_text(
        '{"version":"1.0"}', encoding="utf-8"
    )
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        '[{"conversations": [{"from": "human", "value": "prompt"}]}]',
        encoding="utf-8",
    )
    application = create_app(
        ConfigStore(config_path),
        provider_registry=ProviderRegistry.from_environment(
            {
                "LLMPERF_PROVIDER_TEST_DISCOVERY": "static",
                "LLMPERF_PROVIDER_TEST_MODELS": "test-model",
            }
        ),
        tokenizer_cache=FakeTokenizerCache(tokenizer_directory),
        dataset_cache=FakeDatasetCache(dataset_path),
    )

    async def request_validation() -> httpx.Response:
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://testserver",
            ) as client:
                request_coroutine = client.post(
                    "/api/v1/campaigns/validate-artifacts",
                    json={
                        "campaign": {"name": "artifact-preflight"},
                        "runners": [
                            {
                                "benchmark": {
                                    "provider": "test",
                                    "model": "test-model",
                                    "tokenizer": {
                                        "id": "organization/tokenizer",
                                        "revision": "a" * 40,
                                    },
                                    "dataset": {
                                        "id": "organization/sharegpt",
                                        "filename": "sharegpt.json",
                                        "revision": "b" * 40,
                                        "adapter": "sharegpt",
                                    },
                                }
                            }
                        ],
                    },
                )
                return await asyncio.wait_for(request_coroutine, timeout=3)

    event_loop = asyncio.new_event_loop()
    try:
        response = event_loop.run_until_complete(request_validation())
    finally:
        event_loop.close()

    assert response.status_code == 200
    document = response.json()
    assert document["valid"] is True
    assert document["workload"] == {
        "immediate_runners": 1,
        "runner_plans": 0,
        "task_definitions": 0,
        "task_instances": 0,
        "task_nodes": 0,
    }
    assert {artifact["kind"] for artifact in document["artifacts"]} == {
        "dataset",
        "tokenizer",
    }
    dataset_artifact = next(
        artifact for artifact in document["artifacts"] if artifact["kind"] == "dataset"
    )
    assert dataset_artifact["adapter"] == "sharegpt"
    assert dataset_artifact["record_count"] == 1
    assert all("path" not in artifact for artifact in document["artifacts"])


def test_artifact_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets-cache"))
    config_path = tmp_path / "backend.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    tokenizer_directory.joinpath("tokenizer.json").write_text(
        '{"version":"1.0"}', encoding="utf-8"
    )
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        '[{"conversations": [{"from": "human", "value": "prompt"}]}]',
        encoding="utf-8",
    )
    application = create_app(
        ConfigStore(config_path),
        provider_registry=ProviderRegistry.from_environment(
            {
                "LLMPERF_PROVIDER_TEST_DISCOVERY": "static",
                "LLMPERF_PROVIDER_TEST_MODELS": "test-model",
            }
        ),
        tokenizer_cache=FakeTokenizerCache(tokenizer_directory),
        dataset_cache=FakeDatasetCache(dataset_path),
    )

    async def request_validation() -> httpx.Response:
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/api/v1/campaigns/validate-artifacts/stream",
                    json={
                        "campaign": {"name": "artifact-stream"},
                        "task_definitions": [
                            {
                                "name": "artifact-task",
                                "instances": {"trials": 1},
                                "payloads": {"replay": {"seed_namespace": "replay"}},
                                "workflow": [
                                    {
                                        "invoke": {
                                            "name": "prime",
                                            "payload": "replay",
                                        }
                                    }
                                ],
                                "runner": {
                                    "benchmark": {
                                        "provider": "test",
                                        "model": "test-model",
                                        "mean_input_tokens": 16,
                                        "stddev_input_tokens": 0,
                                        "mean_output_tokens": 8,
                                        "stddev_output_tokens": 0,
                                        "tokenizer": {
                                            "id": "organization/tokenizer",
                                            "revision": "a" * 40,
                                        },
                                        "dataset": {
                                            "id": "organization/sharegpt",
                                            "filename": "sharegpt.json",
                                            "revision": "b" * 40,
                                            "adapter": "sharegpt",
                                        },
                                    },
                                },
                            }
                        ],
                    },
                )

    event_loop = asyncio.new_event_loop()
    try:
        response = event_loop.run_until_complete(request_validation())
    finally:
        event_loop.close()

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    result_event = events[-1]
    assert result_event["event"] == "result", response.text
    assert result_event["result"]["valid"] is True
    assert result_event["result"]["workload"] == {
        "immediate_runners": 0,
        "runner_plans": 0,
        "task_definitions": 1,
        "task_instances": 1,
        "task_nodes": 1,
    }
    assert all("path" not in line for line in response.text.splitlines())

import asyncio
import json
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

from llmperf_backend.api.app import create_app
from llmperf_backend.configuration.loader import ConfigStore
from llmperf_backend.configuration.models import DatabaseConfig
from llmperf_backend.storage.repository import Base, Database
from llmperf_backend.providers.registry import ProviderRegistry
from llmperf_backend.providers.files import load_provider_file
from llmperf_backend.artifacts.resolver import DatasetResolution
from llmperf_backend.artifacts.resolver import TokenizerResolution

CONFIG = """
version: "1.0.0"
environment: test
server:
  host: 127.0.0.1
  port: 8000
database:
  url: "{database_url}"
  auto_create_schema: {auto_create_schema}
scheduler:
  enabled: false
planner:
  enabled: false
benchmark:
  provider: test
  model: test-model
"""


pytestmark = pytest.mark.postgresql


class FakeTokenizerCache:
    def __init__(self, path: Path, revision: str = "resolved-commit"):
        self.path = path
        self.revision = revision
        self.specs = []

    async def resolve(self, spec):
        self.specs.append(spec)
        return TokenizerResolution(
            source="huggingface",
            tokenizer_id=spec["id"],
            revision=self.revision,
            use_fast=spec.get("use_fast", True),
            path=self.path,
            cached=False,
        )


class FakeDatasetCache:
    def __init__(self, path: Path, revision: str = "resolved-dataset-commit"):
        self.path = path
        self.revision = revision
        self.specs = []

    async def resolve(self, spec):
        self.specs.append(spec)
        return DatasetResolution(
            source="huggingface",
            dataset_id=spec["id"],
            filename=spec["filename"],
            revision=self.revision,
            adapter=spec["adapter"],
            path=self.path,
            cached=False,
        )


class ASGITestClient:
    """Small synchronous harness compatible with HTTPX 0.27 and 0.28."""

    def __init__(self, app: Any):
        self.app = app
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.client: Optional[httpx.AsyncClient] = None
        self.lifespan: Optional[AbstractAsyncContextManager[None]] = None

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        lifespan = self.app.router.lifespan_context(self.app)
        self.lifespan = lifespan
        self.loop.run_until_complete(lifespan.__aenter__())
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://testserver",
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        loop = self._active_loop()
        client = self._active_client()
        lifespan = self.lifespan
        if lifespan is None:
            raise RuntimeError("ASGI lifespan was not started")
        loop.run_until_complete(client.aclose())
        loop.run_until_complete(lifespan.__aexit__(exc_type, exc, traceback))
        loop.close()

    def get(self, path, **kwargs):
        return self._active_loop().run_until_complete(
            self._active_client().get(path, **kwargs)
        )

    def post(self, path, **kwargs):
        return self._active_loop().run_until_complete(
            self._active_client().post(path, **kwargs)
        )

    def _active_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is None:
            raise RuntimeError("ASGI test client is not active")
        return self.loop

    def _active_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("ASGI test client is not active")
        return self.client


def make_client(
    tmp_path: Path,
    database_url: str,
    tokenizer_cache=None,
    dataset_cache=None,
    provider_path=None,
    auto_create_schema: bool = True,
    provider_registry=None,
) -> ASGITestClient:
    config_path = tmp_path / "backend.yaml"
    config_path.write_text(
        CONFIG.format(
            database_url=database_url,
            auto_create_schema=str(auto_create_schema).lower(),
        ),
        encoding="utf-8",
    )
    selected_provider_path = provider_path or tmp_path / "test-providers.yaml"
    if provider_path is None:
        selected_provider_path.write_text(
            'version: "1.0.0"\nproviders: {test: {base_url: "http://127.0.0.1:8001/v1"}}'
        )
    providers = ProviderRegistry.from_loader(
        lambda: load_provider_file(selected_provider_path, {})
    )
    if tokenizer_cache is None:
        tokenizer_directory = tmp_path / "tokenizer"
        tokenizer_directory.mkdir(exist_ok=True)
        resolved_tokenizer_cache = FakeTokenizerCache(tokenizer_directory)
    else:
        resolved_tokenizer_cache = tokenizer_cache
    if dataset_cache is None:
        dataset_path = tmp_path / "sharegpt.json"
        dataset_path.write_text("[]", encoding="utf-8")
        resolved_dataset_cache = FakeDatasetCache(dataset_path)
    else:
        resolved_dataset_cache = dataset_cache
    return ASGITestClient(
        create_app(
            ConfigStore(config_path),
            provider_registry=provider_registry or providers,
            tokenizer_cache=resolved_tokenizer_cache,
            dataset_cache=resolved_dataset_cache,
        )
    )


@pytest.fixture(autouse=True)
def clean_postgres(postgresql_url):
    async def reset():
        database = Database(
            DatabaseConfig(url=postgresql_url, auto_create_schema=False)
        )
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
        finally:
            await database.dispose()

    asyncio.run(reset())
    yield
    asyncio.run(reset())


@pytest.fixture
def client_factory(tmp_path: Path, postgresql_url):
    def factory(tokenizer_cache=None, dataset_cache=None, provider_path=None):
        return make_client(
            tmp_path,
            postgresql_url,
            tokenizer_cache=tokenizer_cache,
            dataset_cache=dataset_cache,
            provider_path=provider_path,
        )

    return factory


def test_health(client_factory):
    with client_factory() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["database"] == "connected"

        response = client.get("/api/v1/config")
        assert response.status_code == 200
        assert response.json()["config"]["benchmark"]["model"] == "test-model"


def test_existing_schema(tmp_path, postgresql_url):
    async def prepare():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            await database.create_schema()
        finally:
            await database.dispose()

    asyncio.run(prepare())
    with make_client(tmp_path, postgresql_url, auto_create_schema=False) as client:
        assert client.app.state.schema_state == "ready"
        assert client.get("/api/v1/runners").status_code == 200


def test_provider_startup(tmp_path, postgresql_url):
    provider_path = tmp_path / "providers.yaml"
    registry = ProviderRegistry.from_loader(
        lambda: load_provider_file(provider_path, {})
    )
    with make_client(tmp_path, postgresql_url, provider_registry=registry) as client:
        assert client.get("/health").status_code == 200
        listed = client.get("/api/v1/providers").json()
        assert listed["items"] == []
        assert listed["configuration_error"]
        assert client.post("/api/v1/providers/reload").status_code == 422
        provider_path.write_text(
            'version: "1.0.0"\nproviders: {test: {models: [test-model], api_key: ""}}'
        )
        assert client.post("/api/v1/providers/reload").status_code == 200
        assert (
            client.get("/api/v1/providers").json()["items"][0]["api_key_configured"]
            is False
        )
        assert client.get("/health").json()["status"] == "ok"


def test_provider_reload(client_factory, tmp_path):
    provider_path = tmp_path / "reload-providers.json"
    profile = {
        "base_url": "http://old.example/v1",
        "api_key": "old-secret",
        "discovery": {"mode": "static"},
        "models": ["glm-5.2"],
    }
    document = {"version": "1.0.0", "providers": {"test": profile}}
    provider_path.write_text(json.dumps(document))
    with client_factory(provider_path=provider_path) as client:
        listed = client.get("/api/v1/providers")
        assert listed.status_code == 200
        assert listed.json()["generation"] == 0
        assert listed.json()["items"][0]["base_url"] == "http://old.example/v1"

        profile["base_url"] = "http://new.example/v1"
        profile["api_key"] = "new-secret"
        provider_path.write_text(json.dumps(document))
        reloaded = client.post("/api/v1/providers/reload")

        assert reloaded.status_code == 200
        assert reloaded.json()["generation"] == 1
        assert reloaded.json()["changes"]["updated"] == ["test"]
        assert "new-secret" not in reloaded.text
        assert (
            client.app.state.scheduler.provider_registry.require("test").api_base
            == "http://new.example/v1"
        )

        invalid_runner = client.post(
            "/api/v1/runners",
            json={"worker": {}, "provider": "test", "model": "glm5.2"},
        )
        assert invalid_runner.status_code == 422
        assert "configured models: glm-5.2" in invalid_runner.text
        assert client.get("/api/v1/runners").json()["items"] == []

        profile["base_url"] = "invalid"
        provider_path.write_text(json.dumps(document))
        rejected = client.post("/api/v1/providers/reload")
        assert rejected.status_code == 422
        assert client.get("/api/v1/providers").json()["generation"] == 1


def test_config_validation(client_factory, postgresql_url):
    rendered_config = CONFIG.format(
        database_url=postgresql_url, auto_create_schema="true"
    )
    with client_factory() as client:
        response = client.post(
            "/api/v1/config/validate", json={"yaml_content": rendered_config}
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

        invalid = client.post(
            "/api/v1/config/validate", json={"yaml_content": "benchmark: []"}
        )
        assert invalid.status_code == 422


def test_campaign_log_storage(client_factory):
    with client_factory() as client:
        campaign = client.post("/api/v1/campaigns", json={"name": "logs"}).json()
        campaign_id = campaign["campaign_id"]
        runners = client.post(
            f"/api/v1/campaigns/{campaign_id}/runners",
            json={"runners": [{"label": "first"}, {"label": "second"}]},
        ).json()["items"]
        repository = client.app.state.runner_repository
        client._active_loop().run_until_complete(
            repository.set_logs(runners[0]["runner_id"], 0, "execution events\n", "")
        )
        records = client._active_loop().run_until_complete(
            repository.list_campaign_log_records(campaign_id, 1, 0)
        )
        assert records[0]["runner_id"] == runners[0]["runner_id"]
        assert records[0]["log_key"] is not None
        assert "benchmark" not in records[0]
        following = client._active_loop().run_until_complete(
            repository.list_campaign_log_records(campaign_id, 1, 1)
        )
        assert following[0]["runner_id"] == runners[1]["runner_id"]
        assert following[0]["log_key"] is None
        assert (
            client._active_loop().run_until_complete(
                repository.list_campaign_log_records("missing", 20, 0)
            )
            == []
        )


def test_runner_lifecycle(client_factory):
    with client_factory() as client:
        campaign = client.post(
            "/api/v1/campaigns",
            json={
                "name": "glm-study",
                "tags": {"purpose": "kv-cache"},
            },
        )
        assert campaign.status_code == 201
        campaign_id = campaign.json()["campaign_id"]

        batch = client.post(
            f"/api/v1/campaigns/{campaign_id}/runners",
            json={
                "runners": [
                    {
                        "label": "concurrency-1",
                        "metadata": {"experiment": "glm-kv-cache"},
                    }
                ]
            },
        )
        assert batch.status_code == 202
        created = batch.json()["items"][0]
        runner_id = created["runner_id"]
        assert created["status"] == "queued"

        listed = client.get("/api/v1/runners")
        assert listed.status_code == 200
        assert listed.json()["items"][0]["runner_id"] == runner_id
        assert listed.json()["full"] is False
        assert listed.json()["items"][0]["provider"] == "test"
        assert listed.json()["items"][0]["model"] == "test-model"
        assert "benchmark" not in listed.json()["items"][0]
        assert "stdout" not in listed.json()["items"][0]

        full_list = client.get("/api/v1/runners?full=true")
        assert full_list.status_code == 200
        assert full_list.json()["full"] is True
        assert full_list.json()["items"][0]["benchmark"]["model"] == "test-model"
        assert "stdout" in full_list.json()["items"][0]

        logs = client.get(f"/api/v1/runners/{runner_id}/logs")
        assert logs.status_code == 200
        assert logs.json() == {
            "runner_id": runner_id,
            "status": "queued",
            "scheduler_id": None,
            "worker": {"process_id": None, "exit_code": None},
            "stdout": None,
            "stderr": None,
        }

        cancelled = client.post(f"/api/v1/runners/{runner_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        events = client.get(f"/api/v1/runners/{runner_id}/events")
        assert [item["status"] for item in events.json()["items"]] == [
            "queued",
            "cancelled",
        ]

        campaign_status = client.get(f"/api/v1/campaigns/{campaign_id}")
        assert campaign_status.status_code == 200
        assert campaign_status.json()["status"] == "cancelled"
        assert campaign_status.json()["outcome"] == "cancelled"
        assert campaign_status.json()["runner_count"] == 1

        campaigns = client.get("/api/v1/campaigns")
        assert campaigns.json()["items"][0]["status"] == "cancelled"

        exported = client.get(f"/api/v1/campaigns/{campaign_id}/export")
        assert exported.status_code == 200
        assert exported.json()["version"] == "1.0.0"
        assert exported.json()["aggregate"]["status"] == "cancelled"
        assert exported.json()["aggregate"]["outcome"] == "cancelled"
        assert exported.json()["aggregate"]["runner_count"] == 1
        assert exported.json()["aggregate"]["status_counts"]["cancelled"] == 1
        assert "worker" in exported.json()["runners"][0]
        assert "stdout" in exported.json()["runners"][0]


def test_campaign_cancel(client_factory):
    with client_factory() as client:
        campaign_id = client.post(
            "/api/v1/campaigns", json={"name": "cancel-study"}
        ).json()["campaign_id"]
        started = client.post(
            f"/api/v1/campaigns/{campaign_id}/runners",
            json={"runners": [{"label": "one"}, {"label": "two"}]},
        )
        assert started.status_code == 202

        cancelled = client.post(f"/api/v1/campaigns/{campaign_id}/cancel")

        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert cancelled.json()["status_counts"]["cancelled"] == 2


@pytest.mark.parametrize("prefix_scope", ["round", "family"])
def test_campaign_rounds(client_factory, prefix_scope):
    with client_factory() as client:
        submission = {
            "campaign": {
                "name": "rounds",
                "rounds": {
                    "count": 3,
                    "seed": 20261014,
                    "prefix_seed_scope": prefix_scope,
                },
            },
            "runners": [{"label": "a"}, {"label": "b"}],
        }
        preview = client.post("/api/v1/campaigns/preview", json=submission)
        assert preview.status_code == 200
        assert preview.json()["summary"]["immediate_runners"] == 6
        started = client.post("/api/v1/campaigns/start", json=submission)
        assert started.status_code == 202
        runners = started.json()["items"]
        assert len(runners) == 6
        assert [runner["metadata"]["round"] for runner in runners] == [1, 1, 2, 2, 3, 3]
        assert len({runner["benchmark"]["dataset_seed"] for runner in runners}) == 6
        if prefix_scope == "family":
            assert (
                len({runner["benchmark"]["shared_prefix_seed"] for runner in runners})
                == 2
            )
            assert all(
                runner["metadata"]["runner_round"]["prefix_seed_scope"] == "family"
                for runner in runners
            )


def test_campaign_phases(client_factory):
    with client_factory() as client:
        submission = {
            "campaign": {
                "name": "phases",
                "rounds": {
                    "count": 2,
                    "seed": "date",
                    "prefix_seed_scope": "family",
                    "phases": ["load", "write", "read"],
                },
            },
            "runner_groups": [
                {
                    "name": name,
                    "phase": phase,
                    "matrix": {"member": [1, 2]},
                    "runner": {
                        "inputs": {
                            "default": {"tokens": 200, "shared_prefix_tokens": prefix}
                        }
                    },
                }
                for name, phase, prefix in [
                    ("target", "load", 160),
                    ("pressure", "write", 0),
                    ("target", "read", 160),
                ]
            ],
        }
        preview = client.post("/api/v1/campaigns/preview", json=submission)
        assert preview.status_code == 200, preview.text
        assert preview.json()["summary"]["immediate_runners"] == 12
        assert len(preview.json()["runner_groups"]) == 3
        assert isinstance(preview.json()["rounds"]["seed"], int)
        # Pin the preview seed explicitly when exact preview/start parity is wanted.
        submission["campaign"]["rounds"]["seed"] = preview.json()["rounds"]["seed"]
        started = client.post("/api/v1/campaigns/start", json=submission)
        assert started.status_code == 202, started.text
        runners = started.json()["items"]
        assert len(runners) == 12
        assert {runner["metadata"]["runner_round"]["seed"] for runner in runners} == {
            preview.json()["rounds"]["seed"]
        }
        assert len({runner["benchmark"]["dataset_seed"] for runner in runners}) == 12
        assert (
            len(
                {
                    runner["benchmark"]["shared_prefix_seed"]
                    for runner in runners
                    if runner["metadata"]["runner_group"]["name"] == "target"
                }
            )
            == 2
        )
        persisted = client.get(f"/api/v1/runners/{runners[0]['runner_id']}")
        assert persisted.status_code == 200
        assert (
            persisted.json()["metadata"]["runner_round"]
            == runners[0]["metadata"]["runner_round"]
        )


def test_atomic_campaign_start(client_factory):
    with client_factory() as client:
        empty = client.post(
            "/api/v1/campaigns/start",
            json={"campaign": {"name": "empty-study"}},
        )
        assert empty.status_code == 422

        invalid = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {"name": "invalid-study"},
                "runners": [
                    {
                        "worker": {"unknown_field": True},
                        "provider": "test",
                        "model": "test-model",
                    }
                ],
            },
        )
        assert invalid.status_code == 422
        assert client.get("/api/v1/campaigns").json()["items"] == []

        started = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {
                    "name": "atomic-study",
                    "tags": {"purpose": "kv-cache"},
                },
                "runners": [{"label": "cold"}, {"label": "warm"}],
            },
        )

        assert started.status_code == 202
        document = started.json()
        campaign_id = document["campaign"]["campaign_id"]
        assert len(document["items"]) == 2
        assert {item["campaign_id"] for item in document["items"]} == {campaign_id}
        status = client.get(f"/api/v1/campaigns/{campaign_id}").json()
        assert status["runner_count"] == 2
        assert status["status"] == "queued"

        planned = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {"name": "planned-study"},
                "runner_plans": [
                    {
                        "name": "interval-study",
                        "timezone": "Asia/Shanghai",
                        "starts_at": "2099-08-14T00:00:00+08:00",
                        "max_occurrences": 8,
                        "recurrence": {
                            "kind": "interval",
                            "every": 30,
                        },
                        "overlap_policy": "queue",
                        "runner": {"label": "periodic"},
                    }
                ],
            },
        )

        assert planned.status_code == 202
        planned_document = planned.json()
        assert planned_document["items"] == []
        assert len(planned_document["runner_plans"]) == 1
        planned_id = planned_document["campaign"]["campaign_id"]
        planned_status = client.get(f"/api/v1/campaigns/{planned_id}").json()
        assert planned_status["status"] == "planned"
        assert planned_status["runner_plan_count"] == 1


def test_preview_persistence_free(client_factory):
    payload = {
        "campaign": {"name": "preview-study"},
        "task_definitions": [
            {
                "name": "cache-window",
                "instances": {
                    "matrix": {"delay": [5, 10]},
                    "trials": 2,
                    "seed": 7,
                },
                "payloads": {"prompt": {"seed": "prompt"}},
                "workflow": [
                    {"invoke": {"name": "prime", "payload": "prompt"}},
                    {
                        "invoke": {
                            "name": "probe",
                            "payload": "prompt",
                            "after": "$delay",
                        }
                    },
                ],
                "runner": {},
            }
        ],
    }

    with client_factory() as client:
        preview = client.post(
            "/api/v1/campaigns/preview?limit=2&debug=true",
            json=payload,
        )
        campaigns = client.get("/api/v1/campaigns")

    assert preview.status_code == 200
    document = preview.json()
    assert document["valid"] is True
    assert document["summary"] == {
        "immediate_runners": 0,
        "runner_plans": 0,
        "task_definitions": 1,
        "task_instances": 4,
        "task_nodes": 8,
        "previewed_instances": 2,
    }
    definition = document["task_definitions"][0]
    assert definition["shown_instance_count"] == 2
    assert definition["truncated_instance_count"] == 2
    assert definition["instances"][0]["nodes"][1]["dependencies"] == ["prime"]
    assert definition["instances"][0]["nodes"][1]["after_seconds"] == 5
    assert "payload_seed" in definition["instances"][0]["nodes"][0]
    assert campaigns.json()["items"] == []


def test_compiled_dataset(tmp_path: Path, client_factory):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text("[]", encoding="utf-8")
    tokenizer_cache = FakeTokenizerCache(tokenizer_directory, revision="a" * 40)
    dataset_cache = FakeDatasetCache(dataset_path, revision="b" * 40)
    task = {
        "name": "dataset-replay",
        "payloads": {"replay": {"seed": "replay"}},
        "workflow": [
            {
                "invoke": {"name": "prime", "role": "prime", "payload": "replay"},
            }
        ],
        "runner": {
            "worker": {},
            "provider": "test",
            "model": "test-model",
            "stddev_output_tokens": 0,
            "tokenizer": {"id": "organization/model-tokenizer", "revision": "release"},
            "inputs": {
                "default": {
                    "dataset": {
                        "id": "organization/sharegpt",
                        "filename": "sharegpt.json",
                        "revision": "release",
                        "adapter": "sharegpt",
                    },
                    "dataset_prompt_mode": "concatenate",
                    "tokens": {"mean": 550, "stddev": 0},
                }
            },
        },
    }

    with client_factory(
        tokenizer_cache=tokenizer_cache, dataset_cache=dataset_cache
    ) as client:
        accepted = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {"name": "immutable-dataset"},
                "task_definitions": [task],
            },
        )
        assert accepted.status_code == 202, accepted.text
        campaign_id = accepted.json()["campaign"]["campaign_id"]
        campaign_status = client.get(f"/api/v1/campaigns/{campaign_id}")

    assert (
        accepted.json()["task_definitions"][0]["runner"]["benchmark"][
            "dataset_prompt_mode"
        ]
        == "concatenate"
    )
    assert accepted.json()["summary"] == {
        "immediate_runners": 0,
        "runner_plans": 0,
        "task_definitions": 1,
        "task_instances": 1,
        "task_nodes": 1,
    }
    assert campaign_status.status_code == 200
    assert campaign_status.json()["task_instance_count"] == 1
    assert campaign_status.json()["dispatch_count"] == 1

    mutable_cache = FakeDatasetCache(dataset_path, revision="main")
    with client_factory(
        tokenizer_cache=tokenizer_cache, dataset_cache=mutable_cache
    ) as client:
        rejected = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {"name": "mutable-dataset"},
                "task_definitions": [task],
            },
        )

    assert rejected.status_code == 422
    assert "immutable Hugging Face commit" in rejected.json()["detail"]


def test_scheduler_status(client_factory):
    with client_factory() as client:
        response = client.get("/api/v1/scheduler/status")

        assert response.status_code == 200
        assert response.json()["status"] == "disabled"
        assert response.json()["busy_slots"] == 0
        assert response.json()["live_slots"] == 0


def test_runner_tokenizer(tmp_path: Path, client_factory):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    cache = FakeTokenizerCache(tokenizer_directory)
    with client_factory(tokenizer_cache=cache) as client:
        created = client.post(
            "/api/v1/runners",
            json={
                "worker": {},
                "provider": "test",
                "model": "test-model",
                "tokenizer": {
                    "id": "organization/model-tokenizer",
                    "revision": "release",
                    "use_fast": False,
                },
            },
        )

        assert created.status_code == 202
        tokenizer = created.json()["benchmark"]["tokenizer"]
        assert tokenizer == {
            "source": "huggingface",
            "id": "organization/model-tokenizer",
            "revision": "resolved-commit",
            "use_fast": False,
            "selection": "explicit",
            "accuracy": "compatible",
            "requested_revision": "release",
            "immutable_revision": False,
        }
        assert cache.specs[0]["revision"] == "release"


def test_runner_dataset(tmp_path: Path, client_factory):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text("[]", encoding="utf-8")
    cache = FakeDatasetCache(dataset_path)
    with client_factory(dataset_cache=cache) as client:
        created = client.post(
            "/api/v1/runners",
            json={
                "worker": {},
                "provider": "test",
                "model": "test-model",
                "inputs": {
                    "default": {
                        "dataset": {
                            "id": "organization/sharegpt",
                            "filename": "sharegpt.json",
                            "revision": "release",
                            "adapter": "sharegpt",
                        },
                        "tokens": {"mean": 550, "stddev": 150},
                    }
                },
            },
        )

        assert created.status_code == 202
        assert created.json()["benchmark"]["dataset"] == {
            "source": "huggingface",
            "id": "organization/sharegpt",
            "filename": "sharegpt.json",
            "revision": "resolved-dataset-commit",
            "adapter": "sharegpt",
        }
        assert cache.specs[0]["revision"] == "release"


def test_runner_plan(client_factory):
    with client_factory() as client:
        campaign_id = client.post(
            "/api/v1/campaigns", json={"name": "planner-study"}
        ).json()["campaign_id"]
        created = client.post(
            f"/api/v1/campaigns/{campaign_id}/runner-plans",
            json={
                "name": "shanghai-hourly",
                "timezone": "Asia/Shanghai",
                "starts_at": "2026-08-13T00:00:00Z",
                "max_occurrences": 2,
                "recurrence": {"kind": "interval", "every": 3600},
                "overlap_policy": "skip",
                "runner": {"label": "planned"},
            },
        )

        assert created.status_code == 201
        plan = created.json()
        plan_id = plan["runner_plan_id"]
        assert plan["status"] == "active"
        assert plan["overlap_policy"] == "skip"
        assert plan["runner"]["benchmark"]["model"] == "test-model"
        listed = client.get("/api/v1/runner-plans")
        assert listed.json()["items"][0]["runner_plan_id"] == plan_id
        paused = client.post(f"/api/v1/runner-plans/{plan_id}/pause")
        assert paused.json()["status"] == "paused"
        resumed = client.post(f"/api/v1/runner-plans/{plan_id}/resume")
        assert resumed.json()["status"] == "active"
        cancelled = client.post(f"/api/v1/runner-plans/{plan_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"


def test_default_task_tokenizer(tmp_path, client_factory):
    tokenizer_directory = tmp_path / "general-tokenizer"
    tokenizer_directory.mkdir()
    tokenizer_cache = FakeTokenizerCache(tokenizer_directory, revision="a" * 40)
    task = {
        "name": "general-estimate",
        "workflow": [{"run": {"name": "request"}}],
        "runner": {
            "provider": "test",
            "model": "test-model",
            "stddev_output_tokens": 0,
            "inputs": {"default": {"tokens": 100}},
        },
    }
    with client_factory(tokenizer_cache=tokenizer_cache) as client:
        accepted = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {"name": "general-estimate"},
                "task_definitions": [task],
            },
        )
        assert accepted.status_code == 202, accepted.text
        resolved = accepted.json()["task_definitions"][0]["runner"]["benchmark"]
        assert resolved["input_token_count"] == "estimate"
        assert resolved["tokenizer"]["id"] == "hf-internal-testing/llama-tokenizer"
        assert resolved["tokenizer"]["selection"] == "global_default"
        assert resolved["tokenizer"]["accuracy"] == "approximate"

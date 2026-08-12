import asyncio
from pathlib import Path

import httpx

from llmperf_backend.app import create_app
from llmperf_backend.config import ConfigStore
from llmperf_backend.providers import ProviderRegistry
from llmperf_backend.tokenizers import TokenizerResolution
from llmperf_backend.datasets import DatasetResolution


CONFIG = """
version: 1
environment: test
server:
  host: 127.0.0.1
  port: 8000
database:
  url: "sqlite+aiosqlite:///{database_path}"
scheduler:
  enabled: false
benchmark:
  provider: test
  model: test-model
"""


class FakeTokenizerCache:
    def __init__(self, path: Path):
        self.path = path
        self.specs = []

    async def resolve(self, spec):
        self.specs.append(spec)
        return TokenizerResolution(
            source="huggingface",
            tokenizer_id=spec["id"],
            revision="resolved-commit",
            use_fast=spec.get("use_fast", True),
            path=self.path,
            cached=False,
        )


class FakeDatasetCache:
    def __init__(self, path: Path):
        self.path = path
        self.specs = []

    async def resolve(self, spec):
        self.specs.append(spec)
        return DatasetResolution(
            source="huggingface",
            dataset_id=spec["id"],
            filename=spec["filename"],
            revision="resolved-dataset-commit",
            format=spec["format"],
            path=self.path,
            cached=False,
        )


class ASGITestClient:
    """Small synchronous harness compatible with HTTPX 0.27 and 0.28."""

    def __init__(self, app):
        self.app = app
        self.loop = None
        self.client = None
        self.lifespan = None

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.lifespan = self.app.router.lifespan_context(self.app)
        self.loop.run_until_complete(self.lifespan.__aenter__())
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://testserver",
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.loop.run_until_complete(self.client.aclose())
        self.loop.run_until_complete(self.lifespan.__aexit__(exc_type, exc, traceback))
        self.loop.close()

    def get(self, path, **kwargs):
        return self.loop.run_until_complete(self.client.get(path, **kwargs))

    def post(self, path, **kwargs):
        return self.loop.run_until_complete(self.client.post(path, **kwargs))


def make_client(
    tmp_path: Path, tokenizer_cache=None, dataset_cache=None
) -> ASGITestClient:
    config_path = tmp_path / "backend.yaml"
    config_path.write_text(
        CONFIG.format(database_path=tmp_path / "backend.db"), encoding="utf-8"
    )
    providers = ProviderRegistry.from_environment(
        {"LLMPERF_PROVIDER_TEST_URL": "http://127.0.0.1:8001/v1"}
    )
    if tokenizer_cache is None:
        tokenizer_directory = tmp_path / "tokenizer"
        tokenizer_directory.mkdir(exist_ok=True)
        tokenizer_cache = FakeTokenizerCache(tokenizer_directory)
    if dataset_cache is None:
        dataset_path = tmp_path / "sharegpt.json"
        dataset_path.write_text("[]", encoding="utf-8")
        dataset_cache = FakeDatasetCache(dataset_path)
    return ASGITestClient(
        create_app(
            ConfigStore(config_path),
            provider_registry=providers,
            tokenizer_cache=tokenizer_cache,
            dataset_cache=dataset_cache,
        )
    )


def test_health(tmp_path: Path):
    with make_client(tmp_path) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["database"] == "connected"

        response = client.get("/api/v1/config")
        assert response.status_code == 200
        assert response.json()["config"]["benchmark"]["model"] == "test-model"


def test_config_validation(tmp_path: Path):
    rendered_config = CONFIG.format(database_path=tmp_path / "validated.db")
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/config/validate", json={"yaml_content": rendered_config}
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

        invalid = client.post(
            "/api/v1/config/validate", json={"yaml_content": "benchmark: []"}
        )
        assert invalid.status_code == 422


def test_runner_lifecycle(tmp_path: Path):
    with make_client(tmp_path) as client:
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
        assert campaign_status.json()["runner_count"] == 1

        campaigns = client.get("/api/v1/campaigns")
        assert campaigns.json()["items"][0]["status"] == "cancelled"

        exported = client.get(f"/api/v1/campaigns/{campaign_id}/export")
        assert exported.status_code == 200
        assert exported.json()["aggregate"]["runner_count"] == 1
        assert exported.json()["aggregate"]["status_counts"]["cancelled"] == 1
        assert "worker" in exported.json()["runners"][0]
        assert "stdout" in exported.json()["runners"][0]


def test_campaign_cancel(tmp_path: Path):
    with make_client(tmp_path) as client:
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


def test_atomic_campaign_start(tmp_path: Path):
    with make_client(tmp_path) as client:
        invalid = client.post(
            "/api/v1/campaigns/start",
            json={
                "campaign": {"name": "invalid-study"},
                "runners": [
                    {
                        "benchmark": {
                            "provider": "test",
                            "model": "test-model",
                            "unknown_field": True,
                        }
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


def test_scheduler_status(tmp_path: Path):
    with make_client(tmp_path) as client:
        response = client.get("/api/v1/scheduler/status")

        assert response.status_code == 200
        assert response.json()["status"] == "disabled"
        assert response.json()["active_slots"] == 0


def test_runner_tokenizer(tmp_path: Path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    cache = FakeTokenizerCache(tokenizer_directory)
    with make_client(tmp_path, tokenizer_cache=cache) as client:
        created = client.post(
            "/api/v1/runners",
            json={
                "benchmark": {
                    "provider": "test",
                    "model": "test-model",
                    "tokenizer": {
                        "id": "organization/model-tokenizer",
                        "revision": "release",
                        "use_fast": False,
                    },
                }
            },
        )

        assert created.status_code == 202
        assert created.json()["benchmark"]["tokenizer"] == {
            "source": "huggingface",
            "id": "organization/model-tokenizer",
            "revision": "resolved-commit",
            "use_fast": False,
        }
        assert cache.specs[0]["revision"] == "release"


def test_runner_dataset(tmp_path: Path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text("[]", encoding="utf-8")
    cache = FakeDatasetCache(dataset_path)
    with make_client(tmp_path, dataset_cache=cache) as client:
        created = client.post(
            "/api/v1/runners",
            json={
                "benchmark": {
                    "provider": "test",
                    "model": "test-model",
                    "dataset": {
                        "id": "organization/sharegpt",
                        "filename": "sharegpt.json",
                        "revision": "release",
                        "format": "sharegpt",
                    },
                }
            },
        )

        assert created.status_code == 202
        assert created.json()["benchmark"]["dataset"] == {
            "source": "huggingface",
            "id": "organization/sharegpt",
            "filename": "sharegpt.json",
            "revision": "release",
            "format": "sharegpt",
        }
        assert cache.specs == []

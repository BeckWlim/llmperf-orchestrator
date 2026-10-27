"""Campaign log HTTP contracts without a database or Provider."""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx

from llmperf_backend.api.app import create_app
from llmperf_backend.configuration.loader import ConfigStore


def test_campaign_log_api(tmp_path):
    configuration = tmp_path / "backend.yaml"
    configuration.write_text(
        'version: "1.0.0"\nenvironment: test\n'
        "database:\n  url: postgresql+asyncpg:///unused_test\n"
        "scheduler:\n  enabled: false\nplanner:\n  enabled: false\n"
        "benchmark:\n  provider: test\n  model: test-model\n",
        encoding="utf-8",
    )
    application = create_app(ConfigStore(configuration))
    repository = Mock()
    repository.get_campaign = AsyncMock(return_value={"campaign_id": "campaign"})
    repository.list_campaign_log_records = AsyncMock(
        return_value=[
            {"runner_id": "first", "status": "succeeded", "log_key": "private-key"},
            {"runner_id": "second", "status": "queued", "log_key": None},
        ]
    )
    repository.read_logs = AsyncMock(
        return_value={"stdout": "worker.completed\n", "stderr": None}
    )
    application.state.runner_repository = repository
    application.state.schema_state = "ready"

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/campaigns/campaign/logs?limit=1&offset=3"
            )
            assert response.status_code == 200
            assert response.json() == {
                "campaign_id": "campaign",
                "items": [
                    {
                        "runner_id": "first",
                        "status": "succeeded",
                        "scheduler_id": None,
                        "worker": None,
                        "stdout": "worker.completed\n",
                        "stderr": None,
                    }
                ],
                "limit": 1,
                "offset": 3,
                "next_offset": 4,
            }
            repository.list_campaign_log_records.assert_awaited_once_with(
                "campaign", 2, 3
            )
            repository.read_logs.assert_awaited_once_with("private-key")
            for query in ("limit=0", "limit=201", "offset=-1"):
                assert (
                    await client.get(f"/api/v1/campaigns/campaign/logs?{query}")
                ).status_code == 422
            repository.read_logs.side_effect = ValueError("digest mismatch")
            assert (
                await client.get("/api/v1/campaigns/campaign/logs")
            ).status_code == 503
            repository.list_campaign_log_records.return_value = []
            empty = await client.get("/api/v1/campaigns/campaign/logs")
            assert empty.json()["items"] == []
            assert empty.json()["next_offset"] is None
            repository.get_campaign.return_value = None
            assert (
                await client.get("/api/v1/campaigns/missing/logs")
            ).status_code == 404

    asyncio.run(run())


def test_worker_log_api(tmp_path):
    configuration = tmp_path / "backend.yaml"
    configuration.write_text(
        'version: "1.0.0"\nenvironment: test\n'
        "database:\n  url: postgresql+asyncpg:///unused_test\n"
        "scheduler:\n  enabled: false\nplanner:\n  enabled: false\n"
        "benchmark:\n  provider: test\n  model: test-model\n",
        encoding="utf-8",
    )
    application = create_app(ConfigStore(configuration))
    repository = Mock()
    repository.get_runner = AsyncMock(
        return_value={"runner_id": "r", "status": "running"}
    )
    repository.list_workers = AsyncMock(
        return_value={
            "items": [{"worker_id": "1", "status": "succeeded"}],
            "limit": 1,
            "offset": 0,
            "next_offset": 1,
        }
    )
    repository.runner_exists = AsyncMock(return_value=True)
    repository.get_worker_log = AsyncMock(
        return_value={
            "worker_id": "1",
            "step_id": "1",
            "status": "succeeded",
            "completion_available": True,
            "input_messages": [{"role": "user", "content": "hello"}],
            "output_text": "answer",
            "metrics": {},
        }
    )
    repository.read_logs = AsyncMock(
        return_value={
            "stdout": 'time worker.submitted worker="1"\n'
            'time worker.completed worker="1" state="succeeded"\n'
            'time worker.completed worker="10" state="failed"\n',
            "stderr": "unattributed runner error",
        }
    )
    application.state.runner_repository = repository
    application.state.schema_state = "ready"

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            page = await client.get("/api/v1/runners/r/workers?limit=1&offset=0")
            assert page.status_code == 200
            assert page.json()["next_offset"] == 1
            repository.list_workers.assert_awaited_once_with("r", 1, 0)
            for query in ("limit=0", "limit=201", "offset=-1"):
                assert (
                    await client.get(f"/api/v1/runners/r/workers?{query}")
                ).status_code == 422
            response = await client.get("/api/v1/runners/r/logs?worker_id=1")
            assert response.status_code == 200
            assert response.json()["worker_log"]["output_text"] == "answer"
            assert 'worker="10"' not in response.json()["stdout"]
            assert 'worker.submitted worker="1"' in response.json()["stdout"]
            assert response.json()["stderr"] is None
            repository.get_worker_log.assert_awaited_once_with("r", "1")
            assert (
                await client.get("/api/v1/runners/r/logs?worker_id=")
            ).status_code == 422
            repository.get_worker_log.return_value = None
            assert (
                await client.get("/api/v1/runners/r/logs?worker_id=missing")
            ).status_code == 404
            repository.get_worker_log.side_effect = ValueError("private payload")
            invalid = await client.get("/api/v1/runners/r/logs?worker_id=1")
            assert invalid.status_code == 503
            assert "private payload" not in invalid.text
            repository.get_runner.return_value = None
            repository.runner_exists.return_value = False
            assert (
                await client.get("/api/v1/runners/missing/workers")
            ).status_code == 404
            assert (
                await client.get("/api/v1/runners/missing/logs?worker_id=1")
            ).status_code == 404

    asyncio.run(run())

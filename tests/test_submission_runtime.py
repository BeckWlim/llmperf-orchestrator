"""Submission identity, inherited quotas and compact durable metadata."""

import asyncio
from uuid import uuid4

import pytest

from llmperf_backend.configuration.models import DatabaseConfig, SubmissionRuntimeConfig
from llmperf_backend.storage.logs import RunnerLogStore
from llmperf_backend.storage.repository import (
    Base,
    Database,
    RunnerRepository,
    BenchmarkRunnerRecord,
)


def test_log_files(tmp_path):
    store = RunnerLogStore(str(tmp_path), byte_limit=12)
    runner_id = str(uuid4())
    key = store.write(runner_id, "long output: abcdefghijkl", "failure")
    assert key is not None
    assert store.read(key) == {"stdout": "abcdefghijkl", "stderr": "failure"}
    assert store.write(runner_id, "abcdefghijkl", "failure") == key
    assert store.write(runner_id, "", "") is None
    assert key.startswith(f"{runner_id}/runner/")
    with pytest.raises(ValueError):
        store.read("../../secret")
    (tmp_path / key).write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        store.read(key)
    (tmp_path / key).unlink()
    assert store.read(key) == {"stdout": None, "stderr": None}


def test_worker_log_files(tmp_path):
    from llmperf.clients.models import RequestConfig
    from llmperf_backend.runtime.messages import WorkerCompletion

    store = RunnerLogStore(str(tmp_path), byte_limit=8)
    runner_id = str(uuid4())
    worker_id = str(uuid4())
    completion = WorkerCompletion(
        step_id="one",
        metrics={
            "error_code": "provider_error",
            "error_msg": "sensitive body",
            "ttft_s": float("nan"),
        },
        generated_text="完整输出" * 100,
        request=RequestConfig(
            model="test",
            prompt=("完整输入" * 100, 400),
            sampling_params={"Authorization": "secret-token", "temperature": 0.5},
        ),
    )
    stored_log = store.write_worker(runner_id, worker_id, completion)
    assert stored_log.key.startswith(f"{runner_id}/workers/{worker_id}/")
    assert store.write_worker(runner_id, worker_id, completion) == stored_log
    restored = store.read_worker(runner_id, worker_id, stored_log.key)
    assert restored.generated_text == completion.generated_text
    assert restored.request.prompt == completion.request.prompt
    assert restored.metrics == {"error_code": "provider_error", "ttft_s": None}
    assert restored.request.sampling_params == {"temperature": 0.5}
    assert restored.model_dump(mode="json") == stored_log.payload
    log_file = tmp_path / stored_log.key
    assert log_file.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="key"):
        store.read_worker(runner_id, str(uuid4()), stored_log.key)
    with pytest.raises(ValueError, match="key"):
        store.read_worker(runner_id, worker_id, "../outside.json")
    log_file.write_text("corrupt")
    with pytest.raises(ValueError, match="digest"):
        store.read_worker(runner_id, worker_id, stored_log.key)
    log_file.unlink()
    with pytest.raises(FileNotFoundError):
        store.read_worker(runner_id, worker_id, stored_log.key)


@pytest.mark.postgresql
def test_runtime_storage(postgresql_url, tmp_path):
    async def run():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database, log_directory=str(tmp_path))
            campaign = await repository.create_campaign(
                "bounded",
                None,
                {},
                "test",
                runtime=SubmissionRuntimeConfig(concurrent_workers=2),
            )
            batch = await repository.create_runners(
                campaign["campaign_id"],
                [
                    {"benchmark": {"repeat": 3}, "metadata": {}},
                    {"benchmark": {"repeat": 5}, "metadata": {}},
                ],
                "test",
                runtime=SubmissionRuntimeConfig(concurrent_workers=1),
            )
            assert batch is not None
            assert (
                batch[0]["runtime_id"]
                == batch[1]["runtime_id"]
                != campaign["runtime_id"]
            )
            standalone = await repository.create_runner({"repeat": 1}, {}, "test")
            assert standalone is not None
            standalone_runtime = await repository.get_submission_runtime(
                standalone["runtime_id"]
            )
            assert standalone_runtime is not None
            assert standalone_runtime["limits"] == {}
            # Recreate the repository to prove policy is durable, not held by the API.
            reloaded = RunnerRepository(database, log_directory=str(tmp_path))
            claimed = await reloaded.start_selected(batch[0]["runner_id"], "test")
            assert claimed is not None
            assert list(claimed["runtime_limits"].items()) == [
                (batch[0]["runtime_id"], 1),
                (campaign["runtime_id"], 2),
            ]
            await reloaded.set_logs(batch[0]["runner_id"], 0, "stored in file", "")
            async with database.sessions() as session:
                record = await session.get(BenchmarkRunnerRecord, batch[0]["runner_id"])
                assert record is not None and record.log_key
                assert (tmp_path / record.log_key).exists()
                assert await reloaded.read_logs(record.log_key) == {
                    "stdout": "stored in file",
                    "stderr": "",
                }
            aggregate = await reloaded.campaign_worker_progress(
                [campaign["campaign_id"]]
            )
            assert len(aggregate) == 1
            assert aggregate[campaign["campaign_id"]]["planned"] == 8
            assert aggregate[campaign["campaign_id"]]["submitted"] == 0
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(run())

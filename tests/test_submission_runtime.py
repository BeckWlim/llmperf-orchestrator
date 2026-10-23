"""Submission identity, inherited quotas and compact durable metadata."""

import asyncio

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
    key = store.write("long output: abcdefghijkl", "failure")
    assert key is not None
    assert store.read(key) == {"stdout": "abcdefghijkl", "stderr": "failure"}
    assert store.write("abcdefghijkl", "failure") == key
    assert store.write("", "") is None
    with pytest.raises(ValueError):
        store.read("../../secret")
    (tmp_path / key).write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        store.read(key)
    (tmp_path / key).unlink()
    assert store.read(key) == {"stdout": None, "stderr": None}


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

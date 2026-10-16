"""Fresh-schema, durable history, ownership and restart integration checks."""

import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text, update

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.models import RequestConfig
from llmperf_backend.runtime.runner import WorkerResult
from llmperf_backend.storage.messages import MessageRecord
from llmperf_backend.configuration.models import DatabaseConfig
from llmperf_backend.storage.repository import (
    Base,
    Database,
    RunnerRepository,
    ExecutionLeaseRecord,
    TaskAttemptRecord,
)
from llmperf_backend.runtime.coordinator import RuntimeCoordinator

pytestmark = pytest.mark.postgresql


def test_worker_mailbox(postgresql_url):
    from llmperf.benchmark.execution import RequestStep
    from llmperf.clients.models import RequestConfig
    from llmperf_backend.runtime.runner import WorkerResult
    from llmperf_backend.storage.messages import MessageRecord

    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("worker-test")
            runner = await repository.create_runner(
                {
                    "provider": "test",
                    "model": "test",
                    "repeat": 2,
                    "concurrent_requests": None,
                },
                {},
                "test",
            )
            assert runner is not None
            runner_id = runner["runner_id"]
            assert await repository.start_selected(runner_id, "test") is not None
            request = RequestConfig(model="test", prompt=("hello", 1))
            first = RequestStep("first", request)
            assert await repository.reserve_worker(runner_id, first, None)
            assert not await repository.reserve_worker(runner_id, first, None)
            result = WorkerResult(
                "first", ({}, "answer", request), datetime.now(timezone.utc)
            )
            assert await repository.commit_worker(runner_id, result)
            assert await repository.commit_worker(runner_id, result)
            messages = await repository.pending_messages(
                runner_id, "runner", "test", 10
            )
            assert len(messages) == 1
            assert messages[0].payload["generated_text"] == "answer"
            await repository.acknowledge_message(runner_id, messages[0].id, "test")
            assert not await repository.pending_messages(
                runner_id, "runner", "test", 10
            )
            assert await repository.reserve_worker(
                runner_id, RequestStep("second", request), None
            )
            with pytest.raises(ValueError, match="declared request expansion"):
                await repository.reserve_worker(
                    runner_id, RequestStep("third", request), None
                )
            async with database.sessions() as session:
                stored = (await session.execute(select(MessageRecord))).scalars().all()
                assert len(stored) == 1
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


def test_round_recovery(postgresql_url):
    from llmperf_backend.compiler.rounds import expand_rounds

    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("round-test")
            templates = [{"benchmark": {"provider": "test", "model": "test"}}] * 2
            workload = await repository.create_campaign_workload(
                "rounds",
                None,
                {},
                expand_rounds(templates, 3, 20261014),
                [],
                [],
                "test",
            )
            coordinator = RuntimeCoordinator(repository)
            await coordinator.reconcile()
            first = await coordinator.next_runner("slot-1")
            second = await coordinator.next_runner("slot-2")
            assert first["metadata"]["round"] == second["metadata"]["round"] == 1
            assert not coordinator.order
            await repository.finish_runner(
                first["runner_id"], "failed", "test", 1, "", ""
            )
            await coordinator.reconcile()
            assert not coordinator.order
            # A fresh coordinator must retain the barrier while one member runs.
            recovered = RuntimeCoordinator(repository)
            await recovered.reconcile()
            assert not recovered.order
            await repository.finish_runner(
                second["runner_id"], "failed", "test", 1, "", ""
            )
            await recovered.reconcile()
            next_round = await recovered.next_runner("slot-1")
            assert next_round["metadata"]["round"] == 2
            await repository.request_cancel_campaign(
                workload["campaign"]["campaign_id"]
            )
            final_snapshot = await repository.runtime_snapshot()
            assert not final_snapshot["runners"]
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


def test_sql_bootstrap(postgresql_url):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            async with database.engine.connect() as connection:
                raw_connection = await connection.get_raw_connection()
                driver = raw_connection.driver_connection
                if driver is None:
                    raise RuntimeError("PostgreSQL connection unavailable")
                await driver.execute(
                    (Path(__file__).parents[1] / "sql/postgresql/init.sql").read_text()
                )
            assert await database.schema_status() == "ready"
            async with database.engine.connect() as connection:
                assert (
                    await connection.scalar(
                        text("SELECT to_regclass('schema_metadata')")
                    )
                    is None
                )
            campaign = await RunnerRepository(database).create_campaign(
                "sql-bootstrap", None, {}, "test"
            )
            assert campaign["name"] == "sql-bootstrap"
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ["table", "column"])
def test_schema_structure(postgresql_url, missing):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            campaign = await RunnerRepository(database).create_campaign(
                "preserve-data", None, {}, "test"
            )
            await database.create_schema()
            assert await database.schema_status() == "ready"
            async with database.engine.begin() as connection:
                assert (
                    await connection.scalar(
                        text("SELECT to_regclass('schema_metadata')")
                    )
                    is None
                )
                stored_name = await connection.scalar(
                    text("SELECT name FROM benchmark_campaigns WHERE id = :id"),
                    {"id": campaign["campaign_id"]},
                )
                assert stored_name == "preserve-data"
                if missing == "table":
                    await connection.execute(text("DROP TABLE conversation_messages"))
                else:
                    await connection.execute(
                        text("ALTER TABLE conversation_messages DROP COLUMN content")
                    )
            assert await database.schema_status() == "schema_mismatch"
            with pytest.raises(RuntimeError, match="tables or columns are missing"):
                await database.create_schema()
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


def test_history_recovery(postgresql_url):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            assert await database.schema_status() == "ready"
            repository = RunnerRepository(database)
            await repository.acquire_runtime("first")
            competitor = RunnerRepository(database)
            with pytest.raises(RuntimeError, match="Another Backend"):
                await competitor.acquire_runtime("second")
            benchmark = {
                "provider": "test",
                "model": "test",
                "adapter": "openai",
                "tokenizer": {
                    "id": "test",
                    "revision": "a" * 40,
                    "requested_revision": "main",
                    "immutable_revision": True,
                    "selection": "explicit",
                    "accuracy": "compatible",
                },
                "timeout": 10,
                "repeat": 1,
                "concurrent_requests": 1,
                "mean_input_tokens": 2,
                "stddev_input_tokens": 0,
                "mean_output_tokens": 2,
                "stddev_output_tokens": 0,
                "additional_sampling_params": {},
            }
            workload = await repository.create_campaign_workload(
                "chat",
                None,
                {},
                [],
                [],
                [
                    {
                        "definition": {
                            "name": "chat",
                            "payloads": {"p": {"seed": "p", "text": "hello"}},
                            "workflow": [
                                {
                                    "invoke": {
                                        "name": "first",
                                        "payload": "p",
                                    }
                                },
                                {
                                    "invoke": {
                                        "name": "second",
                                        "payload": "p",
                                        "depends_on": [
                                            {"step": "first", "reuse_messages": True}
                                        ],
                                    }
                                },
                            ],
                        },
                        "runner_template": {
                            "label": "chat",
                            "metadata": {},
                            "benchmark": benchmark,
                        },
                    }
                ],
                "test",
            )
            coordinator = RuntimeCoordinator(repository)
            await coordinator.reconcile()
            first = await coordinator.next_runner("first")
            assert "input_messages" not in first["benchmark"]
            assert (
                not coordinator.order
            )  # A dependent turn is not an actor reservation.
            request = RequestConfig(model="test", prompt=("hello", 1))
            assert await repository.reserve_worker(
                first["runner_id"], RequestStep("0", request), None
            )
            completion = WorkerResult(
                "0", ({}, "world", request), datetime.now(timezone.utc)
            )
            assert await repository.commit_worker(first["runner_id"], completion)
            assert await repository.commit_worker(first["runner_id"], completion)
            result = [{"error_code": None, "request_metadata": {"prompt_hash": "hash"}}]
            await coordinator.reconcile()
            assert not coordinator.order
            assert await repository.complete_runner(
                first["runner_id"], {}, result, 0, "", ""
            )
            assert not await repository.complete_runner(
                first["runner_id"], {}, result, 0, "", ""
            )
            await coordinator.reconcile()
            second = await coordinator.next_runner("first")
            assert second["benchmark"]["input_messages"] == [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
            # Ownership expires after the second send becomes possible. Recovery must
            # preserve the first committed turn and never replay the second attempt.
            async with database.sessions() as session, session.begin():
                await session.execute(
                    update(ExecutionLeaseRecord).values(
                        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
                    )
                )
            await competitor.acquire_runtime("second")
            with pytest.raises(RuntimeError, match="fenced"):
                await repository.complete_runner(
                    second["runner_id"], {}, result, 0, "", ""
                )
            assert await competitor.requeue_stale(300) == 1
            async with database.sessions() as session:
                attempt = (
                    await session.execute(
                        select(TaskAttemptRecord).where(
                            TaskAttemptRecord.runner_id == second["runner_id"]
                        )
                    )
                ).scalar_one()
                assert attempt.state == "indeterminate"
            recovered = RuntimeCoordinator(competitor)
            await recovered.reconcile()
            assert not recovered.order
            exported = await competitor.export_campaign(
                workload["campaign"]["campaign_id"], True
            )
            assert exported is not None
            async with database.sessions() as session:
                messages = (
                    (
                        await session.execute(
                            select(MessageRecord).where(
                                MessageRecord.topic == "worker.completed"
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(messages) == 1
                assert messages[0].payload["generated_text"] == "world"
            await competitor.release_runtime()
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


def test_schema_rejection(postgresql_url):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.execute(text("CREATE TABLE legacy_test (id integer)"))
            assert await database.ping()
            assert await database.schema_status() == "schema_mismatch"
            with pytest.raises(RuntimeError, match="schema_mismatch"):
                await database.create_schema()
        finally:
            async with database.engine.begin() as connection:
                await connection.execute(text("DROP TABLE IF EXISTS legacy_test"))
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())

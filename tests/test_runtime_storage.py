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
from llmperf_backend.runtime.messages import RunnerMessage

pytestmark = pytest.mark.postgresql


def test_shared_lease_fencing(postgresql_url):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        repository = RunnerRepository(database)
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            await repository.acquire_runtime("first-owner")
            async with database.sessions() as first_session, first_session.begin():
                await repository._check_runtime(first_session)
                async with (
                    database.sessions() as second_session,
                    second_session.begin(),
                ):
                    # Ordinary transactions must share ownership without queueing
                    # behind one another's global lease lock.
                    await asyncio.wait_for(repository._check_runtime(second_session), 1)
                renewal = asyncio.create_task(repository.renew_runtime())
                await asyncio.sleep(0.02)
                assert not renewal.done(), "renewal must wait for fenced transactions"
            await asyncio.wait_for(renewal, 1)
            async with database.sessions() as session, session.begin():
                await session.execute(
                    update(ExecutionLeaseRecord).values(
                        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
                    )
                )
            successor = RunnerRepository(database)
            await successor.acquire_runtime("second-owner")
            with pytest.raises(RuntimeError, match="fenced"):
                await repository.heartbeat_runners(["missing"])
            with pytest.raises(RuntimeError, match="fenced"):
                await repository.renew_runtime()
            await repository.release_runtime()
            # Releasing the old owner cannot expire its successor's lease.
            await successor.renew_runtime()
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


def test_batched_durable_messages(postgresql_url):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        repository = RunnerRepository(database)
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            await repository.acquire_runtime("batch-test")
            runner_ids = []
            all_messages = []
            for index in range(2):
                runner = await repository.create_runner(
                    {"provider": "test", "model": "test", "repeat": 1}, {}, "test"
                )
                assert runner is not None
                runner_id = runner["runner_id"]
                runner_ids.append(runner_id)
                await repository.start_selected(runner_id, "batch-test")
                for sequence in range(3):
                    message = RunnerMessage(runner_id, "output", {"sequence": sequence})
                    await repository.publish_message(message)
                    all_messages.append(message)
            recovered = await repository.pending_messages_batch(
                runner_ids, "runner", "consumer", 1
            )
            assert len(recovered) == 2
            assert {message.runner_id for message in recovered} == set(runner_ids)
            first_ids = [
                message.id
                for message in all_messages
                if message.runner_id == runner_ids[0]
            ]
            await repository.acknowledge_messages(runner_ids[0], first_ids, "consumer")
            await repository.acknowledge_messages(runner_ids[0], first_ids, "consumer")
            with pytest.raises(ValueError, match="does not belong"):
                await repository.acknowledge_messages(
                    runner_ids[1], first_ids, "consumer"
                )
            remaining = await repository.pending_messages_batch(
                runner_ids, "runner", "consumer", 10
            )
            assert len(remaining) == 3
            assert all(message.runner_id == runner_ids[1] for message in remaining)
            assert (
                len(
                    await repository.pending_messages_batch(
                        runner_ids, "runner", "other", 10
                    )
                )
                == 6
            )
            assert await repository.heartbeat_runners(runner_ids) == set()
            async with database.sessions() as held_session, held_session.begin():
                await held_session.execute(
                    text("SELECT id FROM runners WHERE id = :id FOR UPDATE"),
                    {"id": runner_ids[0]},
                )
                assert (
                    await asyncio.wait_for(repository.heartbeat_runners(runner_ids), 1)
                    == set()
                )
            await repository.request_cancel(runner_ids[0])
            assert await repository.heartbeat_runners(runner_ids, refresh=False) == {
                runner_ids[0]
            }
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


def test_dedicated_lease_renewal(postgresql_url):
    """Ordinary writes do not renew ownership; maintenance cannot resurrect it."""

    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("activity")
            runner = await repository.create_runner(
                {"provider": "test", "model": "test", "repeat": 1}, {}, "test"
            )
            assert runner is not None
            near_expiry = datetime.now(timezone.utc) + timedelta(seconds=2)
            async with database.sessions() as session, session.begin():
                await session.execute(
                    update(ExecutionLeaseRecord).values(expires_at=near_expiry)
                )
            assert (
                await repository.start_selected(runner["runner_id"], "activity")
                is not None
            )
            async with database.sessions() as session:
                unchanged = await session.get(ExecutionLeaseRecord, 1)
                assert unchanged is not None
                assert unchanged.expires_at == near_expiry
            await repository.renew_runtime()
            async with database.sessions() as session:
                renewed = await session.get(ExecutionLeaseRecord, 1)
                assert renewed is not None
                assert renewed.expires_at > near_expiry + timedelta(seconds=20)
            async with database.sessions() as session, session.begin():
                await session.execute(
                    update(ExecutionLeaseRecord).values(
                        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
                    )
                )
            with pytest.raises(RuntimeError, match="expired"):
                await repository.heartbeat(runner["runner_id"])
            with pytest.raises(RuntimeError, match="expired"):
                await repository.renew_runtime()
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())


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
                    "concurrent_workers": None,
                },
                {},
                "test",
            )
            assert runner is not None
            runner_id = runner["runner_id"]
            initial_progress = await repository.worker_progress(runner_ids=[runner_id])
            assert initial_progress[runner_id]["planned"] == 2
            assert initial_progress[runner_id]["created"] == 0
            assert initial_progress[runner_id]["submitted"] == 0
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
            current_progress = await repository.worker_progress(runner_ids=[runner_id])
            assert current_progress[runner_id]["submitted"] == 2
            assert current_progress[runner_id]["pending_results"] == 1
            assert current_progress[runner_id]["succeeded"] == 1
            assert current_progress[runner_id]["created"] is None
            assert current_progress[runner_id]["not_submitted"] == 0
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


@pytest.mark.parametrize("phases", [("default",), ("load", "pressure", "read")])
def test_round_recovery(postgresql_url, phases):
    from llmperf_backend.compiler.rounds import expand_rounds, expand_runner_group

    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("round-test")
            templates = [
                runner
                for phase in phases
                for runner in expand_runner_group(
                    {"benchmark": {"provider": "test", "model": "test"}},
                    "target",
                    {"member": [1, 2]},
                    phase,
                )
            ]
            workload = await repository.create_campaign_workload(
                "rounds",
                None,
                {},
                expand_rounds(templates, 3, 20261014, "family", phases),
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
                    text("SELECT name FROM campaigns WHERE id = :id"),
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
                "concurrent_workers": 1,
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


def test_large_graph_snapshot(postgresql_url, monkeypatch):
    """Large graph creation yields to runtime work; snapshots exclude heavy payloads."""
    import threading
    from sqlalchemy import event
    from llmperf_backend.compiler.tasks import TaskAssembler, TaskCompiler

    loop_thread = threading.get_ident()
    compilation_threads = []
    original_compile = TaskCompiler.compile
    original_assemble = TaskAssembler.assemble

    def tracked_compile(compiler):
        compilation_threads.append(threading.get_ident())
        return original_compile(compiler)

    def tracked_assemble(assembler):
        compilation_threads.append(threading.get_ident())
        return original_assemble(assembler)

    monkeypatch.setattr(TaskCompiler, "compile", tracked_compile)
    monkeypatch.setattr(TaskAssembler, "assemble", tracked_assemble)

    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        statements = []

        def record_sql(connection, cursor, statement, parameters, context, many):
            statements.append(statement)

        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("large-graph")
            workload = await repository.create_campaign_workload(
                "large-graph",
                None,
                {},
                [],
                [],
                [
                    {
                        "definition": {
                            "name": "large-graph",
                            "payloads": {"p": {"seed": "p", "text": "hello"}},
                            "workflow": [
                                {
                                    "repeat": {
                                        "name": "rounds",
                                        "count": 12,
                                        "node": {
                                            "parallel": {
                                                "name": "wave",
                                                "count": 100,
                                                "node": {
                                                    "invoke": {
                                                        "name": "request",
                                                        "payload": "p",
                                                    }
                                                },
                                            }
                                        },
                                    }
                                }
                            ],
                        },
                        "runner_template": {
                            "label": "large-graph",
                            "metadata": {},
                            "benchmark": {
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
                                "mean_input_tokens": 2,
                                "stddev_input_tokens": 0,
                                "mean_output_tokens": 2,
                                "stddev_output_tokens": 0,
                                "repeat": 1,
                                "concurrent_workers": 1,
                            },
                        },
                    }
                ],
                "test",
            )
            assert workload["summary"]["task_nodes"] == 1200
            assert len(compilation_threads) == 2
            assert all(thread != loop_thread for thread in compilation_threads)
            event.listen(
                database.engine.sync_engine, "before_cursor_execute", record_sql
            )
            snapshot = await repository.runtime_snapshot()
            assert len(snapshot["instances"]) == 1
            nodes = snapshot["instances"][0]["nodes"]
            assert len(nodes) == 100
            assert sum(not node["dependencies"] for node in nodes) == 100
            node_queries = [sql for sql in statements if "FROM execution_nodes" in sql]
            assert len(node_queries) == 1
            assert "runner_template" not in node_queries[0]
            assert "lineage" in node_queries[0]
            statements.clear()
            first_node = next(node for node in nodes if not node["dependencies"])
            runner = await repository.activate_node(
                first_node["node_id"], datetime.now(timezone.utc)
            )
            assert runner is not None
            assert await repository.start_selected(runner["runner_id"], "large-graph")
            assert await repository.complete_runner(
                runner["runner_id"],
                {},
                [{"error_code": None, "request_metadata": {"prompt_hash": "stable"}}],
                0,
                "",
                "",
            )
            after_completion = await repository.runtime_snapshot()
            assert after_completion["instances"][0]["state"] == "active"
            assert len(after_completion["instances"][0]["nodes"]) == 99
            # A synchronized wave must wait for every predecessor. Completing
            # the remaining first-wave requests releases exactly one next wave,
            # with all predecessor timestamps available for its timer barrier.
            for remaining_node in nodes:
                if remaining_node["node_id"] == first_node["node_id"]:
                    continue
                next_runner = await repository.activate_node(
                    remaining_node["node_id"], datetime.now(timezone.utc)
                )
                assert next_runner is not None
                assert await repository.start_selected(
                    next_runner["runner_id"], "large-graph"
                )
                assert await repository.complete_runner(
                    next_runner["runner_id"],
                    {},
                    [
                        {
                            "error_code": None,
                            "request_metadata": {"prompt_hash": "stable"},
                        }
                    ],
                    0,
                    "",
                    "",
                )
            next_snapshot = await repository.runtime_snapshot()
            frontier_nodes = next_snapshot["instances"][0]["nodes"]
            assert len(frontier_nodes) == 200
            assert (
                sum(node["terminal_status"] == "succeeded" for node in frontier_nodes)
                == 100
            )
            assert (
                sum(node["state"] in {"pending", "blocked"} for node in frontier_nodes)
                == 100
            )
            assert not any("LEFT OUTER JOIN runners" in sql for sql in statements)
        finally:
            if event.contains(
                database.engine.sync_engine, "before_cursor_execute", record_sql
            ):
                event.remove(
                    database.engine.sync_engine, "before_cursor_execute", record_sql
                )
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())

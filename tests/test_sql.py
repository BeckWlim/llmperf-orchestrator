"""Opt-in PostgreSQL state-machine, concurrency, and Campaign fairness tests."""

import asyncio
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy.dialects import postgresql

pytest.importorskip("sqlalchemy")
pytest.importorskip("asyncpg")

from llmperf_backend.models import DatabaseConfig
from llmperf_backend.persistence import (
    Base,
    Database,
    RunnerRepository,
)


def _benchmark(model):
    return {
        "provider": "test",
        "model": model,
        "adapter": "openai",
        "timeout_seconds": 10,
        "max_completed_requests": 1,
        "concurrent_requests": 1,
        "mean_input_tokens": 40,
        "stddev_input_tokens": 0,
        "mean_output_tokens": 10,
        "stddev_output_tokens": 0,
        "additional_sampling_params": {},
    }


def test_task_lock_scope():
    statement = RunnerRepository._task_dispatch_rows_statement("task-instance")
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "LEFT OUTER JOIN benchmark_runners" in compiled_sql
    assert compiled_sql.endswith("FOR UPDATE OF benchmark_runner_dispatches")


def test_postgres_lifecycle(postgresql_url):
    async def exercise_repository():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.run_sync(Base.metadata.create_all)

            repository = RunnerRepository(database)
            campaign = await repository.create_campaign(
                "postgres-integration",
                "real PostgreSQL repository test",
                {"database": "postgresql"},
                "bootstrap-test",
            )
            assert campaign is not None
            runner = await repository.create_runner(
                {
                    "model": "glm-test",
                    "adapter": "openai",
                    "timeout_seconds": 10,
                    "max_completed_requests": 1,
                    "concurrent_requests": 1,
                    "mean_input_tokens": 40,
                    "stddev_input_tokens": 0,
                    "mean_output_tokens": 10,
                    "stddev_output_tokens": 0,
                    "additional_sampling_params": {},
                },
                {"suite": "postgresql"},
                "bootstrap-test",
                campaign_id=campaign["campaign_id"],
                label="postgres-runner",
            )
            assert runner is not None
            claimed = await repository.claim_next("integration-scheduler")
            assert claimed is not None
            assert claimed["runner_id"] == runner["runner_id"]

            committed = await repository.complete_runner(
                runner["runner_id"],
                {"num_completed_requests": 1},
                [{"ttft_s": 0.1, "kv_cache_hit_rate": 0.75}],
                0,
                "",
                "",
            )
            assert committed is True
            results = await repository.get_results(runner["runner_id"], True)
            assert results is not None
            assert results["status"] == "succeeded"
            assert results["requests"][0]["kv_cache_hit_rate"] == 0.75

            user = await repository.upsert_trusted_client(
                "postgres-operator",
                "0123456789abcdef",
                "normalized-test-public-key",
                "operator",
                "PostgreSQL Operator",
                "operator@example.test",
                "bootstrap-test",
                120,
            )
            assert user is not None
            assert user["role"] == "operator"
            trusted = await repository.get_trusted_client_by_key_id("0123456789abcdef")
            assert trusted is not None
            assert trusted["username"] == "postgres-operator"

            plan = await repository.create_runner_plan(
                campaign["campaign_id"],
                {
                    "name": "postgres-plan",
                    "timezone": "Asia/Shanghai",
                    "starts_at": datetime.now(timezone.utc) - timedelta(seconds=31),
                    "ends_at": None,
                    "max_occurrences": 2,
                    "recurrence": {"kind": "interval", "every_seconds": 30},
                    "overlap_policy": "queue",
                    "misfire_grace_seconds": 60,
                },
                {
                    "label": "planned-runner",
                    "metadata": {"suite": "planner"},
                    "benchmark": {
                        "provider": "test",
                        "model": "planned-model",
                        "adapter": "openai",
                    },
                },
                "bootstrap-test",
            )
            assert plan is not None
            assert plan["status"] == "active"
            emitted = await asyncio.gather(
                repository.materialize_due_work(10),
                repository.materialize_due_work(10),
            )
            emitted_count = sum(emitted) + await repository.materialize_due_work(10)
            assert emitted_count == 2
            persisted_plan = await repository.get_runner_plan(plan["runner_plan_id"])
            assert persisted_plan is not None
            assert persisted_plan["status"] == "completed"
            assert persisted_plan["emitted_count"] == 2
            queued = await repository.list_runners("queued", 100, 0, full=True)
            planned = [
                runner
                for runner in queued
                if runner["runner_plan_id"] == plan["runner_plan_id"]
            ]
            assert {runner["plan_occurrence"] for runner in planned} == {0, 1}

            skip_plan = await repository.create_runner_plan(
                campaign["campaign_id"],
                {
                    "name": "postgres-skip-plan",
                    "timezone": "Asia/Shanghai",
                    "starts_at": datetime.now(timezone.utc) - timedelta(seconds=31),
                    "ends_at": None,
                    "max_occurrences": 2,
                    "recurrence": {"kind": "interval", "every_seconds": 30},
                    "overlap_policy": "skip",
                    "misfire_grace_seconds": 60,
                },
                {
                    "label": "skip-planned-runner",
                    "metadata": {"suite": "planner"},
                    "benchmark": {
                        "provider": "test",
                        "model": "planned-model",
                        "adapter": "openai",
                    },
                },
                "bootstrap-test",
            )
            assert skip_plan is not None
            assert await repository.materialize_due_work(10) == 1
            assert await repository.materialize_due_work(10) == 0
            persisted_skip = await repository.get_runner_plan(
                skip_plan["runner_plan_id"]
            )
            assert persisted_skip is not None
            assert persisted_skip["status"] == "completed"
            assert persisted_skip["emitted_count"] == 1
            assert persisted_skip["skipped_count"] == 1
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())


def test_task_advancement(postgresql_url):
    async def exercise_repository():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.run_sync(Base.metadata.create_all)

            repository = RunnerRepository(database)
            workload = await repository.create_campaign_workload(
                "postgres-task",
                "PostgreSQL task Dispatch advancement regression test",
                {},
                [],
                [],
                [
                    {
                        "definition": {
                            "name": "postgres-task",
                            "instances": {"trials": 1, "seed": 7},
                            "payloads": {"replay": {"seed_namespace": "replay"}},
                            "workflow": [
                                {
                                    "invoke": {
                                        "name": "prime",
                                        "payload": "replay",
                                    }
                                },
                                {
                                    "invoke": {
                                        "name": "probe",
                                        "payload": "replay",
                                    }
                                },
                            ],
                        },
                        "runner_template": {
                            "label": "postgres-task",
                            "metadata": {},
                            "benchmark": _benchmark("task-model"),
                        },
                    }
                ],
                "bootstrap-test",
            )
            campaign_id = workload["campaign"]["campaign_id"]
            assert await repository.materialize_due_work(10) == 1
            prime_runner = await repository.claim_next("task-scheduler")
            assert prime_runner is not None
            assert prime_runner["label"] == "postgres-task:prime"

            committed = await repository.complete_runner(
                prime_runner["runner_id"],
                {"num_completed_requests": 1},
                [
                    {
                        "request_metadata": {
                            "prompt_hash": "stable-prompt-hash",
                            "payload_evidence": {"record_index": 1},
                        }
                    }
                ],
                0,
                "",
                "",
            )
            assert committed is True
            campaign_status = await repository.get_campaign_status(campaign_id)
            assert campaign_status is not None
            assert campaign_status["task_instance_status_counts"]["active"] == 1
            assert campaign_status["dispatch_status_counts"] == {
                "blocked": 0,
                "pending": 1,
                "emitted": 1,
                "cancelled": 0,
            }
            assert await repository.materialize_due_work(10) == 1
            probe_runner = await repository.claim_next("task-scheduler")
            assert probe_runner is not None
            assert probe_runner["label"] == "postgres-task:probe"
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())


def test_campaign_claim_fairness(postgresql_url):
    async def exercise_repository():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.run_sync(Base.metadata.create_all)
            repository = RunnerRepository(database)
            campaign_a = await repository.create_campaign(
                "campaign-a", "fairness A", {}, "bootstrap-test"
            )
            campaign_b = await repository.create_campaign(
                "campaign-b", "fairness B", {}, "bootstrap-test"
            )
            assert campaign_a is not None
            assert campaign_b is not None
            for index in range(2):
                await repository.create_runner(
                    _benchmark(f"a-{index}"),
                    {},
                    "bootstrap-test",
                    campaign_id=campaign_a["campaign_id"],
                )
            await repository.create_runner(
                _benchmark("b-0"),
                {},
                "bootstrap-test",
                campaign_id=campaign_b["campaign_id"],
            )

            claimed = await asyncio.gather(
                repository.claim_next("fair-slot-1"),
                repository.claim_next("fair-slot-2"),
            )
            first_claim, second_claim = claimed
            assert first_claim is not None
            assert second_claim is not None
            assert {
                runner["campaign_id"] for runner in (first_claim, second_claim)
            } == {
                campaign_a["campaign_id"],
                campaign_b["campaign_id"],
            }
            remaining = await repository.claim_next("fair-slot-3")
            assert remaining is not None
            assert remaining["campaign_id"] == campaign_a["campaign_id"]
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())

"""Opt-in tests against a dedicated disposable PostgreSQL database."""

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("asyncpg")

from llmperf_backend.models import DatabaseConfig
from llmperf_backend.persistence import Base, Database, RunnerRepository


@pytest.mark.postgresql
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
            runner = await repository.create_runner(
                {
                    "model": "glm-test",
                    "llm_api": "openai",
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
            claimed = await repository.claim_next("integration-scheduler")
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
            assert user["role"] == "operator"
            trusted = await repository.get_trusted_client_by_key_id("0123456789abcdef")
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
                        "llm_api": "openai",
                    },
                },
                "bootstrap-test",
            )
            assert plan["status"] == "active"
            emitted = await asyncio.gather(
                repository.materialize_due_plans(10),
                repository.materialize_due_plans(10),
            )
            emitted_count = sum(emitted) + await repository.materialize_due_plans(10)
            assert emitted_count == 2
            persisted_plan = await repository.get_runner_plan(plan["runner_plan_id"])
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
                        "llm_api": "openai",
                    },
                },
                "bootstrap-test",
            )
            assert await repository.materialize_due_plans(10) == 1
            assert await repository.materialize_due_plans(10) == 0
            persisted_skip = await repository.get_runner_plan(
                skip_plan["runner_plan_id"]
            )
            assert persisted_skip["status"] == "completed"
            assert persisted_skip["emitted_count"] == 1
            assert persisted_skip["skipped_count"] == 1
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())


@pytest.mark.postgresql
def test_cache_sweep_protocol(postgresql_url):
    async def exercise_repository():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.run_sync(Base.metadata.create_all)
            repository = RunnerRepository(database)
            benchmark = {
                "provider": "test",
                "model": "cache-model",
                "llm_api": "openai",
                "timeout_seconds": 10,
                "max_completed_requests": 1,
                "concurrent_requests": 1,
                "mean_input_tokens": 64,
                "stddev_input_tokens": 0,
                "mean_output_tokens": 1,
                "stddev_output_tokens": 0,
                "additional_sampling_params": {},
            }
            workload = await repository.create_campaign_workload(
                "cache-sweep",
                None,
                {},
                [],
                [],
                [
                    {
                        "definition": {
                            "name": "ttl",
                            "protocol": "cache-retention/v1",
                            "delay_seconds": [0],
                            "trials_per_delay": 1,
                            "seed": 7,
                            "cold_control": True,
                        },
                        "runner_template": {
                            "label": "ttl",
                            "metadata": {},
                            "benchmark": benchmark,
                        },
                    }
                ],
                "bootstrap-test",
            )
            campaign_id = workload["campaign"]["campaign_id"]
            planned = await repository.export_campaign(campaign_id, False)
            assert len(planned["dispatches"]) == 3
            prime_dispatch = next(
                item for item in planned["dispatches"] if item["role"] == "prime"
            )
            dependents = [
                item for item in planned["dispatches"] if item["role"] != "prime"
            ]
            assert prime_dispatch["state"] == "pending"
            assert all(item["state"] == "blocked" for item in dependents)
            assert all(
                item["parent_dispatch_id"] == prime_dispatch["dispatch_id"]
                for item in dependents
            )
            assert await repository.materialize_due_work(10) == 1
            prime = await repository.claim_next("scheduler")
            prompt_hash = "sha256:" + "a" * 64
            timestamp = datetime.now(timezone.utc).isoformat()
            assert await repository.complete_runner(
                prime["runner_id"],
                {"results": {}},
                [
                    {
                        "request_metadata": {"prompt_hash": prompt_hash},
                        "request_timing": {"completed_utc": timestamp},
                    }
                ],
                0,
                "",
                "",
            )
            waiting = await repository.get_campaign_status(campaign_id)
            assert waiting["status"] == "planned"
            unlocked = await repository.export_campaign(campaign_id, False)
            assert all(
                item["state"] == "pending" and item["due_at"] is not None
                for item in unlocked["dispatches"]
                if item["role"] != "prime"
            )
            assert await repository.materialize_due_plans(10) == 2

            for _ in range(2):
                runner = await repository.claim_next("scheduler")
                role = runner["metadata"]["protocol"]["role"]
                ttft = 0.2 if role == "warm" else 1.0
                cache = (
                    {
                        "request_hit_ratio": 1.0,
                        "weighted_token_hit_ratio": 0.8,
                    }
                    if role == "warm"
                    else {}
                )
                assert await repository.complete_runner(
                    runner["runner_id"],
                    {
                        "results": {
                            "ttft_s": {"quantiles": {"p50": ttft}},
                            "kv_cache": cache,
                        }
                    },
                    [
                        {
                            "request_metadata": {"prompt_hash": prompt_hash},
                            "request_timing": {"client_start_utc": timestamp},
                        }
                    ],
                    0,
                    "",
                    "",
                )

            exported = await repository.export_campaign(campaign_id, False)
            assert exported["version"] == 5
            assert exported["protocol_instances"][0]["state"] == "completed"
            assert (
                exported["protocol_analyses"][0]["curve"][0]["verdict"]
                == "accounting_observed"
            )
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())


@pytest.mark.postgresql
def test_cache_residency(postgresql_url):
    async def exercise_repository():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.run_sync(Base.metadata.create_all)
            repository = RunnerRepository(database)
            workload = await repository.create_campaign_workload(
                "cache-residency",
                None,
                {},
                [],
                [],
                [
                    {
                        "definition": {
                            "name": "daily-chain",
                            "protocol": "cache-residency/v1",
                            "schedule": {
                                "kind": "relative",
                                "offsets_seconds": [1, 2],
                            },
                            "mapping": "one_to_one",
                            "chains": 1,
                            "seed": 9,
                            "cold_control": False,
                        },
                        "runner_template": {
                            "label": "daily-chain",
                            "metadata": {},
                            "benchmark": {
                                "provider": "test",
                                "model": "cache-model",
                                "llm_api": "openai",
                                "timeout_seconds": 10,
                                "max_completed_requests": 1,
                                "concurrent_requests": 1,
                                "mean_input_tokens": 64,
                                "stddev_input_tokens": 0,
                                "mean_output_tokens": 1,
                                "stddev_output_tokens": 0,
                                "additional_sampling_params": {},
                            },
                        },
                    }
                ],
                "bootstrap-test",
            )
            campaign_id = workload["campaign"]["campaign_id"]
            planned = await repository.export_campaign(campaign_id, False)
            assert [item["role"] for item in planned["dispatches"]] == [
                "prime",
                "warm:0",
                "warm:1",
            ]
            assert await repository.materialize_due_work(10) == 1
            prompt_hash = "sha256:" + "b" * 64
            timestamp = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

            prime = await repository.claim_next("scheduler")
            assert prime["metadata"]["protocol"]["role"] == "prime"
            assert await repository.complete_runner(
                prime["runner_id"],
                {"results": {}},
                [
                    {
                        "request_metadata": {
                            "prompt_hash": f"{prompt_hash}-0",
                            "mapping_key": "chain-0:observation-0",
                        },
                        "request_timing": {"completed_utc": timestamp},
                    },
                    {
                        "request_metadata": {
                            "prompt_hash": f"{prompt_hash}-1",
                            "mapping_key": "chain-0:observation-1",
                        },
                        "request_timing": {"completed_utc": timestamp},
                    },
                ],
                0,
                "",
                "",
            )
            after_prime = await repository.export_campaign(campaign_id, False)
            states = {item["role"]: item["state"] for item in after_prime["dispatches"]}
            assert states == {
                "prime": "emitted",
                "warm:0": "pending",
                "warm:1": "blocked",
            }

            for observation_index in range(2):
                assert await repository.materialize_due_work(10) == 1
                warm = await repository.claim_next("scheduler")
                context = warm["metadata"]["protocol"]
                assert context["role"] == "warm"
                assert context["observation_index"] == observation_index
                assert await repository.complete_runner(
                    warm["runner_id"],
                    {
                        "results": {
                            "ttft_s": {"quantiles": {"p50": 0.2}},
                            "kv_cache": {"request_hit_ratio": 1.0},
                        }
                    },
                    [
                        {
                            "request_metadata": {
                                "prompt_hash": f"{prompt_hash}-{observation_index}",
                                "mapping_key": (
                                    f"chain-0:observation-{observation_index}"
                                ),
                            },
                            "request_timing": {"client_start_utc": timestamp},
                        }
                    ],
                    0,
                    "",
                    "",
                )

            exported = await repository.export_campaign(campaign_id, False)
            assert exported["protocol_instances"][0]["state"] == "completed"
            analysis = exported["protocol_analyses"][0]
            assert analysis["protocol"] == "cache-residency/v1"
            assert len(analysis["curve"]) == 2
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())

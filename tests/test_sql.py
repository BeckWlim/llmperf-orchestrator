"""Opt-in tests against a dedicated disposable PostgreSQL database."""

import asyncio
import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("asyncpg")

from sqlalchemy.engine import make_url

from llmperf_backend.models import DatabaseConfig
from llmperf_backend.persistence import Base, Database, RunnerRepository


TEST_DATABASE_URL = os.environ.get("LLMPERF_TEST_DATABASE_URL")


@pytest.mark.postgresql
def test_lifecycle():
    if not TEST_DATABASE_URL:
        pytest.skip("LLMPERF_TEST_DATABASE_URL is not configured")
    url = make_url(TEST_DATABASE_URL)
    database_name = url.database or ""
    assert url.drivername == "postgresql+asyncpg"
    assert (
        "test" in database_name.lower()
    ), "Refusing to reset a PostgreSQL database whose name does not contain 'test'"

    async def exercise_repository():
        database = Database(DatabaseConfig(url=TEST_DATABASE_URL))
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
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise_repository())

"""Worker progress combines durable evidence and age-bounded physical telemetry."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import JsonValue

from llmperf_backend.api.progress import attach_progress
from llmperf_backend.configuration.models import SchedulerConfig


@pytest.mark.parametrize("age", [1.0, 30.0])
def test_campaign_progress(age):
    repository = Mock()
    repository.campaign_worker_progress = AsyncMock(
        return_value={
            campaign: {
                "planned": 10 * count,
                "created": 0,
                "unknown_created": count,
                "submitted": 4 * count,
                "not_submitted": 6 * count,
                "pending_results": 2 * count,
                "succeeded": count,
                "failed": count,
                "cancelled": 0,
                "indeterminate": 0,
                "creation_phases": {"running": count},
            }
            for campaign, count in (("a", 2), ("b", 1))
        }
    )
    repository.worker_progress = AsyncMock(
        return_value={
            identifier: {
                "campaign_id": campaign,
                "planned": 10,
                "created": None,
                "phase": "running",
                "submitted": 4,
                "not_submitted": 6,
                "pending_results": 2,
                "succeeded": 1,
                "failed": 1,
                "cancelled": 0,
                "indeterminate": 0,
            }
            for identifier, campaign in (("one", "a"), ("two", "a"), ("three", "b"))
        }
    )
    scheduler = Mock()
    scheduler.config = SchedulerConfig()
    scheduler.worker_status.return_value = {
        "pool_ready": True,
        "snapshot_age_seconds": age,
        "slots_busy": 3,
        "slots_free": 61,
        "slots_total": 64,
        "runners": {
            identifier: {
                "created": 10,
                "phase": "submitting",
                "admission_waiting": 1,
                "queued": 1,
                "running": 1,
            }
            for identifier in ("one", "two", "three")
        },
    }
    records: list[dict[str, JsonValue]] = [{"campaign_id": "a"}, {"campaign_id": "b"}]
    asyncio.run(attach_progress(records, repository, scheduler, campaigns=True))
    repository.campaign_worker_progress.assert_awaited_once_with(["a", "b"])
    repository.worker_progress.assert_awaited_once_with(
        runner_ids=["one", "two", "three"]
    )
    first = records[0]["workers"]
    second = records[1]["workers"]
    slots = records[0]["request_slots"]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert isinstance(slots, dict)
    assert first["created"] == first["planned"] == 20
    assert first["submitted"] == 8
    assert first["running"] == (2 if age == 1 else None)
    assert first["queued"] == (2 if age == 1 else None)
    creation_phases = first["creation_phases"]
    assert isinstance(creation_phases, dict)
    assert creation_phases["submitting"] == 2
    assert second["submitted"] == 4
    assert records[0]["request_slots"] == records[1]["request_slots"]
    assert slots["scope"] == "global"

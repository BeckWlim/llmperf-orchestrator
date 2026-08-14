import asyncio
from datetime import datetime, timezone

import pytest

from llmperf_backend.models import PlannerConfig, RunnerPlanPreview
from llmperf_backend.planner import (
    Planner,
    next_fire_at,
    next_fire_details,
    preview_fires,
)
from llmperf_backend.persistence import RunnerRepository


UTC = timezone.utc


def test_interval_recurrence():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    recurrence = {"kind": "interval", "every_seconds": 600}

    first = next_fire_at("Asia/Shanghai", recurrence, start)
    second = next_fire_at("Asia/Shanghai", recurrence, start, first)

    assert first == start
    assert second == datetime(2026, 8, 13, 0, 10, tzinfo=UTC)


def test_shanghai_calendar():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    recurrence = {
        "kind": "calendar",
        "frequency": "daily",
        "interval": 1,
        "local_time": "09:30:00",
    }

    fire_at = next_fire_at("Asia/Shanghai", recurrence, start)

    assert fire_at == datetime(2026, 8, 13, 1, 30, tzinfo=UTC)


def test_weekly_calendar():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    recurrence = {
        "kind": "calendar",
        "frequency": "weekly",
        "interval": 1,
        "local_time": "09:30:00",
        "weekdays": ["mon", "wed", "fri"],
    }

    fire_at = next_fire_at("Asia/Shanghai", recurrence, start)

    assert fire_at == datetime(2026, 8, 14, 1, 30, tzinfo=UTC)


def test_dst_gap_skip():
    start = datetime(2026, 3, 8, 0, 0, tzinfo=UTC)
    recurrence = {
        "kind": "calendar",
        "frequency": "daily",
        "interval": 1,
        "local_time": "02:30:00",
    }

    fire_at, adjustments = next_fire_details("America/New_York", recurrence, start)

    assert fire_at == datetime(2026, 3, 9, 6, 30, tzinfo=UTC)
    assert adjustments[0]["reason"] == "nonexistent_local_time"
    assert adjustments[0]["policy"] == "skip"


def test_dst_overlap_first():
    start = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    recurrence = {
        "kind": "calendar",
        "frequency": "daily",
        "interval": 1,
        "local_time": "01:30:00",
    }

    fire_at, adjustments = next_fire_details("America/New_York", recurrence, start)

    assert fire_at == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    assert adjustments[0]["reason"] == "ambiguous_local_time"
    assert adjustments[0]["policy"] == "first"


def test_preview_boundary():
    payload = RunnerPlanPreview.model_validate(
        {
            "timezone": "Asia/Shanghai",
            "starts_at": "2026-08-13T00:00:00Z",
            "max_occurrences": 2,
            "recurrence": {"kind": "interval", "every_seconds": 60},
            "count": 10,
        }
    )
    timing = {
        "timezone": payload.timezone,
        "starts_at": payload.starts_at,
        "ends_at": payload.ends_at,
        "max_occurrences": payload.max_occurrences,
        "recurrence": payload.recurrence.model_dump(mode="json"),
    }

    items = preview_fires(timing, payload.count)

    assert [item["occurrence"] for item in items] == [0, 1]
    assert items[1]["scheduled_for"] == datetime(2026, 8, 13, 0, 1, tzinfo=UTC)


def test_immediate_preview():
    effective_start = datetime(2026, 8, 13, 8, 0, tzinfo=UTC)
    timing = {
        "timezone": "Asia/Shanghai",
        "starts_at": None,
        "ends_at": None,
        "max_occurrences": 2,
        "recurrence": {"kind": "interval", "every_seconds": 30},
    }

    items = preview_fires(timing, 2, default_starts_at=effective_start)

    assert items[0]["scheduled_for"] == effective_start
    assert items[1]["scheduled_for"] == datetime(2026, 8, 13, 8, 0, 30, tzinfo=UTC)


def test_immediate_plan():
    database_now = datetime(2026, 8, 13, 8, 0, tzinfo=UTC)

    plan, adjustments = RunnerRepository._new_runner_plan(
        "campaign-1",
        {
            "name": "immediate",
            "timezone": "Asia/Shanghai",
            "starts_at": None,
            "ends_at": None,
            "max_occurrences": 8,
            "recurrence": {"kind": "interval", "every_seconds": 30},
            "overlap_policy": "queue",
        },
        {"benchmark": {"provider": "aliyun", "model": "deepseek-v4-pro"}},
        "operator",
        database_now,
    )

    assert plan.starts_at == database_now
    assert plan.next_fire_at == database_now
    assert adjustments == []


def test_overlap_policy():
    with pytest.raises(ValueError, match="overlap_policy"):
        RunnerPlanPreview.model_validate(
            {
                "timezone": "Asia/Shanghai",
                "starts_at": "2026-08-13T00:00:00Z",
                "max_occurrences": 2,
                "recurrence": {"kind": "interval", "every_seconds": 30},
                "overlap_policy": "parallel",
            }
        )


def test_immediate_grace():
    with pytest.raises(ValueError, match="misfire_grace_seconds"):
        RunnerPlanPreview.model_validate(
            {
                "timezone": "Asia/Shanghai",
                "max_occurrences": 2,
                "recurrence": {"kind": "interval", "every_seconds": 30},
                "misfire_grace_seconds": 0,
            }
        )


def test_campaign_runtime():
    planned = RunnerRepository._campaign_runtime({}, {"active": 1})
    waiting_protocol = RunnerRepository._campaign_runtime(
        {"succeeded": 1}, {}, {"active": 1}
    )
    queued = RunnerRepository._campaign_runtime(
        {"queued": 2, "succeeded": 100_000}, {"completed": 1}
    )
    completed = RunnerRepository._campaign_runtime({}, {"completed": 1})
    succeeded = RunnerRepository._campaign_runtime({"succeeded": 8}, {"completed": 1})
    partial = RunnerRepository._campaign_runtime(
        {"succeeded": 7, "failed": 1}, {"completed": 1}
    )
    failed = RunnerRepository._campaign_runtime({"failed": 8}, {"completed": 1})
    protocol_failed = RunnerRepository._campaign_runtime(
        {"succeeded": 3}, {}, {"failed": 1}
    )
    cancelled = RunnerRepository._campaign_runtime({"cancelled": 8}, {"cancelled": 1})

    assert planned["status"] == "planned"
    assert planned["outcome"] == "pending"
    assert planned["runner_plan_count"] == 1
    assert waiting_protocol["status"] == "planned"
    assert waiting_protocol["protocol_instance_status_counts"]["active"] == 1
    assert queued["status"] == "queued"
    assert queued["outcome"] == "pending"
    assert queued["runner_count"] == 100_002
    assert completed["status"] == "completed"
    assert completed["outcome"] == "no_runs"
    assert succeeded["status"] == "completed"
    assert succeeded["outcome"] == "succeeded"
    assert succeeded["has_failures"] is False
    assert partial["status"] == "completed"
    assert partial["outcome"] == "partial_failed"
    assert partial["has_failures"] is True
    assert failed["status"] == "completed"
    assert failed["outcome"] == "failed"
    assert protocol_failed["outcome"] == "failed"
    assert protocol_failed["has_failures"] is True
    assert cancelled["status"] == "cancelled"
    assert cancelled["outcome"] == "cancelled"


def test_planner_runtime():
    class Repository:
        def __init__(self):
            self.called = asyncio.Event()
            self.identity = None

        async def materialize_due_work(self, limit, planner_id):
            assert limit == 3
            self.identity = planner_id
            self.called.set()
            return 0

    async def exercise():
        repository = Repository()
        planner = Planner(
            repository,
            PlannerConfig(poll_interval_seconds=0.01, batch_size=3),
        )
        assert planner.status()["status"] == "stopped"
        await planner.start()
        await asyncio.wait_for(repository.called.wait(), timeout=1)
        assert repository.identity == planner.status()["planner_id"]
        assert planner.status()["status"] == "running"
        await planner.stop()
        assert planner.status()["status"] == "stopped"

    asyncio.run(exercise())

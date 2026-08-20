from datetime import datetime, timezone

import pytest

from llmperf_backend.persistence import (
    BenchmarkRunnerDispatchRecord,
    RunnerRecordFactory,
)


UTC = timezone.utc


def test_factory_template():
    due_at = datetime(2026, 8, 20, 8, 30, tzinfo=UTC)
    dispatch = BenchmarkRunnerDispatchRecord(
        id="dispatch-1",
        campaign_id="campaign-1",
        runner_plan_id="plan-1",
        dispatch_key="7",
        due_at=due_at,
        state="pending",
        runner_template={
            "label": "planned-runner",
            "benchmark": {
                "provider": "test",
                "model": "test-model",
                "temperature": float("inf"),
            },
            "metadata": {"tags": ("planner", "factory")},
        },
        lineage={
            "created_by": "operator",
            "plan_occurrence": 7,
            "plan_template_version": "1.0.0",
        },
    )

    runner = RunnerRecordFactory.from_dispatch(dispatch)

    assert runner.campaign_id == "campaign-1"
    assert runner.runner_plan_id == "plan-1"
    assert runner.plan_occurrence == 7
    assert runner.scheduled_for == due_at
    assert runner.plan_template_version == "1.0.0"
    assert runner.label == "planned-runner"
    assert runner.created_by == "operator"
    assert runner.status == "queued"
    assert runner.benchmark_config["temperature"] is None
    assert runner.user_metadata == {"tags": ["planner", "factory"]}


def test_factory_defaults():
    runner = RunnerRecordFactory.from_template(
        {"benchmark": {"provider": "test", "model": "test-model"}},
        created_by="operator",
    )

    assert runner.campaign_id is None
    assert runner.runner_plan_id is None
    assert runner.label is None
    assert runner.status == "queued"
    assert runner.user_metadata == {}


def test_factory_rejects_template():
    with pytest.raises(ValueError, match="runner_template.benchmark"):
        RunnerRecordFactory.from_template(
            {"benchmark": "not-an-object"},
            created_by="operator",
        )

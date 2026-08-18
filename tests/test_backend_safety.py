"""Static admission limits and runtime host-memory circuit-breaker tests."""

from datetime import datetime, timezone

import pytest

from llmperf_backend.models import PerformanceGuardConfig
from llmperf_backend.safety import (
    RuntimePerformanceGuard,
    WorkloadSafetyError,
    assess_workload,
)


NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)
BENCHMARK = {
    "max_completed_requests": 10,
    "concurrent_requests": 2,
    "mean_input_tokens": 100,
    "mean_output_tokens": 20,
}


def test_workload_costs():
    plan = {
        "plan": {
            "timezone": "UTC",
            "starts_at": NOW,
            "ends_at": None,
            "max_occurrences": 3,
            "recurrence": {"kind": "interval", "every_seconds": 60},
        },
        "runner_template": {"benchmark": BENCHMARK},
    }
    retention = {
        "definition": {
            "matrix": {"delay": [0, 60]},
            "trials": 2,
            "sequence": [
                {"kind": "invoke"},
                {"kind": "parallel", "invokes": [{}, {}]},
            ],
        },
        "runner_template": {"benchmark": BENCHMARK},
    }

    result = assess_workload(
        [{"benchmark": BENCHMARK}],
        [plan],
        [retention],
        PerformanceGuardConfig(),
        scheduler_slots=4,
        now=NOW,
    )

    assert result["safe"] is True
    assert result["metrics"]["planned_runners"] == 16
    assert result["metrics"]["provider_requests"] == 52
    assert result["metrics"]["token_budget"] == 6_240
    assert result["metrics"]["effective_concurrency"] == 8


def test_promotion_costs():
    promotion = {
        "definition": {
            "matrix": {"warmups": [0, 2, 4], "quiet": [60, 300]},
            "trials": 2,
            "sequence": [
                {"kind": "invoke"},
                {
                    "kind": "repeat",
                    "count": {"dimension": "warmups"},
                },
                {"kind": "parallel", "invokes": [{}, {}]},
            ],
        },
        "runner_template": {"benchmark": BENCHMARK},
    }

    result = assess_workload(
        [],
        [],
        [promotion],
        PerformanceGuardConfig(),
        scheduler_slots=4,
        now=NOW,
    )

    # For each quiet/trial block: (Prime + Probe + Control) * 3 cells + 0+2+4 Warmups.
    assert result["metrics"]["planned_runners"] == 60
    assert result["metrics"]["provider_requests"] == 60
    assert result["metrics"]["token_budget"] == 7_200


def test_workload_rejection():
    guard = PerformanceGuardConfig(max_campaign_provider_requests=9)

    with pytest.raises(WorkloadSafetyError) as captured:
        assess_workload(
            [{"benchmark": BENCHMARK}], [], [], guard, scheduler_slots=1, now=NOW
        )

    assessment = captured.value.assessment
    assert assessment["safe"] is False
    assert assessment["risks"][0]["code"] == "provider_requests_limit"


def test_runtime_hysteresis():
    samples = iter(
        [
            {"available": True, "utilization": 0.91},
            {"available": True, "utilization": 0.85},
            {"available": True, "utilization": 0.79},
        ]
    )
    guard = RuntimePerformanceGuard(
        PerformanceGuardConfig(sample_interval_seconds=0.001),
        sampler=lambda: next(samples),
    )

    assert guard.allow_claim() is False
    guard._last_sample_at = 0
    assert guard.allow_claim() is False
    guard._last_sample_at = 0
    assert guard.allow_claim() is True


def test_object_store_watermarks():
    with pytest.raises(ValueError, match="resume_ray_object_store"):
        PerformanceGuardConfig(
            min_ray_object_store_available_ratio=0.2,
            resume_ray_object_store_available_ratio=0.1,
        )


def test_ray_capacity():
    with pytest.raises(WorkloadSafetyError) as captured:
        assess_workload(
            [{"benchmark": BENCHMARK}] * 4,
            [],
            [],
            PerformanceGuardConfig(),
            scheduler_slots=4,
            ray_actor_capacity=4,
            now=NOW,
        )

    assert captured.value.assessment["risks"][0]["code"] == (
        "ray_actor_demand_limit"
    )

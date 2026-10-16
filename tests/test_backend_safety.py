"""Static admission limits and runtime host-memory circuit-breaker tests."""

from datetime import datetime, timezone

import pytest

from llmperf_backend.configuration.models import PerformanceGuardConfig
from llmperf_backend.compiler.safety import (
    RuntimePerformanceGuard,
    WorkloadSafetyError,
    assess_workload,
)


NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)
BENCHMARK = {
    "repeat": 10,
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
            "recurrence": {"kind": "interval", "every": 60},
        },
        "runner_template": {"benchmark": BENCHMARK},
    }
    retention = {
        "definition": {
            "name": "retention",
            "instances": {
                "matrix": {"delay": [0, 60]},
                "trials": 2,
            },
            "payloads": {"replay": {"seed": "replay"}},
            "workflow": [
                {"invoke": {"name": "prime", "payload": "replay"}},
                {
                    "parallel": {
                        "name": "observation",
                        "after": "$delay",
                        "branches": [
                            {"invoke": {"name": "warm", "payload": "replay"}},
                            {"invoke": {"name": "cold", "payload": "replay"}},
                        ],
                    }
                },
            ],
        },
        "runner_template": {"benchmark": {**BENCHMARK, "repeat": 1}},
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
            "name": "promotion",
            "instances": {
                "matrix": {"warmups": [0, 2, 4], "quiet": [60, 300]},
                "trials": 2,
            },
            "payloads": {"replay": {"seed": "replay"}},
            "workflow": [
                {"invoke": {"name": "prime", "payload": "replay"}},
                {
                    "repeat": {
                        "name": "warmups",
                        "count": "$warmups",
                        "node": {"invoke": {"name": "warm", "payload": "replay"}},
                    }
                },
                {
                    "parallel": {
                        "name": "observation",
                        "after": "$quiet",
                        "branches": [
                            {"invoke": {"name": "probe", "payload": "replay"}},
                            {"invoke": {"name": "cold", "payload": "replay"}},
                        ],
                    }
                },
            ],
        },
        "runner_template": {"benchmark": {**BENCHMARK, "repeat": 1}},
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


def test_embedded_task_runner():
    task_definition = {
        "name": "embedded-runner",
        "instances": {"trials": 2},
        "payloads": {"replay": {"seed": "replay"}},
        "workflow": [{"invoke": {"name": "prime", "payload": "replay"}}],
        "runner": {"benchmark": {**BENCHMARK, "repeat": 1}},
    }

    result = assess_workload(
        [],
        [],
        [
            {
                "definition": task_definition,
                "runner_template": {"benchmark": {**BENCHMARK, "repeat": 1}},
            }
        ],
        PerformanceGuardConfig(),
        scheduler_slots=1,
        now=NOW,
    )

    assert result["metrics"]["planned_runners"] == 2
    assert result["metrics"]["provider_requests"] == 2


def test_grouped_capacity():
    task = {
        "name": "load64",
        "payloads": {"replay": {"seed": "replay"}},
        "workflow": [
            {"invoke": {"name": "prime", "payload": "replay"}},
            {
                "parallel": {
                    "name": "load",
                    "branches": [
                        {"invoke": {"name": f"request-{index}", "payload": "replay"}}
                        for index in range(64)
                    ],
                }
            },
            {"invoke": {"name": "join", "payload": "replay"}},
        ],
    }
    result = assess_workload(
        [],
        [],
        [
            {
                "definition": task,
                "runner_template": {"benchmark": {**BENCHMARK, "repeat": 1}},
            }
        ],
        PerformanceGuardConfig(max_runner_concurrency=64, max_effective_concurrency=64),
        scheduler_slots=64,
        ray_actor_capacity=8,
        ray_actor_max_concurrency=8,
        now=NOW,
    )
    assert result["metrics"]["planned_runners"] == 66
    assert result["metrics"]["provider_requests"] == 66
    assert result["metrics"]["effective_concurrency"] == 64
    assert result["metrics"]["ray_actor_demand"] == 8


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
    result = assess_workload(
        [{"benchmark": BENCHMARK}] * 4,
        [],
        [],
        PerformanceGuardConfig(),
        scheduler_slots=4,
        ray_actor_capacity=4,
        now=NOW,
    )
    assert result["metrics"]["effective_concurrency"] == 4
    assert any(
        warning["code"] == "request_capacity_limited" for warning in result["warnings"]
    )


@pytest.mark.parametrize(
    "runners,concurrency,capacity,expected_demand",
    [(1, 64, 8, 8), (8, 8, 8, 8), (8, 1, 8, 1), (2, 9, 4, 3)],
)
def test_concurrent_actor_capacity(runners, concurrency, capacity, expected_demand):
    benchmark = dict(BENCHMARK, concurrent_requests=concurrency, repeat=100)
    result = assess_workload(
        [{"benchmark": benchmark}] * runners,
        [],
        [],
        PerformanceGuardConfig(max_runner_concurrency=64, max_effective_concurrency=64),
        scheduler_slots=8,
        ray_actor_capacity=capacity,
        ray_actor_max_concurrency=8,
    )
    assert result["metrics"]["ray_actor_demand"] == expected_demand
    assert result["metrics"]["effective_concurrency"] == runners * concurrency


def test_shared_actor_rounding():
    benchmark = dict(BENCHMARK, concurrent_requests=9, repeat=100)
    result = assess_workload(
        [{"benchmark": benchmark}] * 2,
        [],
        [],
        PerformanceGuardConfig(),
        scheduler_slots=2,
        ray_actor_capacity=3,
        ray_actor_max_concurrency=8,
    )
    assert result["metrics"]["ray_actor_demand"] == 3
    assert result["metrics"]["effective_concurrency"] == 18

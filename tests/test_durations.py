from datetime import datetime, timedelta, timezone

import pytest

from llmperf.configuration.duration import Duration
from llmperf_backend.compiler.tasks import TaskRecipe
from llmperf_backend.configuration.models import (
    BenchmarkConfig,
    CacheProbeConfig,
    RequestStepSpec,
    RunnerPlanRecurrence,
)
from llmperf_backend.runtime.planner import next_fire_at


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        (2, 2.0),
        (0.5, 0.5),
        ({"value": 500, "unit": "ms"}, 0.5),
        ({"value": 2, "unit": "m"}, 120.0),
        ({"value": 1, "unit": "h"}, 3600.0),
        ({"value": 1, "unit": "d"}, 86400.0),
    ],
)
def test_duration_units(raw, seconds):
    assert Duration.model_validate(raw).seconds == seconds


@pytest.mark.parametrize(
    "raw",
    [
        True,
        {"value": True},
        -1,
        float("inf"),
        float("nan"),
        {"value": 1e308, "unit": "d"},
        {"value": 1, "unit": "minute"},
    ],
)
def test_duration_invalid(raw):
    with pytest.raises(ValueError):
        Duration.model_validate(raw)


def test_duration_boundaries():
    assert (
        CacheProbeConfig.model_validate(
            {"mode": "exact_repeat", "delay": {"value": 500, "unit": "ms"}}
        ).delay.seconds
        == 0.5
    )
    assert (
        RequestStepSpec.model_validate(
            {"name": "turn", "input": "opening", "after": 2}
        ).after.seconds
        == 2
    )
    with pytest.raises(ValueError, match="60 seconds"):
        CacheProbeConfig.model_validate(
            {"mode": "exact_repeat", "delay": {"value": 2, "unit": "m"}}
        )
    plan = RunnerPlanRecurrence.model_validate(
        {"kind": "interval", "every": {"value": 500, "unit": "ms"}}
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert next_fire_at(
        "UTC", plan.model_dump(mode="json"), start, start
    ) == start + timedelta(milliseconds=500)


def test_dimension_duration():
    recipe = TaskRecipe.model_validate(
        {
            "name": "milliseconds",
            "instances": {"matrix": {"pause": [500]}},
            "payloads": {"p": {"seed": "p"}},
            "workflow": [
                {
                    "invoke": {
                        "name": "request",
                        "payload": "p",
                        "after": {"value": "$pause", "unit": "ms"},
                    }
                }
            ],
        }
    )
    invoke = recipe.workflow[0].invoke
    assert invoke is not None
    assert invoke.after.resolve({"pause": 500}) == 0.5


@pytest.mark.parametrize("field", ["after_seconds", "every_seconds"])
def test_old_task_fields(field):
    primitive = (
        {
            "repeat": {
                "name": "repeat",
                "count": 1,
                field: 1,
                "node": {"invoke": {"name": "request", "payload": "p"}},
            }
        }
        if field == "every_seconds"
        else {"invoke": {"name": "request", "payload": "p", field: 1}}
    )
    with pytest.raises(ValueError, match="Extra inputs"):
        TaskRecipe.model_validate(
            {"name": "old", "payloads": {"p": {"seed": "p"}}, "workflow": [primitive]}
        )


def test_old_config_fields():
    for old_field in ("timeout_seconds", "max_completed_requests"):
        with pytest.raises(ValueError, match="Extra inputs"):
            BenchmarkConfig.model_validate(
                {"provider": "test", "model": "test", old_field: 1}
            )
    with pytest.raises(ValueError, match="Extra inputs"):
        CacheProbeConfig.model_validate({"mode": "exact_repeat", "delay_seconds": 1})
    with pytest.raises(ValueError, match="Extra inputs"):
        RunnerPlanRecurrence.model_validate({"kind": "interval", "every_seconds": 1})
    with pytest.raises(ValueError, match="Extra inputs"):
        RequestStepSpec.model_validate(
            {"name": "turn", "input": "opening", "after_seconds": 1}
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        TaskRecipe.model_validate(
            {
                "name": "old",
                "payloads": {"p": {"seed_namespace": "p"}},
                "workflow": [{"invoke": {"name": "request", "payload": "p"}}],
            }
        )


def test_runner_expansion():
    base = {
        "provider": "test",
        "model": "test",
        "repeat": 4,
        "timeout": {"value": 500, "unit": "ms"},
    }
    benchmark = BenchmarkConfig.model_validate(base)
    assert benchmark.timeout.seconds == 0.5
    assert benchmark.repeat == 4
    with pytest.raises(ValueError, match="positive"):
        BenchmarkConfig.model_validate({**base, "timeout": 0})
    expanded = BenchmarkConfig.model_validate(
        {
            **base,
            "repeat": 4,
            "inputs": {"one": {"text": "one"}, "two": {"text": "two"}},
            "steps": [
                {"name": "first", "input": "one"},
                {"name": "second", "input": "two"},
            ],
        }
    )
    assert expanded.worker_count == 8
    with pytest.raises(ValueError, match="cache_probe defines its own requests"):
        BenchmarkConfig.model_validate(
            {
                **base,
                "cache_probe": {
                    "mode": "exact_repeat",
                    "trials": 2,
                    "repeats_after_prime": 2,
                },
            }
        )

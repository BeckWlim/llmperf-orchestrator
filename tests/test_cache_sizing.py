"""Cache budgeting and generated concurrency/replay guarantees."""

from dataclasses import replace

import pytest
import yaml

from llmperf_backend.compiler.tasks import TaskCompileContext, TaskCompiler
from llmperf_backend.configuration.models import BenchmarkCampaignStart
from llmperf_tools.cache_sizing import CacheSizing, build_campaigns, evaluate


def test_capacity_budget():
    settings = CacheSizing(l3_gib=1000, l4_gib=4916, kv_bytes_per_token=14336)
    report = evaluate(settings)
    assert settings.prefix_tokens == 52428
    assert settings.background_wait_seconds == 84320
    assert settings.probe_wait_seconds == 83400
    assert settings.probe_requests == 9472
    assert report["combined"] == {
        "maximum_concurrency": 56,
        "unused_slots": 8,
        "requests": 249088,
    }
    assert report["planning_window"] is None
    assert "uncalibrated" in str(report["warnings"])
    footprint = report["footprint_gib"]
    assert isinstance(footprint, dict)
    assert footprint["background_prefix_pool"] == pytest.approx(1075.18359375)
    assert footprint["probe_cohort"] == pytest.approx(179.197265625)


@pytest.mark.parametrize(
    "changes",
    [
        {"l3_gib": float("nan")},
        {"l4_low": 0.9},
        {"probe_lanes": 48},
        {"cohort_size": 31},
        {"background_repeats": 10001},
        {"recall_waves": 40},
        {"background_gap_seconds": 600.0},
        {"write_gib_per_second": 0.0},
        {"calibration_seconds": (float("inf"),)},
        {"probe_pool_seed": 202610211101},
        {"background_pool_seed": -1},
    ],
)
def test_invalid_budgets(changes):
    settings = CacheSizing(l3_gib=1000, l4_gib=4916, kv_bytes_per_token=14336)
    with pytest.raises(ValueError):
        replace(settings, **changes)


def test_rate_sensitivity():
    settings = CacheSizing(l3_gib=1000, l4_gib=4916, kv_bytes_per_token=14336)
    slow = evaluate(replace(settings, write_gib_per_second=0.1))
    fast = evaluate(replace(settings, write_gib_per_second=3.3))
    moderate = evaluate(replace(settings, write_gib_per_second=0.4))
    assert "L3 hits may dominate" in str(slow["warnings"])
    assert "disappear before recall" in str(fast["warnings"])
    assert moderate["warnings"] == []
    assert evaluate(replace(settings, l4_gib=1024))["warnings"]


@pytest.mark.parametrize("visits", [3, 101])
def test_generated_replay(visits):
    settings = CacheSizing(
        l3_gib=1000,
        l4_gib=4916,
        kv_bytes_per_token=14336,
        pool_size=8,
        background_concurrent_workers=2,
        background_repeats=visits,
        cohort_size=8,
        probe_lanes=2,
        calibration_seconds=(5.0, 10.0),
        recall_waves=2,
    )
    runner_template = {
        "provider": "dfs-beta",
        "model": "spark-x2.5-1.7b-L4",
        "inputs": {"default": {"tokens": 128}},
        "worker": {"repeat": 99},
        "concurrent_workers": 99,
    }
    background_document, probe_document = build_campaigns(settings, runner_template)
    assert runner_template["worker"] == {"repeat": 99}
    campaigns = [
        BenchmarkCampaignStart.model_validate(
            {
                key: value
                for key, value in yaml.safe_load(yaml.safe_dump(document)).items()
                if key != "wait"
            }
        )
        for document in (background_document, probe_document)
    ]
    for campaign, expected_requests, expected_peak in zip(
        campaigns, (8 * visits, 40), (8, 2)
    ):
        definition = campaign.task_definitions[0]
        benchmark = definition.runner.execution_document()["benchmark"]
        estimate = TaskCompiler.estimate(definition, benchmark=benchmark)
        assert estimate["requests"] == expected_requests
        assert estimate["peak_requests"] == expected_peak
        assert benchmark["repeat"] == (visits if campaign is campaigns[0] else 1)
    probe_definition = campaigns[1].task_definitions[0]
    context = TaskCompileContext(
        "probe",
        "probe",
        probe_definition,
        probe_definition.runner.execution_document(),
    )
    table = TaskCompiler(context).compile()
    first_seeds = [node.prefix_seed for node in table.nodes[:8]]
    assert len(set(first_seeds)) == 8
    for offset in range(8, 40, 8):
        assert [
            node.prefix_seed for node in table.nodes[offset : offset + 8]
        ] == first_seeds
    assert len({node.payload_seed for node in table.nodes}) == 40
    # Each lane is serial; every recall starts after both prior lane tails.
    assert table.nodes[1].dependencies == (table.nodes[0].node_id,)
    assert set(table.nodes[8].dependencies) == {
        table.nodes[3].node_id,
        table.nodes[7].node_id,
    }
    assert table.nodes[8].after_seconds == 5


def test_reused_token_budget(monkeypatch):
    from unittest.mock import Mock
    from llmperf_backend.compiler import tasks

    budget_counter = Mock(wraps=tasks.planned_token_budget)
    monkeypatch.setattr(tasks, "planned_token_budget", budget_counter)
    definition = {
        "name": "repeated-template",
        "instances": {"matrix": {"lane": [0, 1]}, "trials": 2},
        "workflow": [
            {
                "repeat": {
                    "name": "visits",
                    "count": 100,
                    "node": {"run": {"name": "request"}},
                }
            }
        ],
    }
    total = TaskCompiler.estimate_token_budget(definition, 100, 10)
    assert total == 400 * 110
    assert budget_counter.call_count == 1

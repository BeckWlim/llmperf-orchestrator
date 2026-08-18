import importlib.util
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".codex/skills/generate-llmperf-report/scripts/prepare_report_data.py"
SPEC = importlib.util.spec_from_file_location("llmperf_report_skill", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def task_runner(runner_id, node_id, role, ttft, hit=None, miss=None):
    cache = {}
    if hit is not None and miss is not None:
        cache = {
            "complete_hit_tokens": hit,
            "complete_miss_tokens": miss,
            "counter_coverage": 1.0,
            "weighted_token_hit_ratio": hit / (hit + miss),
        }
    context = {
        "definition_id": "definition-a",
        "instance_id": "instance-a",
        "node_id": node_id,
        "role": role,
        "payload_id": "cold" if role == "cold_control" else "replay",
        "payload_seed": 123 if role == "cold_control" else 456,
        "trial_index": 0,
        "dimensions": {"delay_seconds": 1800},
    }
    return {
        "runner_id": runner_id,
        "status": "succeeded",
        "created_at": "2026-01-01T00:00:00Z",
        "benchmark": {
            "provider": "provider-a",
            "model": "model-a",
            "mean_input_tokens": 100,
            "mean_output_tokens": 8,
            "concurrent_requests": 1,
            "task_request": context,
        },
        "summary": {
            "task_request": context,
            "outcome": {"status": "succeeded"},
            "results": {
                "num_requests_started": 1,
                "num_completed_requests": 1,
                "number_errors": 0,
                "error_rate": 0.0,
                "ttft_s": {"quantiles": {"p50": ttft, "p95": ttft}},
                "kv_cache": cache,
            },
        },
    }


def task_document():
    runners = [
        task_runner("prime-a", "prime", "prime", 10.0),
        task_runner("warm-a", "warm", "warm", 2.0, hit=80, miss=20),
        task_runner("cold-a", "cold", "cold_control", 4.0),
    ]
    return {
        "version": 6,
        "campaign": {"campaign_id": "campaign-a", "name": "retention-a"},
        "aggregate": {"status": "completed", "outcome": "succeeded"},
        "runner_plans": [],
        "task_definitions": [
            {
                "task_definition_id": "definition-a",
                "compiler": "task-graph/v1",
                "name": "retention",
            }
        ],
        "task_instances": [
            {
                "task_instance_id": "instance-a",
                "task_definition_id": "definition-a",
                "state": "completed",
                "spec": {"dimensions": {"delay_seconds": 1800}, "trial_index": 0},
                "checkpoint": {"payload_hashes": {"replay": "sha256:a"}},
            }
        ],
        "dispatches": [
            {
                "task_instance_id": "instance-a",
                "node_id": node_id,
                "runner_id": runner_id,
                "state": "emitted",
                "lineage": {
                    "role": role,
                    "payload_id": "cold" if role == "cold_control" else "replay",
                    "dependencies": dependencies,
                    "after_seconds": 1800 if node_id != "prime" else 0,
                },
            }
            for node_id, role, runner_id, dependencies in (
                ("prime", "prime", "prime-a", []),
                ("warm", "warm", "warm-a", ["prime-dispatch"]),
                ("cold", "cold_control", "cold-a", ["prime-dispatch"]),
            )
        ],
        "task_analyses": [],
        "runners": runners,
    }


def test_task_pipeline():
    analysis = REPORT.prepare_analysis(task_document(), "fixture")
    graph = analysis["evidence"]["task_graphs"][0]
    warm = next(node for node in graph["nodes"] if node["role"] == "warm")

    assert analysis["analysis_version"] == 2
    assert graph["dimensions"] == {"delay_seconds": 1800}
    assert graph["payload_hashes"] == {"replay": "sha256:a"}
    assert warm["runner"]["cache_hit_ratio"] == pytest.approx(0.8)
    assert warm["runner"]["ttft_p50"] == pytest.approx(2.0)
    assert json.loads(json.dumps(analysis))["source"]["version"] == 6


def test_analysis_cohorts():
    analysis = REPORT.prepare_analysis(task_document(), "fixture")

    assert analysis["overview"]["runner_count"] == 3
    assert len(analysis["cohorts"]) == 1
    assert analysis["cohorts"][0]["provider"] == "provider-a"
    assert analysis["cohorts"][0]["model"] == "model-a"
    theme = ROOT / ".codex/skills/generate-llmperf-report/assets/report-theme.css"
    assert ".chart-card.featured" in theme.read_text(encoding="utf-8")
    assert ".fold>summary" in theme.read_text(encoding="utf-8")


def test_series_palette_contract():
    theme = ROOT / ".codex/skills/generate-llmperf-report/assets/report-theme.css"
    palette_path = ROOT / ".codex/skills/generate-llmperf-report/assets/provider-palette.json"
    css = theme.read_text(encoding="utf-8")
    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    expected = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    actual = [
        re.search(rf"--series-{index}:(#[0-9A-Fa-f]{{6}})", css).group(1)
        for index in range(1, 5)
    ]

    assert palette["assignment"]["method"] == "sorted_identity_round_robin"
    assert [entry["color"] for entry in palette["series"][:4]] == expected
    assert actual == expected
    assert len(set(actual)) == len(actual)
    assert len({entry["marker"] for entry in palette["series"][:4]}) == 4


def test_missing_counters():
    document = task_document()
    warm = next(item for item in document["runners"] if item["runner_id"] == "warm-a")
    warm["summary"]["results"]["kv_cache"] = {}

    analysis = REPORT.prepare_analysis(document, "fixture")
    row = next(item for item in analysis["runners"] if item["runner_id"] == "warm-a")

    assert row["cache_hit_ratio"] is None
    assert row["cache_coverage"] is None


def test_runner_export():
    runner = task_runner("runner-a", "warm", "warm", 2.0, hit=80, miss=20)
    document = {
        "version": 1,
        "runner": {key: value for key, value in runner.items() if key != "summary"},
        "results": {"summary": runner["summary"], "request_count": 1, "requests": []},
    }

    analysis = REPORT.prepare_analysis(document, "fixture")

    assert analysis["source"]["version"] == 1
    assert analysis["overview"]["runner_count"] == 1
    assert analysis["runners"][0]["cache_hit_ratio"] == pytest.approx(0.8)

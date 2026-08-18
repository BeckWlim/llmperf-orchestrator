from datetime import datetime, timezone

import pytest

from llmperf_backend.models import BenchmarkConfig, TaskDefinitionCreate, dump_model
from llmperf_backend.task_compiler import (
    TaskCompileContext,
    compile_task_definition,
    estimate_task_definition,
)


def _runner_template():
    return {
        "label": "task",
        "metadata": {},
        "benchmark": {
            "provider": "test",
            "model": "cache-model",
            "llm_api": "openai",
            "mean_input_tokens": 4096,
            "stddev_input_tokens": 0,
            "mean_output_tokens": 16,
            "stddev_output_tokens": 0,
        },
    }


def _definition():
    model = TaskDefinitionCreate.model_validate(
        {
            "name": "repeat-dose",
            "matrix": {"warmup_count": [0, 2], "quiet_seconds": [60]},
            "trials": 1,
            "seed": 456,
            "payloads": {
                "replay": {"seed_namespace": "replay"},
                "cold": {"seed_namespace": "cold"},
            },
            "sequence": [
                {"kind": "invoke", "id": "prime", "role": "prime", "payload": "replay"},
                {
                    "kind": "repeat",
                    "id": "warmups",
                    "count": {"dimension": "warmup_count"},
                    "interval_seconds": 3,
                    "invoke": {
                        "kind": "invoke", "id": "warmup", "role": "warmup", "payload": "replay"
                    },
                },
                {
                    "kind": "parallel",
                    "after_seconds": {"dimension": "quiet_seconds"},
                    "invokes": [
                        {"kind": "invoke", "id": "probe", "role": "probe", "payload": "replay"},
                        {"kind": "invoke", "id": "cold", "role": "cold_control", "payload": "cold"},
                    ],
                },
            ],
            "runner": {},
        }
    )
    result = dump_model(model)
    result.pop("runner")
    return result


def test_graph_compilation():
    definition = _definition()
    instances = compile_task_definition(
        TaskCompileContext(
            definition_id="definition-1",
            definition_name="repeat-dose",
            definition=definition,
            runner_template=_runner_template(),
            database_now=datetime(2026, 8, 18, tzinfo=timezone.utc),
            created_by="test",
        )
    )

    assert estimate_task_definition(definition) == {"instances": 2, "nodes": 8}
    repeated = next(item for item in instances if item.dimensions["warmup_count"] == 2)
    assert [node.node_id for node in repeated.nodes] == [
        "prime", "warmups:1", "warmups:2", "probe", "cold"
    ]
    assert repeated.nodes[1].dependencies == [repeated.nodes[0].dispatch_id]
    assert repeated.nodes[2].dependencies == [repeated.nodes[1].dispatch_id]
    assert all(
        node.dependencies == [repeated.nodes[2].dispatch_id]
        for node in repeated.nodes[3:]
    )
    assert all(node.after_seconds == 60 for node in repeated.nodes[3:])
    assert all(
        BenchmarkConfig.model_validate(node.runner_template["benchmark"])
        for node in repeated.nodes
    )


def test_payload_replay():
    instances = compile_task_definition(
        TaskCompileContext(
            definition_id="definition-1",
            definition_name="repeat-dose",
            definition=_definition(),
            runner_template=_runner_template(),
            database_now=datetime(2026, 8, 18, tzinfo=timezone.utc),
            created_by="test",
        )
    )
    repeated = next(item for item in instances if item.dimensions["warmup_count"] == 2)
    seeds = {
        node.payload_id: node.runner_template["benchmark"]["task_request"]["payload_seed"]
        for node in repeated.nodes
    }
    replay_seeds = {
        node.runner_template["benchmark"]["task_request"]["payload_seed"]
        for node in repeated.nodes
        if node.payload_id == "replay"
    }
    assert len(replay_seeds) == 1
    assert seeds["cold"] not in replay_seeds


def test_task_span():
    payload = {
        "name": "too-long",
        "payloads": {"replay": {"seed_namespace": "replay"}},
        "sequence": [
            {
                "kind": "repeat",
                "id": "waits",
                "count": 7,
                "interval_seconds": 3600,
                "invoke": {
                    "kind": "invoke",
                    "id": "warm",
                    "role": "warm",
                    "payload": "replay",
                },
            }
        ],
        "runner": {},
    }

    with pytest.raises(ValueError, match="six hours"):
        TaskDefinitionCreate.model_validate(payload)

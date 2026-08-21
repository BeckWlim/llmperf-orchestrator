from itertools import count

import pytest

from llmperf_backend.app import _compile_campaign_task_preview
from llmperf_backend.models import (
    BenchmarkCampaignStart,
    CompiledBenchmarkConfig,
    TaskDefinitionCreate,
)
from llmperf_backend.task_compiler import (
    BaseNode,
    CompilationTable,
    ParallelNode,
    RepeatNode,
    TaskAssembler,
    TaskCompileContext,
    TaskCompiler,
)


def _runner_template():
    return {
        "label": "task",
        "metadata": {},
        "benchmark": {
            "provider": "test",
            "model": "cache-model",
            "adapter": "openai",
            "tokenizer": {
                "source": "huggingface",
                "id": "organization/tokenizer",
                "revision": "a" * 40,
                "requested_revision": "main",
                "use_fast": True,
                "immutable_revision": True,
                "selection": "explicit",
                "accuracy": "compatible",
            },
            "mean_input_tokens": 4096,
            "stddev_input_tokens": 0,
            "mean_output_tokens": 16,
            "stddev_output_tokens": 0,
        },
    }


def _definition():
    return TaskDefinitionCreate.model_validate(
        {
            "name": "repeat-dose",
            "instances": {
                "matrix": {"warmup_count": [0, 2], "quiet_seconds": [60]},
                "trials": 1,
                "seed": 456,
            },
            "payloads": {
                "replay": {"seed_namespace": "replay"},
                "cold": {"seed_namespace": "cold"},
            },
            "workflow": [
                {"invoke": {"name": "prime", "payload": "replay"}},
                {
                    "repeat": {
                        "name": "warmups",
                        "count": "$warmup_count",
                        "every_seconds": 3,
                        "node": {
                            "invoke": {
                                "name": "warmup",
                                "payload": "replay",
                            }
                        },
                    }
                },
                {
                    "parallel": {
                        "name": "observe",
                        "after_seconds": "$quiet_seconds",
                        "branches": [
                            {
                                "invoke": {
                                    "name": "probe",
                                    "payload": "replay",
                                }
                            },
                            {
                                "invoke": {
                                    "name": "cold",
                                    "role": "cold_control",
                                    "payload": "cold",
                                }
                            },
                        ],
                    }
                },
            ],
            "runner": {},
        }
    )


def _compiler(definition=None, runner_template=None):
    selected_definition = definition or _definition()
    return TaskCompiler(
        TaskCompileContext(
            definition_id="definition-1",
            definition_name=selected_definition.name,
            definition=selected_definition.recipe(),
            runner_template=runner_template or _runner_template(),
        )
    )


def test_graph_compilation():
    compiler = _compiler()
    table = compiler.compile()

    assert isinstance(table, CompilationTable)
    assert compiler.compile() is table
    assert TaskCompiler.estimate(_definition().recipe()) == {
        "instances": 2,
        "nodes": 8,
    }
    repeated_instance = next(
        instance
        for instance in table.instances
        if instance.dimensions["warmup_count"] == 2
    )
    repeated_nodes = table.nodes_for(repeated_instance.instance_key)
    assert [node.node_id for node in repeated_nodes] == [
        "prime",
        "warmups.1.warmup",
        "warmups.2.warmup",
        "observe.probe",
        "observe.cold",
    ]
    assert repeated_nodes[1].dependencies == ("prime",)
    assert repeated_nodes[2].dependencies == ("warmups.1.warmup",)
    assert all(
        node.dependencies == ("warmups.2.warmup",) for node in repeated_nodes[3:]
    )
    assert all(node.after_seconds == 60 for node in repeated_nodes[3:])


def test_node_hierarchy():
    compiler = _compiler()

    roots = compiler.graph.roots

    assert all(isinstance(node, BaseNode) for node in roots)
    assert isinstance(roots[1], RepeatNode)
    assert isinstance(roots[2], ParallelNode)


def test_model_ownership():
    assert TaskDefinitionCreate.__module__ == "llmperf_backend.task_compiler"


def test_table_assembly():
    compiler = _compiler()
    table = compiler.compile()
    identifier_sequence = count(1)

    def identifier_factory():
        return f"runtime-{next(identifier_sequence)}"

    assemblies = TaskAssembler(
        compiler.context,
        table,
        identifier_factory=identifier_factory,
    ).assemble()
    repeated = next(
        instance for instance in assemblies if instance.dimensions["warmup_count"] == 2
    )

    assert repeated.instance_id.startswith("runtime-")
    assert repeated.nodes[1].dependencies == (repeated.nodes[0].dispatch_id,)
    assert repeated.nodes[2].dependencies == (repeated.nodes[1].dispatch_id,)
    assert all(
        CompiledBenchmarkConfig.model_validate(node.runner_template["benchmark"])
        for node in repeated.nodes
    )


def test_payload_replay():
    table = _compiler().compile()
    repeated_instance = next(
        instance
        for instance in table.instances
        if instance.dimensions["warmup_count"] == 2
    )
    repeated_nodes = table.nodes_for(repeated_instance.instance_key)
    replay_seeds = {
        node.payload_seed for node in repeated_nodes if node.payload_id == "replay"
    }
    cold_seed = next(
        node.payload_seed for node in repeated_nodes if node.payload_id == "cold"
    )

    assert len(replay_seeds) == 1
    assert cold_seed not in replay_seeds


def test_dataset_replay():
    runner_template = _runner_template()
    runner_template["benchmark"].update(
        {
            "dataset": {
                "id": "organization/sharegpt",
                "filename": "sharegpt.json",
                "revision": "a" * 40,
                "adapter": "sharegpt",
            },
            "dataset_prompt_mode": "concatenate",
        }
    )
    compiler = _compiler(runner_template=runner_template)
    assemblies = TaskAssembler(compiler.context, compiler.compile()).assemble()
    replay_nodes = [node for node in assemblies[0].nodes if node.payload_id == "replay"]

    assert {
        node.runner_template["benchmark"]["dataset_seed"] for node in replay_nodes
    } == {replay_nodes[0].runner_template["benchmark"]["task_context"]["payload_seed"]}
    assert all(
        node.runner_template["benchmark"]["dataset_prompt_mode"] == "concatenate"
        for node in assemblies[0].nodes
    )


def test_nested_sequence():
    definition = TaskDefinitionCreate.model_validate(
        {
            "name": "nested",
            "payloads": {"replay": {"seed_namespace": "replay"}},
            "workflow": [
                {
                    "parallel": {
                        "name": "branches",
                        "branches": [
                            {
                                "sequence": {
                                    "name": "left",
                                    "steps": [
                                        {
                                            "invoke": {
                                                "name": "first",
                                                "payload": "replay",
                                            }
                                        },
                                        {
                                            "invoke": {
                                                "name": "second",
                                                "payload": "replay",
                                            }
                                        },
                                    ],
                                }
                            },
                            {
                                "invoke": {
                                    "name": "right",
                                    "payload": "replay",
                                }
                            },
                        ],
                    }
                },
                {"invoke": {"name": "join", "payload": "replay"}},
            ],
            "runner": {},
        }
    )

    table = _compiler(definition=definition).compile()
    nodes = table.nodes_for("trial=0")

    assert [node.node_id for node in nodes] == [
        "branches.left.first",
        "branches.left.second",
        "branches.right",
        "join",
    ]
    assert nodes[-1].dependencies == (
        "branches.left.second",
        "branches.right",
    )


def test_legacy_normalization():
    definition = TaskDefinitionCreate.model_validate(
        {
            "name": "legacy",
            "payloads": {"replay": {"seed_namespace": "replay"}},
            "sequence": [
                {
                    "kind": "invoke",
                    "id": "prime",
                    "role": "prime",
                    "payload": "replay",
                }
            ],
            "runner": {},
        }
    )
    normalized_document = definition.model_dump(mode="json")

    assert "workflow" in normalized_document
    assert "sequence" not in normalized_document
    assert normalized_document["instances"] == {
        "matrix": {},
        "trials": 1,
        "seed": 11111,
    }
    assert normalized_document["workflow"][0]["invoke"]["name"] == "prime"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("matrix", {"delay": [1]}),
        ("trials", 2),
        ("seed", 456),
    ),
)
def test_flat_expansion_rejected(field, value):
    payload = {
        "name": "flat-expansion",
        "payloads": {"replay": {"seed_namespace": "replay"}},
        "workflow": [{"invoke": {"name": "probe", "payload": "replay"}}],
        "runner": {},
        field: value,
    }

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TaskDefinitionCreate.model_validate(payload)


def test_preview_compilation_debug():
    campaign = BenchmarkCampaignStart.model_validate(
        {
            "campaign": {"name": "preview-study"},
            "task_definitions": [
                {
                    "name": "delay-surface",
                    "instances": {
                        "matrix": {"delay": [5, 10]},
                        "trials": 2,
                        "seed": 7,
                    },
                    "payloads": {"replay": {"seed_namespace": "replay"}},
                    "workflow": [
                        {"invoke": {"name": "prime", "payload": "replay"}},
                        {
                            "invoke": {
                                "name": "probe",
                                "payload": "replay",
                                "after_seconds": "$delay",
                            }
                        },
                    ],
                    "runner": {},
                }
            ],
        }
    )

    preview = _compile_campaign_task_preview(campaign, 2, 10, True)
    compact_preview = _compile_campaign_task_preview(campaign, 1, 1, False)

    assert preview["summary"]["task_instances"] == 4
    assert preview["summary"]["task_nodes"] == 8
    definition = preview["task_definitions"][0]
    assert definition["shown_instance_count"] == 2
    assert definition["truncated_instance_count"] == 2
    assert definition["instances"][0]["nodes"][1]["dependencies"] == ["prime"]
    assert definition["instances"][0]["nodes"][1]["after_seconds"] == 5
    assert "payload_seed" in definition["instances"][0]["nodes"][0]
    compact_node = compact_preview["task_definitions"][0]["instances"][0]["nodes"][0]
    assert "payload_seed" not in compact_node
    compact_instance = compact_preview["task_definitions"][0]["instances"][0]
    assert compact_instance["truncated_node_count"] == 1


def test_invalid_reference():
    payload = {
        "name": "invalid-reference",
        "instances": {"matrix": {"known": [1]}},
        "payloads": {"replay": {"seed_namespace": "replay"}},
        "workflow": [
            {
                "invoke": {
                    "name": "probe",
                    "payload": "replay",
                    "after_seconds": "$missing",
                }
            }
        ],
        "runner": {},
    }

    with pytest.raises(ValueError, match="unknown dimension missing"):
        TaskDefinitionCreate.model_validate(payload)


def test_invalid_primitive():
    payload = {
        "name": "invalid-primitive",
        "payloads": {"replay": {"seed_namespace": "replay"}},
        "workflow": [
            {
                "invoke": {"name": "probe", "payload": "replay"},
                "parallel": {
                    "name": "also-parallel",
                    "branches": [{"invoke": {"name": "cold", "payload": "replay"}}],
                },
            }
        ],
        "runner": {},
    }

    with pytest.raises(ValueError, match="exactly one primitive"):
        TaskDefinitionCreate.model_validate(payload)


def test_task_span_boundary():
    boundary_payload = {
        "name": "boundary",
        "payloads": {"replay": {"seed_namespace": "replay"}},
        "workflow": [
            {
                "invoke": {
                    "name": "probe",
                    "payload": "replay",
                    "after_seconds": 86_400,
                }
            }
        ],
        "runner": {},
    }

    definition = TaskDefinitionCreate.model_validate(boundary_payload)

    assert definition.workflow[0].invoke is not None
    assert definition.workflow[0].invoke.after_seconds == 86_400


def test_task_span_limit():
    excessive_payload = {
        "name": "too-long",
        "payloads": {"replay": {"seed_namespace": "replay"}},
        "workflow": [
            {
                "invoke": {
                    "name": "probe",
                    "payload": "replay",
                    "after_seconds": 86_401,
                }
            }
        ],
        "runner": {},
    }

    with pytest.raises(ValueError, match="24 hours"):
        TaskDefinitionCreate.model_validate(excessive_payload)

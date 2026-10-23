"""Exact cohort enumeration, bounded lanes, lexical binding and wave barriers."""

from copy import deepcopy

import pytest

from llmperf_backend.compiler.tasks import (
    GeneratedSeedPool,
    TaskCompileContext,
    TaskCompiler,
    TaskRecipe,
)


def cohort_document(values=None, parallelism=2):
    return {
        "name": "cohort-test",
        "instances": {"seed": 123},
        "workflow": [
            {
                "repeat": {
                    "name": "waves",
                    "count": 3,
                    "every": 10,
                    "node": {
                        "for_each": {
                            "name": "cohort",
                            "variable": "prefix",
                            "values": (
                                [11, 22, 33, 44, 55] if values is None else values
                            ),
                            "parallelism": parallelism,
                            "node": {
                                "run": {"name": "request", "prefix_seed": "$prefix"}
                            },
                        }
                    },
                }
            }
        ],
    }


def compile_document(document):
    recipe = TaskRecipe.model_validate(document)
    context = TaskCompileContext(
        "id", recipe.name, recipe, {"benchmark": {"repeat": 1}}
    )
    return TaskCompiler(context).compile()


def test_cohort_barriers():
    document = cohort_document()
    table = compile_document(document)
    assert len(table.nodes) == 15
    assert len({node.payload_seed for node in table.nodes}) == 15
    for wave_index in range(3):
        wave_nodes = table.nodes[wave_index * 5 : (wave_index + 1) * 5]
        assert [node.prefix_seed for node in wave_nodes] == [11, 22, 33, 44, 55]
        incoming = (
            (
                table.nodes[wave_index * 5 - 3].node_id,
                table.nodes[wave_index * 5 - 1].node_id,
            )
            if wave_index
            else ()
        )
        for head_index in (0, 3):
            assert wave_nodes[head_index].dependencies == incoming
            assert wave_nodes[head_index].after_seconds == 10
        for member_index in (1, 2, 4):
            assert wave_nodes[member_index].dependencies == (
                wave_nodes[member_index - 1].node_id,
            )
            assert wave_nodes[member_index].after_seconds == 0
    estimate = TaskCompiler.estimate(document, benchmark={"repeat": 1})
    assert estimate["peak_requests"] == estimate["peak_runners"] == 2
    assert estimate["requests"] == 15
    assert TaskCompiler.estimate_token_budget(document, 100, 10) == 1650
    restored = TaskRecipe.model_validate(document).model_dump(mode="json")
    assert compile_document(restored).nodes == table.nodes


def test_generated_equivalence():
    generated_document = cohort_document({"seed": 202610211102, "count": 256}, 32)
    explicit_document = cohort_document(
        list(GeneratedSeedPool(seed=202610211102, count=256).values), 32
    )
    assert (
        compile_document(generated_document).nodes
        == compile_document(explicit_document).nodes
    )
    assert TaskCompiler.estimate(generated_document)["peak_requests"] == 32
    assert TaskCompiler.estimate(cohort_document([1, 2], 8))["peak_requests"] == 2


def test_explicit_lowering():
    document = cohort_document()
    expanded_document = deepcopy(document)
    repeat_config = expanded_document["workflow"][0]["repeat"]
    repeat_config["node"] = {
        "parallel": {
            "name": "cohort",
            "branches": [
                {
                    "sequence": {
                        "name": f"lane_{lane_index:02d}",
                        "steps": [
                            {
                                "sequence": {
                                    "name": f"member_{member_index:02d}",
                                    "steps": [
                                        {
                                            "run": {
                                                "name": "request",
                                                "prefix_seed": seed,
                                            }
                                        }
                                    ],
                                }
                            }
                            for member_index, seed in enumerate(seeds, 1)
                        ],
                    }
                }
                for lane_index, seeds in enumerate(([11, 22, 33], [44, 55]), 1)
            ],
        }
    }
    assert compile_document(document).nodes == compile_document(expanded_document).nodes
    assert TaskCompiler.estimate(document) == TaskCompiler.estimate(expanded_document)


def test_scoped_bindings():
    document = cohort_document([2, 3])
    document["instances"]["matrix"] = {"delay": [1, 2]}
    loop_config = document["workflow"][0]["repeat"]["node"]["for_each"]
    loop_config["node"] = {
        "repeat": {
            "name": "copies",
            "count": "$prefix",
            "every": "$delay",
            "node": {"run": {"name": "request", "prefix_seed": "$prefix"}},
        }
    }
    table = compile_document(document)
    assert len(table.nodes) == 30
    assert {node.prefix_seed for node in table.nodes} == {2, 3}
    nested_document = cohort_document([2, 3])
    outer_config = nested_document["workflow"][0]["repeat"]["node"]["for_each"]
    outer_config["node"] = {
        "for_each": {
            "name": "inner",
            "variable": "member",
            "values": [8, 9],
            "parallelism": 2,
            "node": {
                "run": {"name": "request", "prefix_seed": "$prefix", "after": "$member"}
            },
        }
    }
    nested_table = compile_document(nested_document)
    assert len(nested_table.nodes) == 12
    assert TaskCompiler.estimate(nested_document)["peak_requests"] == 4
    outer_config["node"]["for_each"]["variable"] = "prefix"
    with pytest.raises(ValueError, match="shadows"):
        compile_document(nested_document)


@pytest.mark.parametrize(
    "values", [[], [1, 1], [True], [-1], [2**64], {"seed": 1, "count": 0}]
)
def test_invalid_values(values):
    with pytest.raises(ValueError):
        compile_document(cohort_document(values))


@pytest.mark.parametrize("parallelism", [0, -1, True, "2", 10001])
def test_invalid_parallelism(parallelism):
    with pytest.raises(ValueError):
        compile_document(cohort_document(parallelism=parallelism))


def test_binding_boundaries():
    collision_document = cohort_document()
    collision_document["instances"]["matrix"] = {"prefix": [1]}
    with pytest.raises(ValueError, match="shadow a matrix"):
        compile_document(collision_document)
    escaped_document = cohort_document()
    escaped_document["workflow"].append(
        {"run": {"name": "outside", "prefix_seed": "$prefix"}}
    )
    with pytest.raises(ValueError, match="unknown dimension prefix"):
        compile_document(escaped_document)
    oversized_document = cohort_document({"seed": 1, "count": 4000}, 32)
    with pytest.raises(ValueError, match="cannot exceed 10000 nodes"):
        compile_document(oversized_document)

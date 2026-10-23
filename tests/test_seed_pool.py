"""Authoring expansion, deterministic selection, and direct prefix generation."""

from copy import deepcopy
import json

import pytest

from llmperf.datasets.tokenizer import TOKENIZER_PATH
from llmperf_backend.api.app import _compile_campaign_task_preview
from llmperf_backend.compiler.tasks import (
    GeneratedSeedPool,
    TaskAssembler,
    TaskCompileContext,
    TaskCompiler,
    TaskRecipe,
)
from llmperf_backend.configuration.models import BenchmarkCampaignStart
from llmperf_backend.runtime.preparation import prepare_named_inputs
from llmperf_backend.runtime.worker import INPUT_DATASET_PATHS
from llmperf_backend.runtime.worker_input import WorkerInputPreparer, worker_steps


def recipe_document(prefix_seed=None, runners: int | str = 4, rounds: int = 3):
    return {
        "name": "pool",
        "instances": {"seed": 20261021},
        "workflow": [
            {
                "repeat": {
                    "name": "rounds",
                    "count": rounds,
                    "node": {
                        "parallel": {
                            "name": "members",
                            "count": runners,
                            "node": {
                                "run": {
                                    "name": "batch",
                                    "prefix_seed": (
                                        {"pick": [0, 17, 2**64 - 1]}
                                        if prefix_seed is None
                                        else prefix_seed
                                    ),
                                }
                            },
                        }
                    },
                }
            }
        ],
    }


def compiler_context(document):
    recipe = TaskRecipe.model_validate(document)
    return TaskCompileContext(
        "definition",
        recipe.name,
        recipe,
        {
            "benchmark": {
                "provider": "test",
                "model": "test",
                "adapter": "openai",
                "tokenizer": {
                    "id": "test",
                    "revision": "a" * 40,
                    "requested_revision": "a" * 40,
                    "immutable_revision": True,
                    "selection": "explicit",
                    "accuracy": "compatible",
                },
                "repeat": 4,
                "concurrent_workers": 2,
                "mean_input_tokens": 200,
                "stddev_input_tokens": 0,
                "shared_prefix_tokens": 160,
                "mean_output_tokens": 2,
                "stddev_output_tokens": 0,
            }
        },
    )


def test_group_expansion():
    document = recipe_document()
    context = compiler_context(document)
    table = TaskCompiler(context).compile()
    assert len(table.nodes) == 12
    for round_index in range(3):
        members = table.nodes[round_index * 4 : (round_index + 1) * 4]
        previous_ids = (
            tuple(
                node.node_id
                for node in table.nodes[(round_index - 1) * 4 : round_index * 4]
            )
            if round_index
            else ()
        )
        assert all(node.dependencies == previous_ids for node in members)
    expanded_document = deepcopy(document)
    parallel = expanded_document["workflow"][0]["repeat"]["node"]["parallel"]
    template = parallel.pop("node")
    parallel.pop("count")
    parallel["branches"] = [
        {"sequence": {"name": str(index), "steps": [deepcopy(template)]}}
        for index in range(1, 5)
    ]
    assert (
        table.nodes == TaskCompiler(compiler_context(expanded_document)).compile().nodes
    )
    estimate = TaskCompiler.estimate(
        context.definition, benchmark=context.runner_template["benchmark"]
    )
    assert estimate["runners"] == 12
    assert estimate["requests"] == 48
    assert estimate["peak_runners"] == 4
    assert estimate["peak_requests"] == 8
    assert (
        TaskCompiler.estimate_token_budget(
            document, 200, 2, context.runner_template["benchmark"]
        )
        == 48 * 202
    )


def test_seed_selection():
    document = recipe_document(runners=8)
    document["instances"].update(matrix={"lane": [1, 2]}, trials=2)
    context = compiler_context(document)
    table = TaskCompiler(context).compile()
    restored_recipe = TaskRecipe.model_validate_json(
        context.definition.model_dump_json()
    )
    restored_context = TaskCompileContext(
        "other-uuid", "other-name", restored_recipe, context.runner_template
    )
    assert table.nodes == TaskCompiler(restored_context).compile().nodes
    assert {node.prefix_seed for node in table.nodes} == {0, 17, 2**64 - 1}
    assert len({node.payload_seed for node in table.nodes}) == len(table.nodes)
    assert (
        len(
            {
                tuple(node.prefix_seed for node in table.nodes_for(row.instance_key))
                for row in table.instances
            }
        )
        == 4
    )
    changed_document = deepcopy(document)
    changed_document["instances"]["seed"] += 1
    changed_table = TaskCompiler(compiler_context(changed_document)).compile()
    assert [node.prefix_seed for node in changed_table.nodes] != [
        node.prefix_seed for node in table.nodes
    ]
    for assembly in TaskAssembler(context, table).assemble():
        for assembled, compiled in zip(
            assembly.nodes, table.nodes_for(assembly.instance_key)
        ):
            benchmark = assembled.runner_template["benchmark"]
            assert benchmark["shared_prefix_seed"] == compiled.prefix_seed
            assert (
                benchmark["task_context"]["shared_prefix_seed"] == compiled.prefix_seed
            )
            assert (
                benchmark["task_context"]["prefix_seed_version"] == "explicit-prefix-v1"
            )


def test_seed_reference():
    document = recipe_document("$family")
    document["instances"].update(
        matrix={"family": [0, 2**64 - 1]}, prefix_seed_scope="family"
    )
    context = compiler_context(document)
    table = TaskCompiler(context).compile()
    for instance in table.instances:
        assert {
            node.prefix_seed for node in table.nodes_for(instance.instance_key)
        } == {instance.dimensions["family"]}
    for assembly in TaskAssembler(context, table).assemble():
        assert all(
            node.runner_template["benchmark"]["shared_prefix_seed"]
            == assembly.dimensions["family"]
            for node in assembly.nodes
        )


def test_generated_pool():
    pool = GeneratedSeedPool(seed=0, count=10_000)
    assert len(pool.values) == len(set(pool.values)) == 10_000
    assert all(0 <= value < 2**64 for value in pool.values)
    assert pool.values is pool.values
    assert GeneratedSeedPool(seed=0, count=4).values == pool.values[:4]
    assert GeneratedSeedPool(seed=2**64 - 1, count=4).values != pool.values[:4]
    assert pool.model_dump() == {"seed": 0, "count": 10_000}
    document = recipe_document({"pick": {"seed": 20261021, "count": 8}})
    document["instances"]["trials"] = 2
    generated_context = compiler_context(document)
    generated_table = TaskCompiler(generated_context).compile()
    explicit_document = recipe_document(
        {"pick": list(GeneratedSeedPool(seed=20261021, count=8).values)}
    )
    explicit_document["instances"]["trials"] = 2
    assert (
        generated_table.nodes
        == TaskCompiler(compiler_context(explicit_document)).compile().nodes
    )
    restored = TaskRecipe.model_validate_json(
        generated_context.definition.model_dump_json()
    )
    restored_context = TaskCompileContext(
        "restored", restored.name, restored, generated_context.runner_template
    )
    assert TaskCompiler(restored_context).compile().nodes == generated_table.nodes


def test_pool_collision(monkeypatch):
    class Digest:
        def digest(self):
            return b"\xff" * 32

    monkeypatch.setattr(
        "llmperf_backend.compiler.tasks.hashlib.sha256", lambda _: Digest()
    )
    assert GeneratedSeedPool(seed=17, count=3).values == (2**64 - 1, 0, 1)


@pytest.mark.parametrize(
    "pool",
    [
        {},
        {"seed": 1},
        {"count": 4},
        {"seed": -1, "count": 4},
        {"seed": 2**64, "count": 4},
        {"seed": True, "count": 4},
        {"seed": "1", "count": 4},
        {"seed": 1, "count": 0},
        {"seed": 1, "count": 10_001},
        {"seed": 1, "count": True},
        {"seed": 1, "count": 1.5},
        {"seed": 1, "count": "4"},
        {"seed": 1, "count": 4, "unknown": 2},
    ],
)
def test_invalid_pool(pool):
    with pytest.raises(ValueError):
        compiler_context(recipe_document({"pick": pool}))


@pytest.mark.parametrize(
    "prefix_seed",
    [
        -1,
        2**64,
        True,
        1.5,
        "17",
        {"pick": []},
        {"pick": [1, 1]},
        {"pick": [-1]},
        {"pick": [2**64]},
        {"pick": [True]},
        {"pick": [1.5]},
        {"pick": list(range(10_001))},
        {"pick": [1], "unknown": 2},
        "$missing",
    ],
)
def test_invalid_seed(prefix_seed):
    with pytest.raises(ValueError):
        compiler_context(recipe_document(prefix_seed))


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": 0},
        {"count": 10_001},
        {"count": "$missing"},
        {"count": None},
        {"node": None},
        {"branches": [{"run": {"name": "conflict"}}]},
    ],
)
def test_invalid_group(overrides):
    document = recipe_document()
    document["workflow"][0]["repeat"]["node"]["parallel"].update(overrides)
    with pytest.raises(ValueError):
        compiler_context(document)


def test_expansion_limits():
    document = recipe_document(runners="$size", rounds=1)
    document["instances"]["matrix"] = {"size": [512]}
    assert len(TaskCompiler(compiler_context(document)).compile().nodes) == 512
    with pytest.raises(ValueError, match="cannot exceed 10000 nodes"):
        compiler_context(recipe_document(runners=5001, rounds=2))
    invalid_reference = recipe_document("$family")
    invalid_reference["instances"]["matrix"] = {"family": [-1]}
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        compiler_context(invalid_reference)


@pytest.mark.parametrize("calibrated", [False, True])
def test_direct_prompts(tmp_path, monkeypatch, calibrated):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode("ascii"))

    tokenizer = Tokenizer()
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "indexes"))
    monkeypatch.setattr(
        "llmperf_backend.runtime.worker_input._prepared_tokenizer",
        lambda *args: tokenizer,
    )
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": f"record-{index:06d}"}]}
                for index in range(1000)
            ]
        )
    )
    document = recipe_document(0, runners=1, rounds=2)
    document["instances"]["trials"] = 2
    context = compiler_context(document)
    input_definition = {
        "tokens": 200,
        "shared_prefix_tokens": 160,
        "dataset_prompt_mode": "concatenate",
        "dataset": {
            "source": "huggingface",
            "id": "test/corpus",
            "adapter": "sharegpt-user",
            "filename": corpus_path.name,
        },
    }
    context.runner_template["benchmark"].update(
        repeat=1,
        inputs={name: input_definition for name in ("default", "other")},
        steps=[{"name": "a", "input": "default"}, {"name": "b", "input": "other"}],
    )
    environment = {
        TOKENIZER_PATH: "unused",
        INPUT_DATASET_PATHS: json.dumps(
            {name: str(corpus_path) for name in ("default", "other")}
        ),
    }
    prefix_hashes = set()
    suffixes = set()
    for assembly in TaskAssembler(context, TaskCompiler(context).compile()).assemble():
        for node in assembly.nodes:
            benchmark = node.runner_template["benchmark"]
            preparer = WorkerInputPreparer()  # Different actor/cache for each round.
            steps = worker_steps(benchmark, calibrated=calibrated)
            for step in steps:
                assert step.request.preparation is not None
                assert step.request.preparation.prefix_seed == 0
                prepared = preparer.prepare(step.request, environment)
                assert prepared.metadata is not None
                prefix_hashes.add(prepared.metadata["prompt_evidence"]["prefix_hash"])
                suffixes.add(prepared.prompt[0][160:])
            named_prompts = prepare_named_inputs(benchmark, environment, tokenizer)
            assert named_prompts is not None
            assert len({evidence["prefix_hash"] for _, evidence in named_prompts}) == 1
    assert len(prefix_hashes) == 1
    assert len(suffixes) == 8


def test_preview_seeds():
    definition = recipe_document()
    definition["runner"] = {
        "provider": "test",
        "model": "test",
        "worker": {"repeat": 4},
    }
    campaign = BenchmarkCampaignStart.model_validate(
        {"campaign": {"name": "pool"}, "task_definitions": [definition]}
    )
    preview = _compile_campaign_task_preview(campaign, 1, 12, True)
    compact = _compile_campaign_task_preview(campaign, 1, 12, False)
    nodes = preview["task_definitions"][0]["instances"][0]["nodes"]
    assert {node["prefix_seed"] for node in nodes} <= {0, 17, 2**64 - 1}
    assert all("dataset_seed" in node for node in nodes)
    assert all(
        "prefix_seed" not in node
        for node in compact["task_definitions"][0]["instances"][0]["nodes"]
    )

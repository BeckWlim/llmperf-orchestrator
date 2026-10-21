"""Stable family prefixes, family-derived trial suffixes and durable provenance."""

import asyncio
from collections import defaultdict
import json
import hashlib

import pytest

from llmperf.datasets.preparation import calibrate, render_chunk
from llmperf.datasets.prompts import PromptDatasetSource, prepare_prompt_requests
from llmperf_backend.compiler.tasks import (
    TaskAssembler,
    TaskCompileContext,
    TaskCompiler,
    TaskRecipe,
)
from llmperf_backend.configuration.models import BatchBenchmarkConfig, DatabaseConfig
from llmperf_backend.runtime.coordinator import RuntimeCoordinator
from llmperf_backend.runtime.preparation import prepare_named_inputs
from llmperf_backend.runtime.worker import INPUT_DATASET_PATHS
from llmperf_backend.storage.repository import Base, Database, RunnerRepository


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("ascii"))


def family_recipe(families=100, trials=10, **options):
    return TaskRecipe.model_validate(
        {
            "name": "family-recall",
            "instances": {
                "matrix": {"family": list(range(families))},
                "trials": trials,
                "seed": 12345,
                "prefix_seed_scope": "family",
                **options,
            },
            "workflow": [{"run": {"name": "batch"}}],
        }
    )


def runner_template():
    return {
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
            "mean_input_tokens": 200,
            "stddev_input_tokens": 0,
            "shared_prefix_tokens": 160,
            "mean_output_tokens": 2,
            "stddev_output_tokens": 0,
        }
    }


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "indexes"))
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": f"record-{index:06d}"}]}
                for index in range(1000)
            ]
        )
    )
    return path


def test_family_compilation():
    recipe = family_recipe()
    context = TaskCompileContext("definition", recipe.name, recipe, runner_template())
    table = TaskCompiler(context).compile()
    assert table.instances == TaskCompiler(context).compile().instances
    assert len(table.instances) == 1000
    assert len({node.payload_seed for node in table.nodes}) == 1000
    assert all(not node.dependencies for node in table.nodes)
    assert {row.dimensions["family"] for row in table.instances[:10]} == {0}
    assert [row.trial_index for row in table.instances[:10]] == list(range(10))
    prefix_seeds = defaultdict(set)
    for assembly in TaskAssembler(context, table).assemble():
        benchmark = BatchBenchmarkConfig.model_validate(
            assembly.nodes[0].runner_template["benchmark"]
        )
        prefix_seeds[assembly.dimensions["family"]].add(benchmark.shared_prefix_seed)
        assert benchmark.shared_prefix_seed == benchmark.task_context.shared_prefix_seed
        assert benchmark.task_context.prefix_seed_version == "family-prefix-v1"
        assert benchmark.task_context.seed_version == "family-suffix-v1"
        suffix_material = (
            f"family-suffix-v1\0{benchmark.shared_prefix_seed}"
            f"\0{assembly.trial_index}\0batch"
        )
        assert benchmark.dataset_seed == int.from_bytes(
            hashlib.sha256(suffix_material.encode()).digest()[:8], "big"
        )
    assert all(len(seeds) == 1 for seeds in prefix_seeds.values())
    assert len(set.union(*prefix_seeds.values())) == 100
    legacy = family_recipe(prefix_seed_scope="trial")
    legacy_context = TaskCompileContext(
        "definition", legacy.name, legacy, runner_template()
    )
    legacy_table = TaskCompiler(legacy_context).compile()
    assert {row.dimensions["family"] for row in legacy_table.instances[:10]} == {0}
    assert [row.trial_index for row in legacy_table.instances[:10]] == list(range(10))
    assert legacy_table.nodes[0].payload_seed == int.from_bytes(
        hashlib.sha256(
            "\0".join(("batch-v1", "12345", "family=0", "0", "batch")).encode()
        ).digest()[:8],
        "big",
    )
    assert all(
        node.runner_template["benchmark"]["shared_prefix_seed"] is None
        for assembly in TaskAssembler(legacy_context, legacy_table).assemble()
        for node in assembly.nodes
    )
    with pytest.raises(ValueError, match="run-only"):
        TaskRecipe.model_validate(
            {
                "name": "invalid",
                "instances": {"prefix_seed_scope": "family"},
                "payloads": {"p": {"seed": "p"}},
                "workflow": [{"invoke": {"name": "call", "payload": "p"}}],
            }
        )


@pytest.mark.parametrize("mode", ["exact", "calibrated", "named"])
def test_trial_prompts(corpus, mode):
    recipe = family_recipe(2, 2)
    context = TaskCompileContext("definition", recipe.name, recipe, runner_template())
    assemblies = TaskAssembler(context, TaskCompiler(context).compile()).assemble()
    source = PromptDatasetSource.external("sharegpt-user", corpus)
    prompts_by_family = defaultdict(list)
    prefix_hashes = defaultdict(set)
    tokenizer = CharacterTokenizer()
    for assembly in assemblies:
        benchmark = assembly.nodes[0].runner_template["benchmark"]
        if mode == "calibrated":
            plan = calibrate(
                tokenizer,
                benchmark["dataset_seed"],
                200,
                0,
                160,
                2,
                source=source,
                shared_prefix_seed=benchmark["shared_prefix_seed"],
            )
            prepared = render_chunk(plan, 0, 4)
            assert prepared == render_chunk(plan, 0, 2) + render_chunk(plan, 2, 2)
        elif mode == "exact":
            prompts, evidence = prepare_prompt_requests(
                source,
                "concatenate",
                4,
                1,
                200,
                0,
                2,
                len,
                benchmark["dataset_seed"],
                160,
                benchmark["shared_prefix_seed"],
            )
            prepared = list(zip(prompts, evidence))
        else:
            named_benchmark = {
                **benchmark,
                "repeat": 1,
                "steps": [
                    {"name": f"s{index}", "input": "family"} for index in range(4)
                ],
                "inputs": {
                    "family": {
                        "tokens": 200,
                        "shared_prefix_tokens": 160,
                        "dataset_seed": 777,
                        "dataset_prompt_mode": "concatenate",
                        "dataset": {
                            "source": "huggingface",
                            "id": "test/corpus",
                            "filename": "sharegpt.json",
                            "adapter": "sharegpt-user",
                        },
                    }
                },
            }
            environment = {INPUT_DATASET_PATHS: json.dumps({"family": str(corpus)})}
            named_prepared = prepare_named_inputs(
                named_benchmark, environment, tokenizer
            )
            assert named_prepared is not None
            assert named_prepared == prepare_named_inputs(
                named_benchmark, environment, tokenizer
            )
            prepared = list(named_prepared)
        family = assembly.dimensions["family"]
        for (prompt, token_count), evidence_item in prepared:
            assert token_count == len(prompt) == 200
            prompts_by_family[family].append(prompt)
            prefix_hashes[family].add(evidence_item["prefix_hash"])
    for family_prompts in prompts_by_family.values():
        assert len(family_prompts) == 8
        assert len({prompt[:160] for prompt in family_prompts}) == 1
        assert len({prompt[160:] for prompt in family_prompts}) == 8
    assert all(len(hashes) == 1 for hashes in prefix_hashes.values())
    assert prefix_hashes[0] != prefix_hashes[1]


@pytest.mark.postgresql
def test_seed_persistence(postgresql_url):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("family-seed-test")
            recipe = family_recipe(5, 3)
            template = runner_template()
            await repository.create_campaign_workload(
                "families",
                None,
                {},
                [],
                [],
                [
                    {
                        "definition": recipe.model_dump(mode="json"),
                        "runner_template": template,
                    }
                ],
                "test",
            )
            context = TaskCompileContext("definition", recipe.name, recipe, template)
            expected = {
                (
                    assembly.dimensions["family"],
                    assembly.trial_index,
                    assembly.nodes[0].runner_template["benchmark"][
                        "shared_prefix_seed"
                    ],
                    assembly.nodes[0].runner_template["benchmark"]["dataset_seed"],
                )
                for assembly in TaskAssembler(
                    context, TaskCompiler(context).compile()
                ).assemble()
            }
            coordinator = RuntimeCoordinator(repository)
            await coordinator.reconcile()
            # Rebuild the transient queue from durable records before claiming.
            recovered = RuntimeCoordinator(repository)
            await recovered.reconcile()
            observed = set()
            for _ in expected:
                runner = await asyncio.wait_for(recovered.next_runner("test"), 2)
                task_context = runner["benchmark"]["task_context"]
                observed.add(
                    (
                        task_context["dimensions"]["family"],
                        task_context["trial_index"],
                        task_context["shared_prefix_seed"],
                        task_context["dataset_seed"],
                    )
                )
            assert observed == expected
        finally:
            await database.engine.dispose()

    asyncio.run(exercise())

"""Batch task compilation, request execution, and durable family sequencing."""

import asyncio

import pytest

from llmperf.metrics import constants as common_metrics
from llmperf.benchmark import throughput as benchmark_module
from llmperf.datasets.prompts import PromptDatasetSource
from llmperf_backend.configuration.models import (
    BatchBenchmarkConfig,
    DatabaseConfig,
    PerformanceGuardConfig,
)
from llmperf_backend.compiler.safety import assess_workload
from llmperf_backend.storage.repository import Base, Database, RunnerRepository
from llmperf_backend.runtime.coordinator import RuntimeCoordinator
from llmperf_backend.compiler.tasks import (
    TaskAssembler,
    TaskCompileContext,
    TaskCompiler,
    TaskRecipe,
)


def batch_recipe(families=8, rounds=100):
    return TaskRecipe.model_validate(
        {
            "name": "families",
            "instances": {
                "matrix": {"family": list(range(1, families + 1))},
                "seed": 20261014174544,
            },
            "workflow": [
                {
                    "repeat": {
                        "name": "rounds",
                        "count": rounds,
                        "node": {"run": {"name": "batch"}},
                    }
                }
            ],
        }
    )


def batch_template():
    return {
        "label": "family",
        "benchmark": {
            "provider": "test",
            "model": "test",
            "adapter": "openai",
            "tokenizer": {
                "id": "test",
                "revision": "a" * 40,
                "requested_revision": "main",
                "immutable_revision": True,
                "selection": "explicit",
                "accuracy": "compatible",
            },
            "repeat": 256,
            "concurrent_requests": 8,
            "mean_input_tokens": 32768,
            "stddev_input_tokens": 0,
            "shared_prefix_tokens": 26214,
            "mean_output_tokens": 16,
            "stddev_output_tokens": 0,
        },
    }


def test_batch_compilation():
    recipe = batch_recipe()
    template = batch_template()
    context = TaskCompileContext("definition", "families", recipe, template)
    table = TaskCompiler(context).compile()
    assert len(table.instances) == 8
    assert len(table.nodes) == 800
    assert len({node.payload_seed for node in table.nodes}) == 800
    assert TaskCompiler(context).compile().nodes == table.nodes
    for instance in table.instances:
        nodes = table.nodes_for(instance.instance_key)
        assert nodes[0].dependencies == ()
        assert all(
            node.dependencies == (previous.node_id,)
            for previous, node in zip(nodes, nodes[1:])
        )
    assemblies = TaskAssembler(context, table).assemble()
    for assembly in assemblies:
        for node in assembly.nodes:
            benchmark = BatchBenchmarkConfig.model_validate(
                node.runner_template["benchmark"]
            )
            assert benchmark.repeat == 256
            assert benchmark.concurrent_requests == 8
            assert benchmark.shared_prefix_tokens == 26214
            assert benchmark.dataset_seed == benchmark.task_context.dataset_seed
            assert benchmark.task_context.execution_kind == "batch"
    assert TaskCompiler.estimate(recipe, 8, template["benchmark"]) == {
        "instances": 8,
        "nodes": 800,
        "runners": 800,
        "requests": 204800,
        "max_concurrency": 8,
        "peak_runners": 8,
        "peak_requests": 64,
        "peak_actors": 8,
    }
    assert TaskCompiler.estimate_token_budget(
        recipe.model_dump(mode="json"), 32768, 16, template["benchmark"]
    ) == 204800 * (32768 + 16)
    assessment = assess_workload(
        [],
        [],
        [{"definition": recipe.model_dump(mode="json"), "runner_template": template}],
        PerformanceGuardConfig(enabled=False),
        scheduler_slots=8,
        ray_actor_capacity=8,
        ray_actor_max_concurrency=8,
    )
    assert assessment["metrics"]["provider_requests"] == 204800
    assert assessment["metrics"]["effective_concurrency"] == 64


@pytest.mark.parametrize(
    "extra", [{"payload": "p"}, {"conversation": "chat"}, {"benchmark": {}}]
)
def test_batch_contract(extra):
    with pytest.raises(ValueError):
        TaskRecipe.model_validate(
            {"name": "invalid", "workflow": [{"run": {"name": "batch", **extra}}]}
        )
    with pytest.raises(ValueError, match="unknown payload"):
        TaskRecipe.model_validate(
            {
                "name": "invalid",
                "workflow": [{"invoke": {"name": "single", "payload": "missing"}}],
            }
        )


def test_batch_requests(monkeypatch):
    prepared = []
    sent = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

        def decode(self, tokens):
            return bytes(tokens).decode()

    def prepare(**options):
        prepared.append(options)
        return [(f"shared-prefix-{index}", 16) for index in range(4)], [
            {"source": "test", "index": index} for index in range(4)
        ]

    def execute(request):
        sent.append(request)
        return (
            {
                common_metrics.ERROR_CODE: None,
                common_metrics.E2E_LAT: 0.01,
                common_metrics.TTFT: 0.005,
                common_metrics.PROVIDER_OUTPUT_TOKENS: 1,
            },
            "ok",
            request,
        )

    monkeypatch.setattr(benchmark_module, "prepare_prompt_requests", prepare)
    summary, results = benchmark_module.get_token_throughput_latencies(
        model="test",
        prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
        mean_input_tokens=16,
        stddev_input_tokens=0,
        shared_prefix_tokens=12,
        mean_output_tokens=1,
        stddev_output_tokens=0,
        max_num_completed_requests=4,
        num_concurrent_requests=2,
        dataset_seed=1234,
        task_context={
            "execution_kind": "batch",
            "node_id": "rounds.1.batch",
            "dataset_seed": 1234,
        },
        tokenizer_override=Tokenizer(),
        request_executor=execute,
    )
    assert len(results) == len(sent) == 4
    assert len({request.prompt[0] for request in sent}) == 4
    assert prepared[0]["seed"] == 1234
    assert prepared[0]["shared_prefix_tokens"] == 12
    assert summary["task_context"]["execution_kind"] == "batch"
    assert all(request.metadata["execution_kind"] == "batch" for request in sent)


@pytest.mark.postgresql
@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled", "interrupted"])
def test_family_progression(postgresql_url, outcome):
    async def exercise():
        database = Database(DatabaseConfig(url=postgresql_url))
        try:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.create_schema()
            repository = RunnerRepository(database)
            await repository.acquire_runtime("batch-test")
            workload = await repository.create_campaign_workload(
                "batches",
                None,
                {},
                [],
                [],
                [
                    {
                        "definition": batch_recipe(2, 2).model_dump(mode="json"),
                        "runner_template": batch_template(),
                    }
                ],
                "test",
            )
            coordinator = RuntimeCoordinator(repository)
            await coordinator.reconcile()
            first = await coordinator.next_runner("first")
            second = await coordinator.next_runner("second")
            assert first is not None and second is not None
            assert first["benchmark"]["task_context"]["node_id"] == "rounds.1.batch"
            assert second["benchmark"]["task_context"]["node_id"] == "rounds.1.batch"
            await coordinator.reconcile()
            assert not coordinator.order
            metrics = [
                {
                    "error_code": None,
                    "request_metadata": {"prompt_hash": f"hash-{index}"},
                }
                for index in range(3)
            ]
            if outcome == "succeeded":
                assert await repository.complete_runner(
                    first["runner_id"], {}, metrics, 0, "", ""
                )
                assert not await repository.complete_runner(
                    first["runner_id"], {}, metrics, 0, "", ""
                )
            elif outcome == "failed":
                assert await repository.complete_runner(
                    first["runner_id"], {}, metrics, 1, "", "", terminal_status="failed"
                )
            elif outcome == "cancelled":
                await repository.request_cancel(first["runner_id"])
                await repository.finish_runner(
                    first["runner_id"], "cancelled", "cancelled", None, "", ""
                )
            else:
                await repository.release_runtime()
                await repository.acquire_runtime("recovered")
                await repository.requeue_stale(0)
            recovered = RuntimeCoordinator(repository)
            await recovered.reconcile()
            if outcome == "succeeded":
                successor = await asyncio.wait_for(recovered.next_runner("third"), 2)
                assert successor is not None
                assert (
                    successor["benchmark"]["task_context"]["node_id"]
                    == "rounds.2.batch"
                )
                assert (
                    successor["benchmark"]["task_context"]["dimensions"]
                    == first["benchmark"]["task_context"]["dimensions"]
                )
                assert (
                    successor["benchmark"]["dataset_seed"]
                    != first["benchmark"]["dataset_seed"]
                )
                still_running = await repository.get_runner(second["runner_id"])
                assert (
                    still_running is not None and still_running["status"] == "running"
                )
            else:
                assert not recovered.order
            if outcome != "interrupted":
                await repository.complete_runner(
                    second["runner_id"], {}, metrics, 0, "", ""
                )
                await recovered.reconcile()
                other_successor = await recovered.next_runner("fourth")
                assert other_successor is not None
                assert (
                    other_successor["benchmark"]["task_context"]["dimensions"]
                    == second["benchmark"]["task_context"]["dimensions"]
                )
            exported = await repository.export_campaign(
                workload["campaign"]["campaign_id"], include_requests=True
            )
            assert exported is not None
            exported_runner = next(
                runner
                for runner in exported["runners"]
                if runner["runner_id"] == first["runner_id"]
            )
            if outcome in {"succeeded", "failed"}:
                assert len(exported_runner["requests"]) == 3
            assert (
                exported_runner["benchmark"]["task_context"]["execution_kind"]
                == "batch"
            )
            await repository.request_cancel_campaign(
                workload["campaign"]["campaign_id"]
            )
        finally:
            async with database.engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
            await database.dispose()

    asyncio.run(exercise())

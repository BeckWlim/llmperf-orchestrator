from pathlib import Path

import pytest
import yaml

from llmperf_backend.configuration.models import (
    BenchmarkCampaignStart,
    BenchmarkRunnerCreate,
    PerformanceGuardConfig,
)
from llmperf_backend.compiler.rounds import expand_rounds, expand_runner_group
from llmperf_backend.compiler.safety import assess_workload
from llmperf_backend.compiler.tasks import TaskCompileContext, TaskCompiler

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_examples_contract():
    paths = sorted(EXAMPLES.glob("*.yaml"))

    assert paths
    for path in paths:
        assert path.stem.startswith("example-")
        assert path.name.count("-") <= 3

        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "campaign" not in document:
            BenchmarkRunnerCreate.model_validate(document)
            continue

        assert document.get("wait") is True
        validated_campaign = BenchmarkCampaignStart.model_validate(
            {key: value for key, value in document.items() if key != "wait"}
        )

        for definition_index, definition in enumerate(
            validated_campaign.task_definitions
        ):
            compiler = TaskCompiler(
                TaskCompileContext(
                    definition_id=f"{path.stem}:{definition_index}",
                    definition_name=definition.name,
                    definition=definition.recipe(),
                    runner_template=definition.runner.execution_document(),
                )
            )
            compilation_table = compiler.compile()

            assert compilation_table.instances
            assert compilation_table.nodes

        for plan in document.get("runner_plans", []):
            assert plan["max_occurrences"] <= 2
            assert plan["recurrence"]["every"] <= 2

        for definition in document.get("task_definitions", []):
            instances = definition["instances"]
            assert 1 <= instances["trials"] <= 2
            for values in instances.get("matrix", {}).values():
                assert max(values) <= 2


@pytest.mark.parametrize(
    ("purpose", "expected_requests"),
    [
        ("smoke", 1),
        ("campaign", 4),
        ("family-recall", 16),
        ("seed-pool", 32),
        ("cohort-recall", 12),
        ("matrix-phases", 12),
        ("runner-plan", 2),
        ("cache-probe", 4),
        ("cache", 16),
        ("dataset", 3),
        ("message-merge", 3),
    ],
)
def test_example_budgets(purpose: str, expected_requests: int):
    document = yaml.safe_load((EXAMPLES / f"example-{purpose}.yaml").read_text())
    if "campaign" not in document:
        runner = BenchmarkRunnerCreate.model_validate(document)
        assessment = assess_workload(
            [runner.execution_document()],
            [],
            [],
            PerformanceGuardConfig(),
            scheduler_slots=64,
            ray_actor_capacity=8,
            ray_actor_max_concurrency=8,
        )
    else:
        campaign = BenchmarkCampaignStart.model_validate(
            {key: value for key, value in document.items() if key != "wait"}
        )
        runner_templates = [runner.execution_document() for runner in campaign.runners]
        for group in campaign.runner_groups:
            runner_templates.extend(
                expand_runner_group(
                    group.runner.execution_document(),
                    group.name,
                    group.matrix,
                    group.phase,
                )
            )
        rounds = campaign.campaign.rounds
        expanded_runners = (
            expand_rounds(
                runner_templates,
                rounds.count,
                rounds.seed,
                rounds.prefix_seed_scope,
                rounds.phases,
            )
            if rounds is not None
            else runner_templates
        )
        assessment = assess_workload(
            expanded_runners,
            [
                {
                    "plan": plan.model_dump(mode="json", exclude={"runner", "name"}),
                    "runner_template": plan.runner.execution_document(),
                }
                for plan in campaign.runner_plans
            ],
            [
                {
                    "definition": definition.recipe().model_dump(mode="json"),
                    "runner_template": definition.runner.execution_document(),
                }
                for definition in campaign.task_definitions
            ],
            PerformanceGuardConfig(),
            scheduler_slots=64,
            ray_actor_capacity=8,
            ray_actor_max_concurrency=8,
        )
    assert assessment["metrics"]["provider_requests"] == expected_requests
    assert assessment["safe"]


def test_phase_settings():
    document = yaml.safe_load((EXAMPLES / "example-matrix-phases.yaml").read_text())
    campaign = BenchmarkCampaignStart.model_validate(
        {key: value for key, value in document.items() if key != "wait"}
    )
    load_group, pressure_group, read_group = campaign.runner_groups
    assert load_group.runner.provider is not None
    assert load_group.runner.model is not None
    assert load_group.name == read_group.name
    assert load_group.matrix == read_group.matrix
    assert load_group.runner == read_group.runner
    assert pressure_group.runner.provider == load_group.runner.provider
    assert pressure_group.runner.model == load_group.runner.model

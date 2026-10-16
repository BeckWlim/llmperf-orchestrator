from pathlib import Path

import yaml

from llmperf_backend.configuration.models import (
    BenchmarkCampaignStart,
    BenchmarkRunnerCreate,
)
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

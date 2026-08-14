from pathlib import Path

import yaml

from llmperf_backend.models import BenchmarkCampaignStart, BenchmarkRunnerCreate


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
        BenchmarkCampaignStart.model_validate(
            {key: value for key, value in document.items() if key != "wait"}
        )

        for plan in document.get("runner_plans", []):
            assert plan["max_occurrences"] <= 2
            assert plan["recurrence"]["every_seconds"] <= 2

        for definition in document.get("protocol_definitions", []):
            if definition["protocol"] == "cache-retention/v1":
                assert max(definition["delay_seconds"]) <= 2
                assert definition["trials_per_delay"] == 1
            if definition["protocol"] == "cache-residency/v1":
                assert definition["schedule"]["kind"] == "relative"
                assert max(definition["schedule"]["offsets_seconds"]) <= 2

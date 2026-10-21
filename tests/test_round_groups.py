"""Matrix phases retain family identity while imposing durable completion barriers."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
import yaml

from llmperf_backend.compiler.rounds import expand_rounds, expand_runner_group
from llmperf_backend.configuration.models import BenchmarkCampaignStart, RunnerRounds
from llmperf_cli.commands.main import build_parser, preview_campaign_tasks
from llmperf_cli.presentation.projections import _campaign_preview
from llmperf_cli.transport.client import ClientError, LLMPerfClient


def submission():
    return {
        "version": "1.0.0",
        "campaign": {
            "name": "phase-test",
            "rounds": {
                "count": 3,
                "seed": 42,
                "prefix_seed_scope": "family",
                "phases": ["load", "pressure", "read"],
            },
        },
        "runner_groups": [
            {
                "name": name,
                "phase": phase,
                "matrix": {"row": [1, 2], "column": [1, 2]},
                "runner": {
                    "provider": "test",
                    "model": "test",
                    "inputs": {
                        "default": {
                            "tokens": 200,
                            "shared_prefix_tokens": 0 if name == "pressure" else 160,
                        }
                    },
                },
            }
            for name, phase in [
                ("target", "load"),
                ("pressure", "pressure"),
                ("target", "read"),
            ]
        ],
    }


def expanded(document):
    model = BenchmarkCampaignStart.model_validate(document)
    rounds = model.campaign.rounds
    assert rounds is not None
    templates = [
        runner
        for group in model.runner_groups
        for runner in expand_runner_group(
            group.runner.execution_document(), group.name, group.matrix, group.phase
        )
    ]
    return expand_rounds(
        templates, rounds.count, rounds.seed, rounds.prefix_seed_scope, rounds.phases
    )


def test_matrix_identity():
    document = submission()
    original = deepcopy(document)
    runners = expanded(document)
    assert document == original
    assert len(runners) == 36
    assert [item["metadata"]["runner_round"]["phase"] for item in runners] == [
        phase
        for _ in range(3)
        for phase in ("load", "pressure", "read")
        for _ in range(4)
    ]
    family_prefixes = {}
    for runner in runners:
        group = runner["metadata"]["runner_group"]
        identity = (group["name"], tuple(sorted(group["dimensions"].items())))
        family_prefixes.setdefault(identity, set()).add(
            runner["benchmark"]["shared_prefix_seed"]
        )
        if group["name"] == "pressure":
            assert runner["benchmark"]["shared_prefix_tokens"] == 0
    assert len(family_prefixes) == 8
    assert all(len(values) == 1 for values in family_prefixes.values())
    assert len(set.union(*family_prefixes.values())) == 8
    assert len({runner["benchmark"]["dataset_seed"] for runner in runners}) == 36
    # Author ordering has no effect on coordinate identity or generated content.
    reordered = deepcopy(document)
    reordered["runner_groups"].reverse()
    reordered["runner_groups"][0]["matrix"] = {"column": [2, 1], "row": [2, 1]}
    assert sorted(
        item["benchmark"]["dataset_seed"] for item in expanded(reordered)
    ) == sorted(item["benchmark"]["dataset_seed"] for item in runners)


def test_date_resolution(monkeypatch):
    generator = Mock(side_effect=[101, 202])
    monkeypatch.setattr(
        "llmperf_backend.configuration.models.dated_campaign_seed", generator
    )
    first = RunnerRounds.model_validate({"count": 3, "seed": "date"})
    assert first.seed == 101
    persisted = first.model_dump(mode="json")
    assert RunnerRounds.model_validate(persisted).seed == 101
    assert RunnerRounds.model_validate({"count": 3, "seed": "date"}).seed == 202
    assert generator.call_count == 2
    assert expanded(submission()) == expanded(submission())


@pytest.mark.parametrize(
    "case", ["phase", "empty", "duplicate", "coordinates", "limit", "plans", "scope"]
)
def test_group_validation(case):
    document = submission()
    if case == "phase":
        document["runner_groups"][0]["phase"] = "missing"
    elif case == "empty":
        document["campaign"]["rounds"]["phases"].append("unused")
    elif case == "duplicate":
        document["runner_groups"].append(document["runner_groups"][0])
    elif case == "coordinates":
        document["runner_groups"][0]["matrix"]["row"] = [1, 1]
    elif case == "limit":
        document["campaign"]["rounds"]["count"] = 1000
    elif case == "plans":
        document["campaign"].pop("rounds")
    else:
        document["campaign"]["rounds"]["prefix_seed_scope"] = "phase"
    with pytest.raises(ValidationError):
        BenchmarkCampaignStart.model_validate(document)


@pytest.mark.parametrize(
    "profile,total",
    [("memory-recall", 8192), ("dfs-recall", 7168), ("tiered-recall", 8960)],
)
def test_recall_workloads(profile, total):
    document = yaml.safe_load(
        (Path(__file__).parents[1] / "workloads" / f"{profile}.yaml").read_text()
    )
    document.pop("wait")
    runners = expanded(document)
    assert len(runners) == total
    target_seeds = {
        runner["benchmark"]["shared_prefix_seed"]
        for runner in runners
        if runner["metadata"]["runner_group"]["name"] == "target"
    }
    assert len(target_seeds) == 128
    assert len({runner["benchmark"]["dataset_seed"] for runner in runners}) == total


def test_cli_groups(tmp_path):
    document = submission()
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(document))
    client = Mock(spec=LLMPerfClient)
    arguments = build_parser().parse_args(["campaign", "preview", "-f", str(path)])
    preview_campaign_tasks(client, arguments)
    assert (
        client.preview_campaign_tasks.call_args.kwargs["runner_groups"]
        == document["runner_groups"]
    )
    assert client.preview_campaign_tasks.call_args.args[0] == document["campaign"]


def test_group_projection():
    source = {
        "valid": True,
        "campaign": "test",
        "debug": False,
        "summary": dict(
            immediate_runners=4,
            runner_plans=0,
            task_definitions=0,
            task_instances=0,
            task_nodes=0,
            previewed_instances=0,
        ),
        "task_definitions": [],
        "rounds": dict(
            count=2, seed=123, prefix_seed_scope="family", phases=["a"], secret="hidden"
        ),
        "runner_groups": [
            dict(
                name="x",
                phase="a",
                matrix={"member": [1, 2]},
                families=2,
                cycles=2,
                workers_per_runner=1,
                secret="hidden",
            )
        ],
    }
    projected = _campaign_preview(source, False)
    assert "secret" not in str(projected)
    assert projected["rounds"]["seed"] == 123
    source["runner_groups"][0]["matrix"]["member"] = ["secret"]
    with pytest.raises(ClientError):
        _campaign_preview(source, False)

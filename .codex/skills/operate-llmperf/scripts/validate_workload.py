#!/usr/bin/env python3
"""Validate an LLMPerf Runner or Campaign YAML without submitting it."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

from pydantic import ValidationError
import yaml

from llmperf_backend.configuration.models import (
    BenchmarkCampaignStart,
    BenchmarkRunnerCreate,
    PerformanceGuardConfig,
    dump_model,
)
from llmperf.version import PROTOCOL_VERSION
from llmperf_backend.compiler.rounds import expand_rounds, expand_runner_group
from llmperf_backend.compiler.safety import WorkloadSafetyError, assess_workload


def load_document(path: Path) -> Dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read YAML {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("YAML document must be a mapping")
    return document


def _safety_summary(assessment: Dict[str, Any]) -> str:
    metrics = assessment["metrics"]
    return (
        f"campaign_admission={assessment['campaign_admission_enabled']} "
        f"within_limits={assessment['safe']} "
        f"planned_runners={metrics['planned_runners']} "
        f"provider_requests={metrics['provider_requests']} "
        f"token_budget={metrics['token_budget']} "
        f"effective_concurrency={metrics['effective_concurrency']} "
        f"advisory_risks={len(assessment['risks'])} "
        f"warnings={len(assessment['warnings'])}"
    )


def validate_document(
    document: Dict[str, Any],
    scheduler_slots: int = 1,
    ray_num_cpus: int = 8,
    ray_actor_num_cpus: float = 1.0,
    ray_actor_max_concurrency: int = 1,
    max_runner_concurrency: int = 32,
    max_effective_concurrency: int = 32,
    enforce_campaign_limits: bool = False,
    preparation_workers: int = 0,
) -> str:
    if ray_num_cpus < 1 or ray_actor_num_cpus <= 0:
        raise ValueError("Ray CPU and actor CPU values must be greater than zero")
    if preparation_workers < 0:
        raise ValueError("preparation_workers must be nonnegative")
    ray_actor_capacity = int((ray_num_cpus - preparation_workers) / ray_actor_num_cpus)
    if ray_actor_capacity < 1:
        raise ValueError("Ray runtime cannot schedule even one configured actor")
    guard = PerformanceGuardConfig(
        campaign_admission_enabled=enforce_campaign_limits,
        max_runner_concurrency=max_runner_concurrency,
        max_effective_concurrency=max_effective_concurrency,
    )
    if document.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"YAML version must be {PROTOCOL_VERSION!r}")
    if "campaign" not in document:
        runner_model = BenchmarkRunnerCreate.model_validate(document)
        provider = runner_model.provider or "<backend-default>"
        model = runner_model.model or "<backend-default>"
        assessment = assess_workload(
            [
                (
                    runner_model.execution_document()
                    if runner_model.worker is not None
                    or runner_model.inputs is not None
                    else {"benchmark": None}
                )
            ],
            [],
            [],
            guard,
            scheduler_slots,
            ray_actor_capacity,
            ray_actor_max_concurrency=ray_actor_max_concurrency,
        )
        return (
            f"valid runner workload: provider={provider} model={model} "
            f"{_safety_summary(assessment)}"
        )

    payload = {
        key: value for key, value in document.items() if key not in {"wait", "export"}
    }
    campaign = BenchmarkCampaignStart.model_validate(payload)
    runner_templates = [
        (
            runner.execution_document()
            if runner.worker is not None or runner.inputs is not None
            else {"benchmark": None}
        )
        for runner in campaign.runners
    ]
    rounds = campaign.campaign.rounds
    for group in campaign.runner_groups:
        runner_templates.extend(
            expand_runner_group(
                group.runner.execution_document(), group.name, group.matrix, group.phase
            )
        )
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
    plans = []
    for runner_plan in campaign.runner_plans:
        plan_document = dump_model(runner_plan)
        plan_document.pop("runner")
        plan_runner_document = (
            runner_plan.runner.execution_document()
            if runner_plan.runner.worker is not None
            or runner_plan.runner.inputs is not None
            else {"benchmark": None}
        )
        plan_document.pop("name", None)
        plans.append({"plan": plan_document, "runner_template": plan_runner_document})
    tasks = []
    for definition in campaign.task_definitions:
        task_document = dump_model(definition)
        task_document.pop("runner")
        task_runner_document = (
            definition.runner.execution_document()
            if definition.runner.worker is not None
            or definition.runner.inputs is not None
            else {"benchmark": None}
        )
        tasks.append(
            {"definition": task_document, "runner_template": task_runner_document}
        )
    assessment = assess_workload(
        expanded_runners,
        plans,
        tasks,
        guard,
        scheduler_slots,
        ray_actor_capacity,
        ray_actor_max_concurrency=ray_actor_max_concurrency,
        runtime_concurrent_workers=campaign.runtime.concurrent_workers,
    )
    return (
        "valid campaign workload: "
        f"runners={len(expanded_runners)} "
        f"runner_groups={len(campaign.runner_groups)} "
        f"runner_plans={len(campaign.runner_plans)} "
        f"task_definitions={len(campaign.task_definitions)} "
        f"{_safety_summary(assessment)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Runner/Campaign YAML without Backend or Provider calls."
    )
    parser.add_argument("file", type=Path, metavar="FILE")
    parser.add_argument(
        "--scheduler-slots", type=int, default=1, help="Backend Scheduler slot count"
    )
    parser.add_argument(
        "--ray-num-cpus", type=int, default=8, help="Shared Ray CPU resource budget"
    )
    parser.add_argument(
        "--ray-actor-num-cpus",
        type=float,
        default=1.0,
        help="CPU resource reserved by each LLM client actor",
    )
    parser.add_argument(
        "--ray-actor-max-concurrency",
        type=int,
        default=1,
        help="Maximum simultaneous requests per client actor",
    )
    parser.add_argument("--max-runner-concurrency", type=int, default=32)
    parser.add_argument("--preparation-workers", type=int, default=0)
    parser.add_argument("--max-effective-concurrency", type=int, default=32)
    parser.add_argument(
        "--enforce-campaign-limits",
        action="store_true",
        help="Reject campaign totals above packaged budgets; advisory by default.",
    )
    arguments = parser.parse_args()
    try:
        print(
            validate_document(
                load_document(arguments.file.expanduser()),
                scheduler_slots=arguments.scheduler_slots,
                ray_num_cpus=arguments.ray_num_cpus,
                ray_actor_num_cpus=arguments.ray_actor_num_cpus,
                ray_actor_max_concurrency=arguments.ray_actor_max_concurrency,
                max_runner_concurrency=arguments.max_runner_concurrency,
                max_effective_concurrency=arguments.max_effective_concurrency,
                enforce_campaign_limits=arguments.enforce_campaign_limits,
                preparation_workers=arguments.preparation_workers,
            )
        )
    except (ValueError, ValidationError, WorkloadSafetyError) as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

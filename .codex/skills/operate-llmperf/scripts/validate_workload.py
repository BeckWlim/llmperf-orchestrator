#!/usr/bin/env python3
"""Validate an LLMPerf Runner or Campaign YAML without submitting it."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

from pydantic import ValidationError
import yaml

from llmperf_backend.models import (
    BenchmarkCampaignStart,
    BenchmarkRunnerCreate,
    PerformanceGuardConfig,
    dump_model,
)
from llmperf_backend.safety import WorkloadSafetyError, assess_workload


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
        f"safety=safe planned_runners={metrics['planned_runners']} "
        f"provider_requests={metrics['provider_requests']} "
        f"token_budget={metrics['token_budget']} "
        f"effective_concurrency={metrics['effective_concurrency']} "
        f"warnings={len(assessment['warnings'])}"
    )


def validate_document(
    document: Dict[str, Any],
    scheduler_slots: int = 1,
    ray_num_cpus: int = 8,
    ray_actor_num_cpus: float = 1.0,
) -> str:
    if ray_num_cpus < 1 or ray_actor_num_cpus <= 0:
        raise ValueError("Ray CPU and actor CPU values must be greater than zero")
    ray_actor_capacity = int(ray_num_cpus / ray_actor_num_cpus)
    if ray_actor_capacity < 1:
        raise ValueError("Ray runtime cannot schedule even one configured actor")
    if "campaign" not in document:
        runner = BenchmarkRunnerCreate.model_validate(document)
        benchmark = runner.benchmark
        provider = benchmark.provider if benchmark is not None else "<backend-default>"
        model = benchmark.model if benchmark is not None else "<backend-default>"
        assessment = assess_workload(
            [dump_model(runner)],
            [],
            [],
            PerformanceGuardConfig(),
            scheduler_slots,
            ray_actor_capacity,
        )
        return (
            f"valid runner workload: provider={provider} model={model} "
            f"{_safety_summary(assessment)}"
        )

    payload = {
        key: value for key, value in document.items() if key not in {"wait", "export"}
    }
    campaign = BenchmarkCampaignStart.model_validate(payload)
    runners = [dump_model(runner) for runner in campaign.runners]
    plans = []
    for runner_plan in campaign.runner_plans:
        item = dump_model(runner_plan)
        runner = item.pop("runner")
        item.pop("name", None)
        plans.append({"plan": item, "runner_template": runner})
    tasks = []
    for definition in campaign.task_definitions:
        item = dump_model(definition)
        runner = item.pop("runner")
        tasks.append({"definition": item, "runner_template": runner})
    assessment = assess_workload(
        runners,
        plans,
        tasks,
        PerformanceGuardConfig(),
        scheduler_slots,
        ray_actor_capacity,
    )
    return (
        "valid campaign workload: "
        f"runners={len(campaign.runners)} "
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
    arguments = parser.parse_args()
    try:
        print(
            validate_document(
                load_document(arguments.file.expanduser()),
                scheduler_slots=arguments.scheduler_slots,
                ray_num_cpus=arguments.ray_num_cpus,
                ray_actor_num_cpus=arguments.ray_actor_num_cpus,
            )
        )
    except (ValueError, ValidationError, WorkloadSafetyError) as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

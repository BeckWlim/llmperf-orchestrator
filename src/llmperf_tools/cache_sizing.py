"""Estimate cache workload sizes and emit two independent campaign definitions.

This is an offline planning tool. It neither submits requests nor infers measured
hit rates from prefix ratios. KV sizes must describe the stored representation,
summed over TP shards; include replicas and non-attention state when applicable.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import yaml

from llmperf_backend.compiler.tasks import GeneratedSeedPool


GIB = 1024**3
BACKGROUND_POOL_SEED = 202610211101
PROBE_POOL_SEED = 202610211102


@dataclass(frozen=True)
class CacheSizing:
    l3_gib: float
    l4_gib: float
    kv_bytes_per_token: float
    l3_high: float = 0.8
    l4_high: float = 0.8
    l4_low: float = 0.6
    input_tokens: int = 65536
    prefix_ratio: float = 0.8
    pool_size: int = 1536
    cohort_size: int = 256
    background_pool_seed: int = BACKGROUND_POOL_SEED
    probe_pool_seed: int = PROBE_POOL_SEED
    total_slots: int = 64
    background_concurrent_workers: int = 24
    probe_lanes: int = 32
    background_repeats: int = 156
    background_gap_seconds: float = 544.0
    calibration_seconds: tuple[float, ...] = (600.0, 1200.0, 2400.0)
    recall_waves: int = 33
    recall_gap_seconds: float = 2400.0
    write_gib_per_second: float | None = None
    mean_request_seconds: float | None = None

    def __post_init__(self) -> None:
        positive_values = (
            self.l3_gib,
            self.l4_gib,
            self.kv_bytes_per_token,
            self.background_gap_seconds,
            self.recall_gap_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError(
                "capacities, KV bytes/token and gaps must be finite and positive"
            )
        if not 0 < self.l3_high < 1 or not 0 < self.l4_low < self.l4_high < 1:
            raise ValueError(
                "watermarks require 0 < L3 high < 1 and 0 < L4 low < high < 1"
            )
        if not 0 < self.prefix_ratio < 1 or self.input_tokens < 2:
            raise ValueError("input needs at least two tokens and 0 < prefix ratio < 1")
        if not 0 < self.prefix_tokens < self.input_tokens:
            raise ValueError("rounded prefix and suffix must both contain tokens")
        if not 1 <= self.pool_size <= 10000 or not 1 <= self.cohort_size <= 10000:
            raise ValueError("prefix pools must contain 1..10000 seeds")
        if any(
            not 0 <= seed < 2**64
            for seed in (self.background_pool_seed, self.probe_pool_seed)
        ):
            raise ValueError("pool seeds must be unsigned 64-bit integers")
        if self.background_pool_seed == self.probe_pool_seed:
            raise ValueError("background and probe need different parent seeds")
        if (
            not 1 <= self.background_concurrent_workers <= 100
            or not 1 <= self.probe_lanes <= 100
        ):
            raise ValueError("lane counts must be between 1 and 100")
        if self.total_slots < self.background_concurrent_workers + self.probe_lanes:
            raise ValueError(
                "background and probe concurrency exceed shared request slots"
            )
        if self.probe_lanes > self.cohort_size:
            raise ValueError("probe lanes must not exceed cohort size")
        if not 1 <= self.background_repeats <= 10000 or self.recall_waves < 1:
            raise ValueError(
                "background repeats require 1..10000; recall waves must be positive"
            )
        if any(
            not math.isfinite(delay) or delay <= 0 for delay in self.calibration_seconds
        ):
            raise ValueError("calibration gaps must be finite and positive")
        if len(self.calibration_seconds) > 98:
            raise ValueError("too many calibration phases for the task workflow")
        if self.probe_requests > 10000:
            raise ValueError(
                "probe expansion exceeds the 10000-node per-instance limit"
            )
        if max(self.background_wait_seconds, self.probe_wait_seconds) > 86400:
            raise ValueError(
                "configured waits exceed the sizing tool's 24-hour wait budget"
            )
        for optional_value in (self.write_gib_per_second, self.mean_request_seconds):
            if optional_value is not None and (
                not math.isfinite(optional_value) or optional_value <= 0
            ):
                raise ValueError(
                    "observed write rate and mean service time must be finite and positive"
                )

    @property
    def prefix_tokens(self) -> int:
        return math.floor(self.input_tokens * self.prefix_ratio)

    @property
    def background_wait_seconds(self) -> float:
        return (self.background_repeats - 1) * self.background_gap_seconds

    @property
    def probe_wait_seconds(self) -> float:
        return (
            sum(self.calibration_seconds) + self.recall_waves * self.recall_gap_seconds
        )

    @property
    def probe_requests(self) -> int:
        return self.cohort_size * (
            1 + len(self.calibration_seconds) + self.recall_waves
        )


def evaluate(settings: CacheSizing) -> dict[str, object]:
    prefix_gib = settings.prefix_tokens * settings.kv_bytes_per_token / GIB
    suffix_gib = (
        (settings.input_tokens - settings.prefix_tokens)
        * settings.kv_bytes_per_token
        / GIB
    )
    pool_gib = settings.pool_size * prefix_gib
    cohort_gib = settings.cohort_size * prefix_gib
    l3_trigger_gib = settings.l3_gib * settings.l3_high
    l4_low_gib = settings.l4_gib * settings.l4_low
    churn_headroom_gib = l4_low_gib - pool_gib - cohort_gib
    background_requests = settings.pool_size * settings.background_repeats
    warnings: list[str] = []
    if pool_gib <= l3_trigger_gib:
        warnings.append(
            "The background prefix pool fits below the L3 trigger; suffix writes or other traffic must drive eviction."
        )
    if churn_headroom_gib <= l3_trigger_gib:
        warnings.append(
            "After reserving both prefix sets, conservative L4 churn headroom is no larger than one L3 turnover. Reduce pool/cohort size."
        )
    if settings.probe_wait_seconds > settings.background_wait_seconds:
        warnings.append(
            "Probe wait budget exceeds background wait budget. Verify that the background remains active throughout recall."
        )
    planning_window: dict[str, object] | None = None
    if settings.write_gib_per_second is not None:
        early_seconds = l3_trigger_gib / settings.write_gib_per_second
        late_seconds = max(0.0, churn_headroom_gib) / settings.write_gib_per_second
        planning_window = {
            "write_rate_gib_per_second": settings.write_gib_per_second,
            "one_l3_turnover_seconds": early_seconds,
            "reserved_l4_churn_seconds": late_seconds,
            "steady_gap_written_gib": settings.recall_gap_seconds
            * settings.write_gib_per_second,
            "feasible_by_volume": early_seconds < late_seconds,
        }
        if settings.recall_gap_seconds < early_seconds:
            warnings.append(
                "Steady recall gap is shorter than the estimated L3 turnover; L3 hits may dominate."
            )
        if settings.recall_gap_seconds > late_seconds:
            warnings.append(
                "Steady recall gap exceeds reserved L4 churn headroom; target prefixes may disappear before recall."
            )
    else:
        warnings.append(
            "No measured new-workload write rate supplied: the recall interval is an uncalibrated starting point."
        )
    estimated_background_seconds = (
        settings.background_wait_seconds
        + settings.background_repeats * settings.mean_request_seconds
        if settings.mean_request_seconds is not None
        else None
    )
    return {
        "assumptions": [
            "KV bytes/token is supplied, not inferred from parameter count. Sizes exclude any unaccounted SWA state, alignment and replicas.",
            "Footprints use configured input tokens; provider tokenization can differ.",
            "Background Runners retain distinct pool seeds and repeat Workers after a cooldown. All suffixes vary; pool reuse does not guarantee a tier hit.",
            "Write rate must be measured under the proposed background, including other traffic. Eviction/metadata counters are not physical I/O bandwidth.",
            "The retention window reserves both complete prefix sets below L4's low watermark. It is a conservative volume heuristic, not an LRU or lease guarantee.",
            "Delays are completion-relative; provider latency, preparation and scheduling extend duration. No hard 24-hour cancellation is encoded.",
        ],
        "tokens": {
            "input": settings.input_tokens,
            "prefix": settings.prefix_tokens,
            "suffix": settings.input_tokens - settings.prefix_tokens,
        },
        "capacity_gib": {
            "l3": settings.l3_gib,
            "l3_trigger": l3_trigger_gib,
            "l4": settings.l4_gib,
            "l4_high": settings.l4_gib * settings.l4_high,
            "l4_low": l4_low_gib,
        },
        "footprint_gib": {
            "one_prefix": prefix_gib,
            "one_suffix": suffix_gib,
            "background_prefix_pool": pool_gib,
            "probe_cohort": cohort_gib,
            "reserved_l4_churn_headroom": churn_headroom_gib,
            "suffix_writes_between_mean_pool_draw_revisits": settings.pool_size
            * suffix_gib,
        },
        "background": {
            "runners": settings.pool_size,
            "workers_per_runner": settings.background_repeats,
            "runner_slots_released_during_cooldown": True,
            "requests": background_requests,
            "maximum_concurrency": settings.background_concurrent_workers,
            "gap_only_rate_ceiling_rps": settings.pool_size
            / settings.background_gap_seconds,
            "minimum_wait_budget_seconds": settings.background_wait_seconds,
            "estimated_duration_seconds": estimated_background_seconds,
        },
        "probe": {
            "runners": settings.probe_requests,
            "requests": settings.probe_requests,
            "maximum_concurrency": settings.probe_lanes,
            "prefixes_per_wave": settings.cohort_size,
            "recall_waves": len(settings.calibration_seconds) + settings.recall_waves,
            "minimum_wait_budget_seconds": settings.probe_wait_seconds,
            "maximum_offered_prefix_read_gib": (
                len(settings.calibration_seconds) + settings.recall_waves
            )
            * cohort_gib,
        },
        "combined": {
            "maximum_concurrency": settings.background_concurrent_workers
            + settings.probe_lanes,
            "unused_slots": settings.total_slots
            - settings.background_concurrent_workers
            - settings.probe_lanes,
            "requests": background_requests + settings.probe_requests,
        },
        "planning_window": planning_window,
        "warnings": warnings,
    }


def build_campaigns(
    settings: CacheSizing, runner_template: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    runner_document = deepcopy(dict(runner_template))
    raw_inputs = runner_document.get("inputs")
    if not isinstance(raw_inputs, dict) or not isinstance(
        raw_inputs.get("default"), dict
    ):
        raise ValueError("Runner template requires inputs.default")
    input_document = dict(raw_inputs["default"])
    input_document.update(
        tokens=settings.input_tokens, shared_prefix_tokens=settings.prefix_tokens
    )
    # Explicit seeds are assigned by the task compiler; do not retain legacy overrides.
    input_document.pop("dataset_seed", None)
    input_document.pop("shared_prefix_seed", None)
    runner_document.update(
        inputs={"default": input_document},
        worker={"repeat": 1},
        concurrent_workers=1,
        timeout=1800,
    )
    runner_document.pop("metadata", None)
    runner_document.pop("request", None)
    runner_document.pop("concurrent_requests", None)
    background_runner = dict(
        runner_document,
        label="background-prefix-pool",
        timeout=172800,
        worker={
            "repeat": settings.background_repeats,
            "every": settings.background_gap_seconds,
        },
    )
    background_document: dict[str, object] = {
        "version": "1.0.0",
        "wait": False,
        "runtime": {"concurrent_workers": settings.background_concurrent_workers},
        "campaign": {
            "name": "background-pressure",
            "description": "One Runner per distinct pool seed, with repeated Workers and completion-relative cooldowns.",
        },
        "task_definitions": [
            {
                "name": "background-families",
                "instances": {"seed": 202610211201},
                "workflow": [
                    {
                        "for_each": {
                            "name": "prefixes",
                            "variable": "prefix",
                            "values": {
                                "seed": settings.background_pool_seed,
                                "count": settings.pool_size,
                            },
                            "parallelism": settings.pool_size,
                            "node": {
                                "run": {
                                    "name": "batch",
                                    "role": "background",
                                    "prefix_seed": "$prefix",
                                }
                            },
                        }
                    }
                ],
                "runner": background_runner,
            }
        ],
    }
    probe_seeds = GeneratedSeedPool(
        seed=settings.probe_pool_seed, count=settings.cohort_size
    ).values
    background_seeds = GeneratedSeedPool(
        seed=settings.background_pool_seed, count=settings.pool_size
    ).values
    if not set(probe_seeds).isdisjoint(background_seeds):
        raise ValueError("generated background and probe prefix pools overlap")
    cohort = {
        "name": "cohort",
        "variable": "prefix",
        "values": {"seed": settings.probe_pool_seed, "count": settings.cohort_size},
        "parallelism": settings.probe_lanes,
        "node": {"run": {"name": "request", "prefix_seed": "$prefix"}},
    }
    workflow: list[dict[str, object]] = [
        {"sequence": {"name": "prime", "steps": [{"for_each": cohort}]}}
    ]
    for delay_index, delay_seconds in enumerate(settings.calibration_seconds):
        workflow.append(
            {
                "sequence": {
                    "name": f"calibration_{delay_index + 1}",
                    "after": delay_seconds,
                    "steps": [{"for_each": cohort}],
                }
            }
        )
    workflow.append(
        {
            "repeat": {
                "name": "sustained_recall",
                "count": settings.recall_waves,
                "every": settings.recall_gap_seconds,
                "node": {"for_each": cohort},
            }
        }
    )
    probe_document: dict[str, object] = {
        "version": "1.0.0",
        "wait": False,
        "campaign": {
            "name": "l4-periodic-recall",
            "description": "Bounded concurrent recall waves with fixed prefixes and fresh suffixes.",
            "tags": {"cohort_parent_seed": str(settings.probe_pool_seed)},
        },
        "task_definitions": [
            {
                "name": "l4-cohort",
                "instances": {"seed": 202610211202},
                "workflow": workflow,
                "runner": dict(runner_document, label="l4-periodic-prefix-recall"),
            }
        ],
    }
    return background_document, probe_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l3-gib", type=float, required=True)
    parser.add_argument("--l4-gib", type=float, required=True)
    parser.add_argument("--kv-bytes-per-token", type=float, required=True)
    defaults = CacheSizing(l3_gib=1000, l4_gib=4916, kv_bytes_per_token=14336)
    for option in (
        "l3_high",
        "l4_high",
        "l4_low",
        "prefix_ratio",
        "background_gap_seconds",
        "recall_gap_seconds",
    ):
        parser.add_argument(
            "--" + option.replace("_", "-"),
            type=float,
            default=getattr(defaults, option),
        )
    for option in (
        "input_tokens",
        "pool_size",
        "cohort_size",
        "background_pool_seed",
        "probe_pool_seed",
        "total_slots",
        "background_concurrent_workers",
        "probe_lanes",
        "background_repeats",
        "recall_waves",
    ):
        parser.add_argument(
            "--" + option.replace("_", "-"), type=int, default=getattr(defaults, option)
        )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        nargs="*",
        default=list(defaults.calibration_seconds),
    )
    parser.add_argument("--write-gib-per-second", type=float)
    parser.add_argument("--mean-request-seconds", type=float)
    parser.add_argument(
        "--runner-template",
        type=Path,
        help="Existing campaign YAML supplying its first task's Runner template",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write basic-pressure.yaml and l4-recall.yaml; does not submit them",
    )
    arguments = parser.parse_args()
    try:
        parameter_values = {
            name: getattr(arguments, name) for name in CacheSizing.__dataclass_fields__
        }
        parameter_values["calibration_seconds"] = tuple(arguments.calibration_seconds)
        settings = CacheSizing(**parameter_values)
        assessment = evaluate(settings)
        if arguments.output_dir is not None:
            if arguments.runner_template is None:
                raise ValueError("--output-dir requires --runner-template")
            raw_template = yaml.safe_load(arguments.runner_template.read_text())
            if not isinstance(raw_template, dict):
                raise ValueError("Runner template campaign must be a mapping")
            from llmperf_backend.configuration.models import BenchmarkCampaignStart

            template_campaign = BenchmarkCampaignStart.model_validate(
                {key: value for key, value in raw_template.items() if key != "wait"}
            )
            if not template_campaign.task_definitions:
                raise ValueError(
                    "Runner template campaign must contain a task definition"
                )
            runner_template = template_campaign.task_definitions[0].runner.model_dump(
                mode="json", exclude_none=True
            )
            campaign_documents = build_campaigns(settings, runner_template)
            for campaign_document in campaign_documents:
                BenchmarkCampaignStart.model_validate(
                    {
                        key: value
                        for key, value in campaign_document.items()
                        if key != "wait"
                    }
                )
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            for filename, campaign_document in zip(
                ("basic-pressure.yaml", "l4-recall.yaml"), campaign_documents
            ):
                output_path = arguments.output_dir / filename
                output_path.write_text(
                    "# Generated by python -m llmperf_tools.cache_sizing; see workloads/README.md.\n"
                    "# Worker cooldowns and task delays are completion-relative; hit rates require measurement.\n"
                    + yaml.safe_dump(
                        campaign_document,
                        sort_keys=False,
                        width=110,
                        default_flow_style=None,
                    )
                )
            assessment["written_to"] = str(arguments.output_dir)
        print(json.dumps(assessment, indent=2, allow_nan=False))
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Campaign-level analysis for durable cross-Runner cache-retention pairs."""

from collections import Counter
import math
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from llmperf import common_metrics


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    samples = [value for value in values if value is not None]
    return {
        "count": len(samples),
        "median": median(samples) if samples else None,
        "p95": _quantile(samples, 0.95),
    }


def _runner_ttft(runner: Optional[Mapping[str, Any]]) -> Optional[float]:
    results = ((runner or {}).get("summary") or {}).get("results") or {}
    ttft = results.get(common_metrics.TTFT) or {}
    quantiles = ttft.get("quantiles") or {}
    return _number(quantiles.get("p50", ttft.get("mean")))


def _runner_cache(runner: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    results = ((runner or {}).get("summary") or {}).get("results") or {}
    cache = results.get(common_metrics.KV_CACHE) or {}
    return {
        "request_hit_ratio": _number(cache.get("request_hit_ratio")),
        "weighted_token_hit_ratio": _number(cache.get("weighted_token_hit_ratio")),
    }


def analyze_cache_retention(
    instances: Sequence[Mapping[str, Any]],
    dispatches: Sequence[Mapping[str, Any]],
    runners: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build curves from generic cache-retention protocol instances."""

    dispatch_map = {
        (str(item.get("protocol_instance_id")), str(item.get("role"))): item
        for item in dispatches
        if item.get("protocol_instance_id") and item.get("role")
    }
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for instance in instances:
        if instance.get("protocol") != "cache-retention/v1":
            continue
        grouped.setdefault(str(instance["protocol_definition_id"]), []).append(instance)
    analyses = []
    for sweep_id, sweep_pairs in sorted(grouped.items()):
        by_delay: Dict[int, List[Mapping[str, Any]]] = {}
        for instance in sweep_pairs:
            spec = instance.get("spec") or {}
            by_delay.setdefault(int(spec["delay_seconds"]), []).append(instance)
        curve = []
        for delay, delay_pairs in sorted(by_delay.items()):
            actual_delays: List[Optional[float]] = []
            hit_probabilities: List[Optional[float]] = []
            token_hit_ratios: List[Optional[float]] = []
            prime_warm_benefits: List[Optional[float]] = []
            control_warm_benefits: List[Optional[float]] = []
            for instance in delay_pairs:
                instance_id = str(instance["protocol_instance_id"])
                prime_dispatch = dispatch_map.get((instance_id, "prime")) or {}
                warm_dispatch = dispatch_map.get((instance_id, "warm")) or {}
                control_dispatch = dispatch_map.get((instance_id, "cold_control")) or {}
                prime = runners.get(str(prime_dispatch.get("runner_id")))
                warm = runners.get(str(warm_dispatch.get("runner_id")))
                control = runners.get(str(control_dispatch.get("runner_id")))
                prime_ttft = _runner_ttft(prime)
                warm_ttft = _runner_ttft(warm)
                control_ttft = _runner_ttft(control)
                cache = _runner_cache(warm)
                outcome = instance.get("outcome") or {}
                actual_delays.append(_number(outcome.get("actual_delay_seconds")))
                hit_probabilities.append(cache["request_hit_ratio"])
                token_hit_ratios.append(cache["weighted_token_hit_ratio"])
                prime_warm_benefits.append(
                    prime_ttft - warm_ttft
                    if prime_ttft is not None and warm_ttft is not None
                    else None
                )
                control_warm_benefits.append(
                    control_ttft - warm_ttft
                    if control_ttft is not None and warm_ttft is not None
                    else None
                )
            hit_summary = _summary(hit_probabilities)
            control_summary = _summary(control_warm_benefits)
            if hit_summary["count"]:
                verdict = (
                    "accounting_observed"
                    if (hit_summary["median"] or 0) > 0
                    else "not_observed"
                )
            elif control_summary["count"] and (control_summary["median"] or 0) > 0:
                verdict = "latency_inferred"
            else:
                verdict = "inconclusive"
            curve.append(
                {
                    "planned_delay_seconds": delay,
                    "instances": len(delay_pairs),
                    "states": dict(Counter(str(item["state"]) for item in delay_pairs)),
                    "actual_delay_seconds": _summary(actual_delays),
                    "provider_request_hit_probability": hit_summary,
                    "provider_weighted_token_hit_ratio": _summary(token_hit_ratios),
                    "prime_minus_warm_ttft_s": _summary(prime_warm_benefits),
                    "cold_control_minus_warm_ttft_s": control_summary,
                    "verdict": verdict,
                }
            )
        observed = [
            point["planned_delay_seconds"]
            for point in curve
            if point["verdict"] == "accounting_observed"
        ]
        analyses.append(
            {
                "protocol_definition_id": sweep_id,
                "protocol": "cache-retention/v1",
                "pairing": "independent_family",
                "curve": curve,
                "last_accounting_observed_delay_seconds": (
                    max(observed) if observed else None
                ),
            }
        )
    return analyses


def analyze_cache_residency(
    instances: Sequence[Mapping[str, Any]],
    dispatches: Sequence[Mapping[str, Any]],
    runners: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build access-conditioned curves from single-Prime dispatch chains."""

    dispatch_map = {
        (str(item.get("protocol_instance_id")), str(item.get("role"))): item
        for item in dispatches
        if item.get("protocol_instance_id") and item.get("role")
    }
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for instance in instances:
        if instance.get("protocol") != "cache-residency/v1":
            continue
        grouped.setdefault(str(instance["protocol_definition_id"]), []).append(instance)

    analyses = []
    for definition_id, chains in sorted(grouped.items()):
        observation_specs = list(
            ((chains[0].get("spec") or {}).get("observations") or [])
        )
        curve = []
        for observation_spec in observation_specs:
            observation_index = int(observation_spec["observation_index"])
            offset_seconds = int(observation_spec["offset_seconds"])
            actual_delays: List[Optional[float]] = []
            hit_probabilities: List[Optional[float]] = []
            token_hit_ratios: List[Optional[float]] = []
            prime_warm_benefits: List[Optional[float]] = []
            control_warm_benefits: List[Optional[float]] = []
            state_counts: Counter[str] = Counter()
            for chain in chains:
                instance_id = str(chain["protocol_instance_id"])
                state_counts[str(chain["state"])] += 1
                prime_dispatch = dispatch_map.get((instance_id, "prime")) or {}
                warm_dispatch = (
                    dispatch_map.get((instance_id, f"warm:{observation_index}")) or {}
                )
                control_dispatch = (
                    dispatch_map.get((instance_id, f"cold_control:{observation_index}"))
                    or {}
                )
                prime = runners.get(str(prime_dispatch.get("runner_id")))
                warm = runners.get(str(warm_dispatch.get("runner_id")))
                control = runners.get(str(control_dispatch.get("runner_id")))
                prime_ttft = _runner_ttft(prime)
                warm_ttft = _runner_ttft(warm)
                control_ttft = _runner_ttft(control)
                cache = _runner_cache(warm)
                observation = (
                    (chain.get("outcome") or {}).get("observations") or {}
                ).get(str(observation_index)) or {}
                actual_delays.append(
                    _number(observation.get("warm_actual_delay_seconds"))
                )
                hit_probabilities.append(cache["request_hit_ratio"])
                token_hit_ratios.append(cache["weighted_token_hit_ratio"])
                prime_warm_benefits.append(
                    prime_ttft - warm_ttft
                    if prime_ttft is not None and warm_ttft is not None
                    else None
                )
                control_warm_benefits.append(
                    control_ttft - warm_ttft
                    if control_ttft is not None and warm_ttft is not None
                    else None
                )
            hit_summary = _summary(hit_probabilities)
            control_summary = _summary(control_warm_benefits)
            if hit_summary["count"]:
                verdict = (
                    "accounting_observed"
                    if (hit_summary["median"] or 0) > 0
                    else "not_observed"
                )
            elif control_summary["count"] and (control_summary["median"] or 0) > 0:
                verdict = "latency_inferred"
            else:
                verdict = "inconclusive"
            curve.append(
                {
                    "observation_index": observation_index,
                    "planned_offset_seconds": offset_seconds,
                    "planned_scheduled_at": observation_spec.get("scheduled_at"),
                    "chains": len(chains),
                    "states": dict(state_counts),
                    "actual_delay_seconds": _summary(actual_delays),
                    "provider_request_hit_probability": hit_summary,
                    "provider_weighted_token_hit_ratio": _summary(token_hit_ratios),
                    "prime_minus_warm_ttft_s": _summary(prime_warm_benefits),
                    "cold_control_minus_warm_ttft_s": control_summary,
                    "verdict": verdict,
                }
            )
        observed = [
            point["planned_offset_seconds"]
            for point in curve
            if point["verdict"] == "accounting_observed"
        ]
        analyses.append(
            {
                "protocol_definition_id": definition_id,
                "protocol": "cache-residency/v1",
                "pairing": "bundled_prime_mapping",
                "interpretation": "access_conditioned_residency",
                "schedule": (chains[0].get("spec") or {}).get("schedule"),
                "curve": curve,
                "last_accounting_observed_offset_seconds": (
                    max(observed) if observed else None
                ),
            }
        )
    return analyses


def analyze_cache_protocols(
    instances: Sequence[Mapping[str, Any]],
    dispatches: Sequence[Mapping[str, Any]],
    runners: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Analyze every supported cache protocol without coupling the exporter."""

    return analyze_cache_retention(
        instances, dispatches, runners
    ) + analyze_cache_residency(instances, dispatches, runners)

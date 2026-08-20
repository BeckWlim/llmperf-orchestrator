"""Aggregation and paired analysis for external KV-cache probes."""

import math
import random
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from llmperf import common_metrics


def _valid_counter(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "median": median(samples) if samples else None,
        "p50": _quantile(samples, 0.5),
        "p95": _quantile(samples, 0.95),
    }


def _summarize_cache_counters(
    requests: Sequence[Mapping[str, Any]], include_roles: bool
) -> Dict[str, Any]:
    """Aggregate cache counters without treating missing values as zero."""

    hit_values: List[int] = []
    complete_pairs: List[Tuple[int, int]] = []
    schemas = set()
    invalid_requests = 0
    request_hits = 0
    for request in requests:
        hit = _valid_counter(request.get(common_metrics.KV_CACHE_HIT_TOKENS))
        miss = _valid_counter(request.get(common_metrics.KV_CACHE_MISS_TOKENS))
        normalized = request.get(common_metrics.NORMALIZED_USAGE)
        if isinstance(normalized, Mapping):
            schema = normalized.get("source_schema")
            if schema:
                schemas.add(str(schema))
            if normalized.get("valid") is False:
                invalid_requests += 1
                continue
        if hit is not None:
            hit_values.append(hit)
            if hit > 0:
                request_hits += 1
        if hit is not None and miss is not None:
            complete_pairs.append((hit, miss))

    complete_hit = sum(hit for hit, _ in complete_pairs)
    complete_miss = sum(miss for _, miss in complete_pairs)
    denominator = complete_hit + complete_miss
    request_count = len(requests)
    result = {
        "measured_requests": len(hit_values),
        "requests_total": request_count,
        "complete_counter_requests": len(complete_pairs),
        "counter_coverage": len(complete_pairs) / request_count if request_count else 0,
        "request_hit_probability": (
            request_hits / len(hit_values) if hit_values else None
        ),
        "weighted_token_hit_ratio": complete_hit / denominator if denominator else None,
        "complete_hit_tokens": complete_hit,
        "complete_miss_tokens": complete_miss,
        "invalid_counter_requests": invalid_requests,
        "counter_schemas": sorted(schemas),
    }
    if include_roles:
        roles: Dict[str, List[Mapping[str, Any]]] = {}
        for request in requests:
            metadata = request.get(common_metrics.REQUEST_METADATA)
            if isinstance(metadata, Mapping) and metadata.get("role"):
                roles.setdefault(str(metadata["role"]), []).append(request)
        result["by_role"] = {
            role: _summarize_cache_counters(role_requests, include_roles=False)
            for role, role_requests in sorted(roles.items())
        }
    return result


def summarize_cache_counters(requests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate cache counters without treating missing values as zero."""

    return _summarize_cache_counters(requests, include_roles=True)


def _bootstrap_median_interval(
    values: Sequence[float], samples: int, confidence: float, seed: int
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        bootstrapped.append(float(median(draw)))
    tail = (1 - confidence) / 2
    return _quantile(bootstrapped, tail), _quantile(bootstrapped, 1 - tail)


def analyze_cache_probe(
    requests: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 11111,
    minimum_counter_coverage: float = 0.8,
) -> Dict[str, Any]:
    """Analyze prime/warm pairs and return an evidence-strength verdict."""

    families: Dict[str, Dict[str, Any]] = {}
    role_ttft: Dict[str, List[float]] = {"prime": [], "warm": []}
    for request in requests:
        metadata = request.get(common_metrics.REQUEST_METADATA)
        if not isinstance(metadata, Mapping):
            continue
        family_id = metadata.get("family_id")
        role = metadata.get("role")
        if not family_id or role not in {"prime", "warm"}:
            continue
        if request.get(common_metrics.ERROR_CODE) is not None:
            continue
        ttft = request.get(common_metrics.TTFT)
        if not isinstance(ttft, (int, float)):
            continue
        role_ttft[role].append(float(ttft))
        family = families.setdefault(str(family_id), {"prime": None, "warm": []})
        if role == "prime":
            family["prime"] = request
        else:
            family["warm"].append(request)

    deltas: List[float] = []
    speedups: List[float] = []
    paired_warm_requests: List[Mapping[str, Any]] = []
    for family in families.values():
        prime = family["prime"]
        if prime is None:
            continue
        prime_ttft = float(prime[common_metrics.TTFT])
        for warm in family["warm"]:
            warm_ttft = float(warm[common_metrics.TTFT])
            deltas.append(prime_ttft - warm_ttft)
            if warm_ttft > 0:
                speedups.append(prime_ttft / warm_ttft)
            paired_warm_requests.append(warm)

    ci_low, ci_high = _bootstrap_median_interval(
        deltas, bootstrap_samples, confidence_level, seed
    )
    cache = summarize_cache_counters(paired_warm_requests)

    def valid_hit(request: Mapping[str, Any]) -> bool:
        normalized = request.get(common_metrics.NORMALIZED_USAGE)
        if isinstance(normalized, Mapping) and normalized.get("valid") is False:
            return False
        return (
            _valid_counter(request.get(common_metrics.KV_CACHE_HIT_TOKENS)) or 0
        ) > 0

    observed_hit = any(valid_hit(request) for request in paired_warm_requests)
    counters_present = cache["measured_requests"] > 0
    counter_coverage_ok = cache["counter_coverage"] >= minimum_counter_coverage
    latency_positive = ci_low is not None and ci_low > 0

    if observed_hit and counter_coverage_ok and latency_positive:
        verdict = "confirmed_external"
    elif observed_hit:
        verdict = "accounting_confirmed"
    elif not counters_present and latency_positive:
        verdict = "latency_inferred"
    elif counters_present and counter_coverage_ok and not observed_hit:
        verdict = "not_observed"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "paired_samples": len(deltas),
        "prime_ttft_s": _summary(role_ttft["prime"]),
        "warm_ttft_s": _summary(role_ttft["warm"]),
        "paired_ttft_delta_s": {
            **_summary(deltas),
            "confidence_level": confidence_level,
            "confidence_interval": [ci_low, ci_high],
        },
        "speedup": _summary(speedups),
        "cache": cache,
    }

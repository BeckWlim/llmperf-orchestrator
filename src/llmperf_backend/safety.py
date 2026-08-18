"""Static performance admission estimates for Runner and Campaign workloads."""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from llmperf_backend.models import PerformanceGuardConfig
from llmperf_backend.planner import preview_fires
from llmperf_backend.task_compiler import estimate_task_definition


UTC = timezone.utc
LOGGER = logging.getLogger(__name__)


class WorkloadSafetyError(ValueError):
    """Raised when a workload exceeds a configured performance limit."""

    def __init__(self, assessment: Dict[str, Any]):
        self.assessment = assessment
        codes = ", ".join(item["code"] for item in assessment["risks"])
        super().__init__(f"workload rejected by performance guard: {codes}")


def host_memory_snapshot() -> Dict[str, Any]:
    """Read host memory pressure without adding a runtime dependency."""

    values: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw_value = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable"}:
                values[name] = int(raw_value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        values = {}
    if "MemTotal" not in values or "MemAvailable" not in values:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            values = {
                "MemTotal": int(os.sysconf("SC_PHYS_PAGES")) * page_size,
                "MemAvailable": int(os.sysconf("SC_AVPHYS_PAGES")) * page_size,
            }
        except (AttributeError, OSError, ValueError):
            return {"available": False}
    total = values["MemTotal"]
    available = values["MemAvailable"]
    return {
        "available": total > 0,
        "total_bytes": total,
        "available_bytes": available,
        "utilization": 1 - (available / total) if total > 0 else None,
    }


class RuntimePerformanceGuard:
    """Hysteretic host-memory circuit breaker for Scheduler claims."""

    def __init__(self, config: PerformanceGuardConfig, sampler=host_memory_snapshot):
        self.config = config
        self.sampler = sampler
        self._tripped = False
        self._last_sample_at = 0.0
        self._snapshot: Dict[str, Any] = {"available": False}

    def status(self) -> Dict[str, Any]:
        now = time.monotonic()
        if now - self._last_sample_at >= self.config.sample_interval_seconds:
            self._snapshot = self.sampler()
            self._last_sample_at = now
            utilization = self._snapshot.get("utilization")
            if self.config.enabled and isinstance(utilization, (int, float)):
                was_tripped = self._tripped
                if self._tripped:
                    self._tripped = (
                        utilization > self.config.resume_host_memory_utilization
                    )
                else:
                    self._tripped = (
                        utilization >= self.config.max_host_memory_utilization
                    )
                if self._tripped and not was_tripped:
                    LOGGER.warning(
                        "Performance guard stopped new Runner claims: "
                        "host memory utilization %.3f >= %.3f",
                        utilization,
                        self.config.max_host_memory_utilization,
                    )
                elif was_tripped and not self._tripped:
                    LOGGER.info(
                        "Performance guard resumed Runner claims: "
                        "host memory utilization %.3f <= %.3f",
                        utilization,
                        self.config.resume_host_memory_utilization,
                    )
        return {
            "enabled": self.config.enabled,
            "tripped": self._tripped if self.config.enabled else False,
            "reason": "host_memory_high" if self._tripped else None,
            "max_host_memory_utilization": (self.config.max_host_memory_utilization),
            "resume_host_memory_utilization": (
                self.config.resume_host_memory_utilization
            ),
            "host_memory": dict(self._snapshot),
        }

    def allow_claim(self) -> bool:
        return not self.status()["tripped"]


def _as_datetime(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _benchmark_cost(benchmark: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    if benchmark is None:
        return {"requests": 0, "tokens": 0, "concurrency": 0, "unknown": 1}
    probe = benchmark.get("cache_probe")
    if probe:
        requests = int(probe["trials"]) * (1 + int(probe.get("repeats_after_prime", 1)))
        concurrency = min(int(benchmark["concurrent_requests"]), int(probe["trials"]))
    else:
        requests = int(benchmark["max_completed_requests"])
        concurrency = min(int(benchmark["concurrent_requests"]), requests)
    tokens_per_request = int(benchmark["mean_input_tokens"]) + int(
        benchmark["mean_output_tokens"]
    )
    return {
        "requests": requests,
        "tokens": requests * tokens_per_request,
        "concurrency": concurrency,
        "unknown": 0,
    }


def _plan_occurrences(plan: Mapping[str, Any], now: datetime) -> int:
    starts_at = _as_datetime(plan.get("starts_at")) or now
    ends_at = _as_datetime(plan.get("ends_at"))
    maximum = plan.get("max_occurrences")
    recurrence = plan["recurrence"]
    if recurrence["kind"] == "interval":
        if ends_at is None:
            return int(maximum)
        start_utc = starts_at.astimezone(UTC)
        end_utc = ends_at.astimezone(UTC)
        by_end = max(
            0,
            int((end_utc - start_utc).total_seconds())
            // int(recurrence["every_seconds"])
            + 1,
        )
        return min(by_end, int(maximum)) if maximum is not None else by_end

    scan_limit = int(maximum) if maximum is not None else 100_001
    timing = dict(plan)
    timing["starts_at"] = starts_at
    timing["ends_at"] = ends_at
    return len(preview_fires(timing, scan_limit, default_starts_at=now))


def _add_cost(total: Dict[str, int], cost: Mapping[str, int], multiplier: int) -> None:
    total["planned_runners"] += multiplier
    total["provider_requests"] += int(cost["requests"]) * multiplier
    total["token_budget"] += int(cost["tokens"]) * multiplier
    total["max_runner_concurrency"] = max(
        total["max_runner_concurrency"], int(cost["concurrency"])
    )
    total["unknown_benchmarks"] += int(cost["unknown"])


def assess_workload(
    runners: Sequence[Mapping[str, Any]],
    runner_plans: Sequence[Mapping[str, Any]],
    task_definitions: Sequence[Mapping[str, Any]],
    guard: PerformanceGuardConfig,
    scheduler_slots: int,
    ray_actor_capacity: Optional[int] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Estimate fan-out and reject workloads beyond configured hard limits."""

    assessed_at = now or datetime.now(UTC)
    total = {
        "planned_runners": 0,
        "provider_requests": 0,
        "token_budget": 0,
        "max_runner_concurrency": 0,
        "unknown_benchmarks": 0,
    }
    for runner in runners:
        _add_cost(total, _benchmark_cost(runner.get("benchmark")), 1)

    for item in runner_plans:
        plan = item.get("plan", item)
        template = item.get("runner_template", item.get("runner", {}))
        _add_cost(
            total,
            _benchmark_cost(template.get("benchmark")),
            _plan_occurrences(plan, assessed_at),
        )

    for item in task_definitions:
        definition = item.get("definition", item)
        template = item.get("runner_template", definition.get("runner", {}))
        benchmark = template.get("benchmark")
        expansion = estimate_task_definition(definition)
        requests = expansion["nodes"]
        runners_count = requests
        tokens_per_request = (
            int(benchmark["mean_input_tokens"]) + int(benchmark["mean_output_tokens"])
            if benchmark is not None
            else 0
        )
        total["planned_runners"] += runners_count
        total["provider_requests"] += requests
        total["token_budget"] += requests * tokens_per_request
        total["max_runner_concurrency"] = max(
            total["max_runner_concurrency"], 1 if benchmark is not None else 0
        )
        total["unknown_benchmarks"] += int(benchmark is None)

    effective_concurrency = (
        min(total["planned_runners"], scheduler_slots) * total["max_runner_concurrency"]
    )
    metrics = {**total, "effective_concurrency": effective_concurrency}
    limits = {
        "planned_runners": guard.max_campaign_runners,
        "provider_requests": guard.max_campaign_provider_requests,
        "token_budget": guard.max_campaign_token_budget,
        "max_runner_concurrency": guard.max_runner_concurrency,
        "effective_concurrency": guard.max_effective_concurrency,
    }
    if ray_actor_capacity is not None:
        metrics["ray_actor_demand"] = effective_concurrency
        limits["ray_actor_demand"] = ray_actor_capacity
    risks: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for metric, limit in limits.items():
        value = metrics[metric]
        issue = {"code": f"{metric}_limit", "value": value, "limit": limit}
        if value > limit:
            risks.append(issue)
        elif value >= limit * guard.warning_ratio:
            warnings.append(issue)
    if total["unknown_benchmarks"]:
        warnings.append(
            {
                "code": "backend_default_benchmark_unknown",
                "value": total["unknown_benchmarks"],
                "limit": 0,
            }
        )
    assessment = {
        "safe": not risks,
        "metrics": metrics,
        "limits": limits,
        "risks": risks,
        "warnings": warnings,
    }
    if guard.enabled and risks:
        raise WorkloadSafetyError(assessment)
    return assessment

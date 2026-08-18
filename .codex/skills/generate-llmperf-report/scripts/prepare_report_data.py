#!/usr/bin/env python3
"""Normalize versioned LLMPerf exports into auditable, chart-neutral evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ANALYSIS_VERSION = 2


def get_path(document: Any, path: str, default: Any = None) -> Any:
    current = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def integer(value: Any) -> Optional[int]:
    value = number(value)
    return int(value) if value is not None else None


def median(values: Iterable[Any]) -> Optional[float]:
    clean = [item for value in values if (item := number(value)) is not None]
    return float(statistics.median(clean)) if clean else None


def bounded_message(value: Any, limit: int = 280) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"https?://[^\s)'\"]+", "[endpoint redacted]", value)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def normalize_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    if document.get("version") == 6 and isinstance(document.get("runners"), list):
        return {
            "kind": "campaign",
            "version": 6,
            "meta": dict(document.get("campaign") or {}),
            "aggregate": dict(document.get("aggregate") or {}),
            "runner_plans": list(document.get("runner_plans") or []),
            "task_definitions": list(document.get("task_definitions") or []),
            "task_instances": list(document.get("task_instances") or []),
            "dispatches": list(document.get("dispatches") or []),
            "runners": [
                dict(item) for item in document["runners"] if isinstance(item, Mapping)
            ],
        }
    if document.get("version") == 1 and isinstance(document.get("runner"), Mapping):
        runner = dict(document["runner"])
        persisted = document.get("results") or {}
        if isinstance(persisted, Mapping):
            runner["summary"] = persisted.get("summary")
            runner["request_count"] = persisted.get("request_count")
            runner["requests"] = persisted.get("requests") or []
        return {
            "kind": "runner",
            "version": 1,
            "meta": runner,
            "aggregate": {},
            "runner_plans": [],
            "task_definitions": [],
            "task_instances": [],
            "dispatches": [],
            "runners": [runner],
        }
    raise ValueError(
        "Unsupported JSON: expected Campaign export version 6 or Runner export version 1"
    )


def runner_information(runner: Mapping[str, Any], index: int) -> Dict[str, Any]:
    summary = runner.get("summary") or {}
    results = summary.get("results") or {}
    outcome = summary.get("outcome") or {}
    probe = summary.get("cache_probe_analysis") or {}
    benchmark = runner.get("benchmark") or {}
    task = benchmark.get("task_request") or summary.get("task_request") or {}
    cache = (probe.get("cache") or results.get("kv_cache") or {})
    first_error = get_path(outcome, "first_error.message") or runner.get("error_message")
    return {
        "order": index + 1,
        "runner_id": runner.get("runner_id"),
        "label": runner.get("label"),
        "status": runner.get("status") or "unknown",
        "outcome": outcome.get("status"),
        "provider": benchmark.get("provider"),
        "model": benchmark.get("model") or summary.get("model"),
        "mean_input_tokens": benchmark.get("mean_input_tokens") or summary.get("mean_input_tokens"),
        "mean_output_tokens": benchmark.get("mean_output_tokens") or summary.get("mean_output_tokens"),
        "concurrency": benchmark.get("concurrent_requests") or summary.get("num_concurrent_requests"),
        "task_definition_id": task.get("definition_id"),
        "task_instance_id": task.get("instance_id"),
        "node_id": task.get("node_id"),
        "role": task.get("role"),
        "payload_id": task.get("payload_id"),
        "payload_seed": integer(task.get("payload_seed")),
        "trial_index": integer(task.get("trial_index")),
        "dimensions": dict(task.get("dimensions") or {}),
        "scheduled_for": runner.get("scheduled_for") or runner.get("created_at"),
        "started_at": runner.get("started_at"),
        "finished_at": runner.get("finished_at"),
        "started": integer(results.get("num_requests_started")),
        "completed": integer(results.get("num_completed_requests")),
        "errors": integer(results.get("number_errors")),
        "error_rate": number(results.get("error_rate")),
        "ttft_p50": number(get_path(results, "ttft_s.quantiles.p50")),
        "ttft_p95": number(get_path(results, "ttft_s.quantiles.p95")),
        "e2e_p50": number(get_path(results, "end_to_end_latency_s.quantiles.p50")),
        "e2e_p95": number(get_path(results, "end_to_end_latency_s.quantiles.p95")),
        "output_tps": number(results.get("mean_output_throughput_token_per_s")),
        "cache_hit_ratio": number(cache.get("weighted_token_hit_ratio", cache.get("hit_ratio"))),
        "cache_coverage": number(cache.get("counter_coverage")),
        "cache_hit_tokens": number(cache.get("complete_hit_tokens", cache.get("hit_tokens"))),
        "cache_miss_tokens": number(cache.get("complete_miss_tokens", cache.get("miss_tokens"))),
        "cache_speedup": number(get_path(probe, "speedup.p50")),
        "paired_ttft_delta": number(get_path(probe, "paired_ttft_delta_s.p50")),
        "cache_verdict": probe.get("verdict"),
        "timed_out": bool(summary.get("timed_out")),
        "first_error": bounded_message(first_error),
        "request_records": len(runner.get("requests") or []),
    }


def build_cohorts(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("provider"), row.get("model"), row.get("mean_input_tokens"),
            row.get("mean_output_tokens"), row.get("concurrency"),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, members in groups.items():
        result.append({
            "key": "|".join(str(item) for item in key),
            "provider": key[0], "model": key[1], "mean_input_tokens": key[2],
            "mean_output_tokens": key[3], "concurrency": key[4],
            "runner_count": len(members),
            "successful_runners": sum(row.get("status") == "succeeded" for row in members),
            "ttft_p50_median": median(row.get("ttft_p50") for row in members),
            "cache_hit_ratio_median": median(row.get("cache_hit_ratio") for row in members),
            "runner_ids": [row.get("runner_id") for row in members],
        })
    return result


def build_task_evidence(
    data: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    runner_by_id = {str(row.get("runner_id")): row for row in rows if row.get("runner_id")}
    dispatches: Dict[str, List[Mapping[str, Any]]] = {}
    for dispatch in data.get("dispatches") or []:
        if isinstance(dispatch, Mapping) and dispatch.get("task_instance_id"):
            dispatches.setdefault(str(dispatch["task_instance_id"]), []).append(dispatch)
    evidence = []
    for instance in data.get("task_instances") or []:
        if not isinstance(instance, Mapping):
            continue
        instance_id = str(instance.get("task_instance_id"))
        nodes = []
        for dispatch in dispatches.get(instance_id, []):
            lineage = dispatch.get("lineage") or {}
            runner = runner_by_id.get(str(dispatch.get("runner_id")))
            nodes.append({
                "node_id": dispatch.get("node_id"),
                "role": lineage.get("role"),
                "payload_id": lineage.get("payload_id"),
                "dependencies": list(lineage.get("dependencies") or []),
                "planned_after_seconds": integer(lineage.get("after_seconds")),
                "due_at": dispatch.get("due_at"),
                "actual_started_at": lineage.get("actual_started_at"),
                "actual_completed_at": lineage.get("actual_completed_at"),
                "state": dispatch.get("state"),
                "runner": runner,
            })
        evidence.append({
            "task_instance_id": instance_id,
            "task_definition_id": instance.get("task_definition_id"),
            "instance_key": instance.get("instance_key"),
            "state": instance.get("state"),
            "dimensions": get_path(instance, "spec.dimensions", {}),
            "trial_index": get_path(instance, "spec.trial_index"),
            "payload_hashes": get_path(instance, "checkpoint.payload_hashes", {}),
            "error": bounded_message(instance.get("error")),
            "nodes": nodes,
        })
    return evidence


def prepare_analysis(document: Mapping[str, Any], source: str) -> Dict[str, Any]:
    data = normalize_document(document)
    raw = sorted(data["runners"], key=lambda item: str(item.get("created_at") or ""))
    rows = [runner_information(runner, index) for index, runner in enumerate(raw)]
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"description": source, "kind": data["kind"], "version": data["version"]},
        "meta": data["meta"],
        "overview": {
            "runner_count": len(rows),
            "status_counts": status_counts,
            "completed_requests": sum(row.get("completed") or 0 for row in rows),
            "has_failures": bool(status_counts.get("failed")),
            "campaign_aggregate": data["aggregate"],
        },
        "cohorts": build_cohorts(rows),
        "task_definitions": data["task_definitions"],
        "evidence": {
            "task_graphs": build_task_evidence(data, rows),
            "runner_cache_probes": [row for row in rows if row.get("cache_verdict") is not None],
        },
        "runners": rows,
    }


def find_llmperfctl(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("llmperfctl")
    if discovered:
        return discovered
    local = Path.cwd() / ".venv" / "bin" / "llmperfctl"
    if local.is_file():
        return str(local)
    raise RuntimeError("llmperfctl not found; pass --llmperfctl PATH")


def export_document(arguments: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    if arguments.input:
        path = Path(arguments.input).expanduser().resolve()
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    if any(item.startswith(("--token", "--private-key")) for item in arguments.llmperfctl_arg):
        raise ValueError("Do not pass credentials through --llmperfctl-arg")
    cli = find_llmperfctl(arguments.llmperfctl)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temporary:
        export_path = Path(temporary.name)
    try:
        command = [cli, *arguments.llmperfctl_arg]
        if arguments.campaign_id:
            command += ["campaign", "export", arguments.campaign_id]
            if arguments.include_requests:
                command.append("--include-requests")
        else:
            command += ["runner", "export", arguments.runner_id]
        command += ["-o", str(export_path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"llmperfctl export failed ({completed.returncode}): "
                f"{bounded_message(completed.stderr or completed.stdout, 500)}"
            )
        document = json.loads(export_path.read_text(encoding="utf-8"))
        if arguments.keep_json:
            keep = Path(arguments.keep_json).expanduser().resolve()
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(export_path, keep)
        source = f"llmperfctl {'campaign' if arguments.campaign_id else 'runner'} export"
        return document, source
    finally:
        export_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-id")
    source.add_argument("--runner-id")
    source.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-requests", action="store_true")
    parser.add_argument("--keep-json")
    parser.add_argument("--llmperfctl")
    parser.add_argument("--llmperfctl-arg", action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        document, source = export_document(arguments)
        analysis = prepare_analysis(document, source)
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a self-contained HTML report from LLMPerf exports."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "violet": "#7c3aed",
    "slate": "#64748b",
}


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
    value = float(value)
    return value if math.isfinite(value) else None


def integer(value: Any) -> Optional[int]:
    value = number(value)
    return int(value) if value is not None else None


def median(values: Iterable[Any]) -> Optional[float]:
    clean = [item for value in values if (item := number(value)) is not None]
    return float(statistics.median(clean)) if clean else None


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    value = number(value)
    if value is None:
        return "—"
    rendered = f"{value:,.{digits}f}"
    return f"{rendered}{suffix}"


def fmt_int(value: Any) -> str:
    value = integer(value)
    return f"{value:,}" if value is not None else "—"


def fmt_pct(value: Any, digits: int = 1) -> str:
    value = number(value)
    return fmt(value * 100, digits, "%") if value is not None else "—"


def fmt_seconds(value: Any) -> str:
    return fmt(value, 3, " s")


def parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def compact_time(value: Any) -> str:
    parsed = parse_time(value)
    if parsed is None:
        return "—"
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def duration_seconds(started: Any, finished: Any) -> Optional[float]:
    start = parse_time(started)
    end = parse_time(finished)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def bounded_message(value: Any, limit: int = 280) -> str:
    if not isinstance(value, str) or not value.strip():
        return "—"
    text = re.sub(r"https?://[^\s)'\"]+", "[endpoint redacted]", value)
    text = re.sub(
        r"(host\s*=\s*)['\"][^'\"]+['\"]",
        r"\1'[host redacted]'",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(with\s+url\s*:)\s*\S+",
        r"\1 [path redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def classify_diagnostic(runner: Mapping[str, Any], info: Mapping[str, Any]) -> str:
    stderr = runner.get("stderr") if runner.get("status") == "failed" else ""
    combined = " ".join(
        str(value or "")
        for value in (
            runner.get("error_message"),
            info.get("first_error"),
            stderr,
        )
    )
    lowered = combined.lower()
    if "outofmemoryerror" in lowered or "running low on memory" in lowered:
        return "基础设施：Ray/节点内存不足"
    if "proxyerror" in lowered or "cannot connect to proxy" in lowered:
        return "网络：代理连接失败"
    if "ssl" in lowered or "tls" in lowered:
        return "网络：TLS/SSL 连接失败"
    if "timeout" in lowered or "timed out" in lowered:
        return "超时：请求或实验超过期限"
    if "429" in lowered or "rate limit" in lowered:
        return "Provider：限流"
    if "401" in lowered or "unauthorized" in lowered:
        return "Provider：认证失败"
    if runner.get("status") == "cancelled":
        return "控制面：Runner 已取消"
    if info.get("timed_out"):
        return "实验：达到 timeout_seconds，结果可能不完整"
    message = info.get("first_error") or runner.get("error_message")
    return bounded_message(message) if message else "未记录明确原因"


def normalize_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    if document.get("version") == 5 and isinstance(document.get("runners"), list):
        return {
            "kind": "campaign",
            "version": document["version"],
            "meta": dict(document.get("campaign") or {}),
            "aggregate": dict(document.get("aggregate") or {}),
            "runner_plans": list(document.get("runner_plans") or []),
            "protocol_definitions": list(
                document.get("protocol_definitions") or []
            ),
            "protocol_instances": list(document.get("protocol_instances") or []),
            "dispatches": list(document.get("dispatches") or []),
            "protocol_analyses": list(document.get("protocol_analyses") or []),
            "runners": [dict(item) for item in document["runners"] if isinstance(item, Mapping)],
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
            "runners": [runner],
        }
    if "runner_id" in document and "summary" in document:
        runner = dict(document)
        return {
            "kind": "runner",
            "version": "status",
            "meta": runner,
            "aggregate": {},
            "runner_plans": [],
            "runners": [runner],
        }
    raise ValueError(
        "Unsupported JSON: expected Campaign export version 5 or Runner export version 1"
    )


def runner_information(runner: Mapping[str, Any], index: int) -> Dict[str, Any]:
    summary = runner.get("summary") or {}
    results = summary.get("results") or {}
    outcome = summary.get("outcome") or {}
    analysis = summary.get("cache_probe_analysis") or {}
    cache = analysis.get("cache") or {}
    quality = analysis.get("quality_flags") or {}
    benchmark = runner.get("benchmark") or {}
    occurrence = runner.get("plan_occurrence")
    round_number = occurrence + 1 if isinstance(occurrence, int) else index + 1
    first_error = get_path(outcome, "first_error.message") or runner.get("error_message")
    info = {
        "round": round_number,
        "runner_id": runner.get("runner_id"),
        "status": runner.get("status") or "unknown",
        "outcome": outcome.get("status"),
        "provider": benchmark.get("provider"),
        "model": benchmark.get("model") or summary.get("model"),
        "concurrency": benchmark.get("concurrent_requests")
        or summary.get("num_concurrent_requests"),
        "mean_input_tokens": benchmark.get("mean_input_tokens")
        or summary.get("mean_input_tokens"),
        "mean_output_tokens": benchmark.get("mean_output_tokens")
        or summary.get("mean_output_tokens"),
        "cache_mode": get_path(benchmark, "cache_probe.mode")
        or get_path(summary, "cache_probe.mode"),
        "scheduled_for": runner.get("scheduled_for") or runner.get("created_at"),
        "started_at": runner.get("started_at"),
        "finished_at": runner.get("finished_at"),
        "duration_s": duration_seconds(runner.get("started_at"), runner.get("finished_at")),
        "started": integer(results.get("num_requests_started")),
        "completed": integer(results.get("num_completed_requests")),
        "errors": integer(results.get("number_errors")),
        "error_rate": number(results.get("error_rate")),
        "ttft_p50": number(get_path(results, "ttft_s.quantiles.p50")),
        "ttft_p95": number(get_path(results, "ttft_s.quantiles.p95")),
        "e2e_p50": number(get_path(results, "end_to_end_latency_s.quantiles.p50")),
        "e2e_p95": number(get_path(results, "end_to_end_latency_s.quantiles.p95")),
        "output_tps": number(results.get("mean_output_throughput_token_per_s")),
        "request_tps_p50": number(
            get_path(results, "request_output_throughput_token_per_s.quantiles.p50")
        ),
        "warm_hit_ratio": number(
            cache.get("weighted_token_hit_ratio", cache.get("hit_ratio"))
        ),
        "cache_coverage": number(cache.get("counter_coverage")),
        "cache_hit_tokens": number(
            cache.get("complete_hit_tokens", cache.get("hit_tokens"))
        ),
        "cache_miss_tokens": number(
            cache.get("complete_miss_tokens", cache.get("miss_tokens"))
        ),
        "cache_speedup": number(get_path(analysis, "speedup.p50")),
        "paired_delta": number(get_path(analysis, "paired_ttft_delta_s.p50")),
        "paired_samples": integer(analysis.get("paired_samples")),
        "cache_verdict": analysis.get("verdict"),
        "timed_out": bool(summary.get("timed_out") or quality.get("timed_out")),
        "skipped_dependencies": integer(quality.get("skipped_dependency_requests")),
        "tokenizer_mismatches": integer(quality.get("tokenizer_mismatch_requests")),
        "first_error": bounded_message(first_error),
        "request_records": len(runner.get("requests") or []),
    }
    info["diagnostic"] = classify_diagnostic(runner, info)
    return info


def ordered_runners(runners: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        runners,
        key=lambda item: (
            str(item.get("scheduled_for") or item.get("created_at") or ""),
            str(item.get("runner_id") or ""),
        ),
    )


def sum_available(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[int]:
    values = [integer(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return sum(clean) if clean else None


def compute_overview(data: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in rows)
    aggregate = data.get("aggregate") or {}
    started = sum_available(rows, "started")
    completed = sum_available(rows, "completed")
    errors = sum_available(rows, "errors")
    hit = sum(number(row.get("cache_hit_tokens")) or 0 for row in rows)
    miss = sum(number(row.get("cache_miss_tokens")) or 0 for row in rows)
    weighted_hit_ratio = hit / (hit + miss) if hit + miss > 0 else None
    cohorts = {
        (
            row.get("provider"),
            row.get("model"),
            row.get("concurrency"),
            row.get("mean_input_tokens"),
            row.get("mean_output_tokens"),
            row.get("cache_mode"),
        )
        for row in rows
    }
    return {
        "status": aggregate.get("status") or (rows[0]["status"] if len(rows) == 1 else "—"),
        "outcome": aggregate.get("outcome") or (rows[0].get("outcome") if len(rows) == 1 else "—"),
        "status_counts": status_counts,
        "runner_count": len(rows),
        "started": started,
        "completed": completed,
        "errors": errors,
        "reliability": completed / started if started and completed is not None else None,
        "ttft_p50_median": median(row.get("ttft_p50") for row in rows),
        "ttft_p95_median": median(row.get("ttft_p95") for row in rows),
        "output_tps_median": median(row.get("output_tps") for row in rows),
        "weighted_hit_ratio": weighted_hit_ratio,
        "cache_coverage_median": median(row.get("cache_coverage") for row in rows),
        "cache_speedup_median": median(row.get("cache_speedup") for row in rows),
        "timed_out_count": sum(bool(row.get("timed_out")) for row in rows),
        "degraded_count": sum(row.get("outcome") == "degraded" for row in rows),
        "request_records": sum(integer(row.get("request_records")) or 0 for row in rows),
        "cohort_count": len(cohorts),
        "verdict_counts": Counter(
            str(row["cache_verdict"]) for row in rows if row.get("cache_verdict")
        ),
    }


def status_class(status: Any) -> str:
    status = str(status or "unknown")
    if status in {"succeeded", "completed"}:
        return "good"
    if status in {"failed", "cancelled", "partial_failed"}:
        return "bad"
    if status in {"degraded", "running", "paused"}:
        return "warn"
    return "neutral"


def badge(value: Any) -> str:
    text = str(value or "—")
    return f'<span class="badge {status_class(text)}">{escape(text)}</span>'


def svg_frame(title: str, inner: str, description: str = "") -> str:
    return (
        f'<figure class="chart-card"><figcaption>{escape(title)}</figcaption>'
        f'<div class="chart-note">{escape(description)}</div>{inner}</figure>'
    )


def svg_line_chart(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    series: Sequence[Tuple[str, str, str]],
    value_formatter=fmt,
    baseline: Optional[float] = None,
    description: str = "",
) -> str:
    width, height = 760, 286
    left, right, top, bottom = 62, 20, 34, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [
        value
        for row in rows
        for _, key, _ in series
        if (value := number(row.get(key))) is not None
    ]
    if baseline is not None:
        values.append(baseline)
    if not values:
        return svg_frame(title, '<div class="empty">无可用指标</div>', description)
    y_min = min(0.0, min(values))
    y_max = max(values)
    if y_max == y_min:
        y_max = y_min + 1.0
    padding = (y_max - y_min) * 0.08
    y_max += padding
    if y_min < 0:
        y_min -= padding

    def x_pos(index: int) -> float:
        return left + (plot_w / max(1, len(rows) - 1)) * index

    def y_pos(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = y_pos(value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
            f'<text class="axis" x="{left-8}" y="{y+4:.1f}" text-anchor="end">'
            f'{escape(value_formatter(value))}</text>'
        )
    if baseline is not None and y_min <= baseline <= y_max:
        y = y_pos(baseline)
        parts.append(
            f'<line class="baseline" x1="{left}" y1="{y:.1f}" '
            f'x2="{width-right}" y2="{y:.1f}"/>'
        )
    for index, row in enumerate(rows):
        x = x_pos(index)
        parts.append(
            f'<text class="axis" x="{x:.1f}" y="{height-17}" text-anchor="middle">'
            f'R{escape(str(row.get("round", index + 1)))}</text>'
        )
    for label, key, color in series:
        segments: List[List[Tuple[float, float, float, int]]] = []
        current: List[Tuple[float, float, float, int]] = []
        for index, row in enumerate(rows):
            value = number(row.get(key))
            if value is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append((x_pos(index), y_pos(value), value, index))
        if current:
            segments.append(current)
        for segment in segments:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in segment)
            parts.append(
                f'<polyline class="series" stroke="{color}" points="{points}"/>'
            )
            for x, y, value, index in segment:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
                    f'<title>R{rows[index].get("round")}: {label} {value_formatter(value)}</title>'
                    "</circle>"
                )
    legend_x = left
    for label, _, color in series:
        parts.append(
            f'<circle cx="{legend_x+5}" cy="14" r="4" fill="{color}"/>'
            f'<text class="legend" x="{legend_x+14}" y="18">{escape(label)}</text>'
        )
        legend_x += max(112, len(label) * 9 + 34)
    parts.append("</svg>")
    return svg_frame(title, "".join(parts), description)


def svg_bar_chart(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    series: Sequence[Tuple[str, str, str]],
    value_formatter=fmt,
    max_value: Optional[float] = None,
    description: str = "",
) -> str:
    width, height = 760, 286
    left, right, top, bottom = 62, 20, 34, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [
        value
        for row in rows
        for _, key, _ in series
        if (value := number(row.get(key))) is not None
    ]
    if not values:
        return svg_frame(title, '<div class="empty">无可用指标</div>', description)
    observed_max = max(values)
    if max_value is not None and observed_max <= max_value:
        y_max = max_value
    else:
        y_max = (observed_max or 1) * 1.08
    group_w = plot_w / max(1, len(rows))
    bar_w = min(28, group_w * 0.7 / max(1, len(series)))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
    ]
    for tick in range(5):
        value = y_max * tick / 4
        y = top + plot_h - value / y_max * plot_h
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
            f'<text class="axis" x="{left-8}" y="{y+4:.1f}" text-anchor="end">'
            f'{escape(value_formatter(value))}</text>'
        )
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        total_w = bar_w * len(series)
        for series_index, (label, key, color) in enumerate(series):
            value = number(row.get(key))
            if value is None:
                continue
            bar_h = value / y_max * plot_h
            x = center - total_w / 2 + series_index * bar_w
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, bar_w-2):.1f}" '
                f'height="{bar_h:.1f}" rx="3" fill="{color}">'
                f'<title>R{row.get("round")}: {label} {value_formatter(value)}</title></rect>'
            )
        parts.append(
            f'<text class="axis" x="{center:.1f}" y="{height-17}" text-anchor="middle">'
            f'R{escape(str(row.get("round", index + 1)))}</text>'
        )
    legend_x = left
    for label, _, color in series:
        parts.append(
            f'<rect x="{legend_x}" y="9" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text class="legend" x="{legend_x+15}" y="18">{escape(label)}</text>'
        )
        legend_x += max(112, len(label) * 9 + 34)
    parts.append("</svg>")
    return svg_frame(title, "".join(parts), description)


def svg_status_donut(counts: Mapping[str, int]) -> str:
    ordered = [
        ("succeeded", COLORS["green"]),
        ("failed", COLORS["red"]),
        ("running", COLORS["blue"]),
        ("queued", COLORS["amber"]),
        ("cancelled", COLORS["slate"]),
    ]
    total = sum(int(counts.get(status, 0)) for status, _ in ordered)
    if total == 0:
        return svg_frame("Runner 状态分布", '<div class="empty">无 Runner</div>')
    radius = 66
    circumference = 2 * math.pi * radius
    offset = 0.0
    parts = ['<svg viewBox="0 0 420 286" role="img" aria-label="Runner 状态分布">']
    for status, color in ordered:
        count = int(counts.get(status, 0))
        if count <= 0:
            continue
        length = circumference * count / total
        parts.append(
            f'<circle cx="140" cy="142" r="{radius}" fill="none" stroke="{color}" '
            f'stroke-width="24" stroke-dasharray="{length:.2f} {circumference-length:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 140 142)">'
            f'<title>{status}: {count}</title></circle>'
        )
        offset += length
    parts.append(
        f'<text x="140" y="137" text-anchor="middle" class="donut-total">{total}</text>'
        '<text x="140" y="158" text-anchor="middle" class="axis">Runners</text>'
    )
    y = 78
    for status, color in ordered:
        count = int(counts.get(status, 0))
        parts.append(
            f'<circle cx="270" cy="{y}" r="5" fill="{color}"/>'
            f'<text class="legend" x="283" y="{y+4}">{status}: {count}</text>'
        )
        y += 30
    parts.append("</svg>")
    return svg_frame("Runner 状态分布", "".join(parts), "生命周期终态与执行成功并非同一概念")


def executive_findings(overview: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    status, outcome = overview.get("status"), overview.get("outcome")
    severity = "bad" if outcome in {"failed", "partial_failed", "cancelled"} else "good"
    findings.append(
        (
            severity,
            f"实验生命周期为 {status or '—'}，聚合执行结果为 {outcome or '—'}；"
            f"{overview['runner_count']} 个 Runner 中 "
            f"{overview['status_counts'].get('succeeded', 0)} 个成功、"
            f"{overview['status_counts'].get('failed', 0)} 个失败。",
        )
    )
    if overview.get("started") is not None:
        findings.append(
            (
                "good" if (overview.get("reliability") or 0) >= 0.99 else "warn",
                f"共启动 {fmt_int(overview['started'])} 个请求，完成 "
                f"{fmt_int(overview['completed'])} 个，记录错误 {fmt_int(overview['errors'])} 个；"
                f"请求完成率为 {fmt_pct(overview.get('reliability'))}。",
            )
        )
    verdicts = overview.get("verdict_counts") or {}
    if verdicts:
        if verdicts.get("confirmed_external"):
            cache_text = "至少一轮同时观察到缓存命中与统计支持的配对延迟改善"
            severity = "good"
        elif verdicts.get("accounting_confirmed"):
            cache_text = "Provider 计数确认了缓存复用，但未建立稳定的配对延迟收益"
            severity = "warn"
        else:
            cache_text = "缓存证据尚不足以确认外部加速"
            severity = "warn"
        findings.append(
            (
                severity,
                f"Warm KV Cache 加权 Token 命中率为 {fmt_pct(overview.get('weighted_hit_ratio'))}，"
                f"计数覆盖率跨轮中位数为 {fmt_pct(overview.get('cache_coverage_median'))}；{cache_text}。",
            )
        )
    quality_items = []
    if overview.get("timed_out_count"):
        quality_items.append(f"{overview['timed_out_count']} 轮超时")
    if overview.get("degraded_count"):
        quality_items.append(f"{overview['degraded_count']} 轮请求结果 degraded")
    if overview.get("cohort_count", 1) > 1:
        quality_items.append(f"存在 {overview['cohort_count']} 个配置 cohort，不应直接混合比较")
    if quality_items:
        findings.append(("warn", "数据质量提示：" + "；".join(quality_items) + "。"))
    ttft_values = [number(row.get("ttft_p50")) for row in rows]
    ttft_values = [value for value in ttft_values if value is not None]
    if ttft_values:
        findings.append(
            (
                "neutral",
                f"各轮 TTFT P50 范围为 {fmt_seconds(min(ttft_values))}–"
                f"{fmt_seconds(max(ttft_values))}，跨轮中位数为 "
                f"{fmt_seconds(overview.get('ttft_p50_median'))}。",
            )
        )
    return findings


def render_kpi(label: str, value: str, note: str, tone: str = "blue") -> str:
    return (
        f'<div class="kpi {escape(tone)}"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div><div class="kpi-note">{escape(note)}</div></div>'
    )


def render_runner_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f'<td class="mono">R{escape(str(row["round"]))}</td>'
            f'<td>{badge(row.get("status"))}</td>'
            f'<td class="mono id" title="{escape(str(row.get("runner_id") or ""))}">'
            f'{escape(str(row.get("runner_id") or "—"))}</td>'
            f'<td>{escape(str(row.get("provider") or "—"))}/'
            f'{escape(str(row.get("model") or "—"))}</td>'
            f'<td>{fmt_int(row.get("started"))}/{fmt_int(row.get("completed"))}/'
            f'{fmt_int(row.get("errors"))}</td>'
            f'<td>{fmt_seconds(row.get("ttft_p50"))}</td>'
            f'<td>{fmt_seconds(row.get("ttft_p95"))}</td>'
            f'<td>{fmt(row.get("output_tps"), 2, " tok/s")}</td>'
            f'<td>{fmt_pct(row.get("warm_hit_ratio"))}</td>'
            f'<td>{fmt_pct(row.get("cache_coverage"))}</td>'
            f'<td>{fmt(row.get("cache_speedup"), 3, "×")}</td>'
            f'<td>{badge(row.get("cache_verdict")) if row.get("cache_verdict") else "—"}</td>'
            f'<td>{"是" if row.get("timed_out") else "否"}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>轮次</th><th>Runner 状态</th><th>Runner ID</th><th>Provider/Model</th>"
        "<th>请求 S/C/E</th><th>TTFT P50</th><th>TTFT P95</th><th>输出吞吐</th>"
        "<th>Warm 命中率</th><th>计数覆盖</th><th>Cache Speedup</th><th>证据结论</th><th>超时</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def render_diagnostics(rows: Sequence[Mapping[str, Any]]) -> str:
    issues = [
        row
        for row in rows
        if row.get("status") in {"failed", "cancelled"}
        or row.get("outcome") == "degraded"
        or (row.get("errors") or 0) > 0
        or row.get("timed_out")
    ]
    if not issues:
        return '<div class="notice good">未发现失败、请求错误或超时标记。</div>'
    body = []
    for row in issues:
        flags = []
        if row.get("outcome") == "degraded":
            flags.append("degraded")
        if row.get("timed_out"):
            flags.append("timed_out")
        if row.get("errors"):
            flags.append(f"errors={row['errors']}")
        body.append(
            "<tr>"
            f'<td>R{escape(str(row["round"]))}</td><td>{badge(row.get("status"))}</td>'
            f'<td class="mono id">{escape(str(row.get("runner_id") or "—"))}</td>'
            f'<td>{escape(", ".join(flags) or "—")}</td>'
            f'<td>{escape(str(row.get("diagnostic") or "—"))}</td>'
            f'<td>{escape(str(row.get("first_error") or "—"))}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>轮次</th><th>状态</th>'
        "<th>Runner ID</th><th>质量标记</th><th>诊断分类</th><th>首个错误摘要</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def report_html(document: Mapping[str, Any], source: str, custom_title: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    data = normalize_document(document)
    raw_runners = ordered_runners(data["runners"])
    rows = [runner_information(runner, index) for index, runner in enumerate(raw_runners)]
    overview = compute_overview(data, rows)
    meta = data["meta"]
    report_id = meta.get("campaign_id") or meta.get("runner_id") or "unknown"
    name = meta.get("name") or meta.get("label") or report_id
    title = custom_title or f"LLMPerf 实验分析报告 · {name}"
    providers = sorted({str(row["provider"]) for row in rows if row.get("provider")})
    models = sorted({str(row["model"]) for row in rows if row.get("model")})
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    findings = executive_findings(overview, rows)
    finding_html = "".join(
        f'<li class="finding {severity}"><span></span>{escape(text)}</li>'
        for severity, text in findings
    )
    charts = [
        svg_status_donut(overview["status_counts"]),
        svg_bar_chart(
            "请求完成与错误",
            rows,
            (
                ("Completed", "completed", COLORS["green"]),
                ("Errors", "errors", COLORS["red"]),
            ),
            value_formatter=lambda value: fmt(value, 0),
            description="每轮 summary 中的完成请求与错误请求；不将错误计入成功",
        ),
        svg_line_chart(
            "TTFT 跨轮趋势",
            rows,
            (
                ("P50", "ttft_p50", COLORS["blue"]),
                ("P95", "ttft_p95", COLORS["violet"]),
            ),
            value_formatter=lambda value: fmt(value, 2, "s"),
            description="每个点是单个 Runner 的请求级分位数，未跨轮池化请求",
        ),
        svg_bar_chart(
            "输出吞吐",
            rows,
            (("Overall output TPS", "output_tps", COLORS["cyan"]),),
            value_formatter=lambda value: fmt(value, 2),
            description="mean_output_throughput_token_per_s",
        ),
        svg_bar_chart(
            "Warm KV Cache Token 命中率",
            rows,
            (("Warm hit ratio", "warm_hit_ratio", COLORS["green"]),),
            value_formatter=lambda value: fmt_pct(value),
            max_value=1.0,
            description="按 complete warm hit/miss tokens 加权；同时检查计数覆盖率",
        ),
        svg_line_chart(
            "配对 Cache Speedup",
            rows,
            (("Prime/Warm TTFT", "cache_speedup", COLORS["amber"]),),
            value_formatter=lambda value: fmt(value, 2, "×"),
            baseline=1.0,
            description="1× 为无加速基线；是否确认加速仍以 verdict 与置信区间为准",
        ),
    ]
    css = """
:root{--ink:#0f172a;--muted:#64748b;--line:#dbe4f0;--paper:#fff;--bg:#f4f7fb;
--blue:#2563eb;--green:#059669;--amber:#d97706;--red:#dc2626;--violet:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,
ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.page{max-width:1480px;margin:auto;padding:34px}.hero{background:linear-gradient(135deg,#0f172a,#172554 58%,#1e3a8a);
color:white;border-radius:22px;padding:34px 38px;box-shadow:0 20px 55px #17255422}.eyebrow{font-size:12px;
letter-spacing:.16em;text-transform:uppercase;color:#bfdbfe;font-weight:750}.hero h1{font-size:30px;line-height:1.2;
margin:10px 0 14px}.hero-meta{display:flex;gap:20px;flex-wrap:wrap;color:#dbeafe}.hero-meta b{color:white}
.section{margin-top:28px}.section h2{font-size:20px;margin:0 0 14px}.section-sub{color:var(--muted);margin:-8px 0 16px}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(165px,1fr));gap:13px;margin-top:-18px;padding:0 18px}
.kpi{background:white;border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:14px;padding:16px;
box-shadow:0 10px 25px #3341550c}.kpi.green{border-top-color:var(--green)}.kpi.amber{border-top-color:var(--amber)}
.kpi.violet{border-top-color:var(--violet)}.kpi-label{font-size:12px;color:var(--muted);font-weight:700;text-transform:uppercase}
.kpi-value{font-size:24px;font-weight:800;margin:3px 0}.kpi-note{font-size:12px;color:var(--muted)}
.panel{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 8px 24px #3341550a}
.findings{list-style:none;padding:0;margin:0;display:grid;gap:10px}.finding{padding:12px 14px;border-radius:10px;background:#f8fafc;
display:flex;gap:10px}.finding span{width:8px;height:8px;border-radius:99px;background:#64748b;margin-top:7px;flex:none}
.finding.good span{background:var(--green)}.finding.warn span{background:var(--amber)}.finding.bad span{background:var(--red)}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.chart-card{margin:0;background:white;border:1px solid var(--line);
border-radius:16px;padding:16px;min-width:0;box-shadow:0 8px 24px #3341550a}.chart-card figcaption{font-weight:800;font-size:15px}
.chart-note{color:var(--muted);font-size:12px;min-height:19px}.chart-card svg{display:block;width:100%;height:auto;max-height:315px}
.grid{stroke:#e5edf6;stroke-width:1}.axis,.legend{fill:#64748b;font-size:11px}.series{fill:none;stroke-width:2.5;stroke-linejoin:round;
stroke-linecap:round}.baseline{stroke:#94a3b8;stroke-width:1.3;stroke-dasharray:5 5}.donut-total{font-size:29px;font-weight:800;fill:#0f172a}
.empty{height:240px;display:grid;place-items:center;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);
border-radius:14px;background:white}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px}th{position:sticky;top:0;
background:#f1f5f9;text-align:left;color:#475569;font-weight:800;padding:11px;border-bottom:1px solid var(--line)}td{padding:10px 11px;
border-bottom:1px solid #edf2f7}tbody tr:hover{background:#f8fafc}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.id{max-width:235px;overflow:hidden;text-overflow:ellipsis}
.badge{display:inline-flex;border-radius:99px;padding:2px 8px;font-size:11px;font-weight:750;background:#e2e8f0;color:#334155}
.badge.good{background:#d1fae5;color:#047857}.badge.warn{background:#fef3c7;color:#b45309}.badge.bad{background:#fee2e2;color:#b91c1c}
.notice{padding:13px;border-radius:10px;background:#ecfdf5;color:#047857}.method{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.method div{background:white;border:1px solid var(--line);border-radius:12px;padding:14px}.method b{display:block;font-size:12px;color:var(--muted)}
.footer{color:var(--muted);font-size:12px;margin:28px 0 8px;text-align:center}@media(max-width:1050px){.kpis{grid-template-columns:repeat(3,1fr)}
.charts{grid-template-columns:1fr}}@media(max-width:680px){.page{padding:16px}.hero{padding:24px}.hero h1{font-size:24px}.kpis{grid-template-columns:repeat(2,1fr)}
.method{grid-template-columns:1fr}}@media print{body{background:white}.page{max-width:none;padding:0}.hero,.panel,.chart-card,.kpi{box-shadow:none}
.chart-card{break-inside:avoid}.table-wrap{overflow:visible}th{position:static}}
"""
    kpis = "".join(
        [
            render_kpi(
                "Runner 结果",
                f"{overview['status_counts'].get('succeeded', 0)}/{overview['runner_count']}",
                f"Outcome: {overview.get('outcome') or '—'}",
                "green" if overview["status_counts"].get("failed", 0) == 0 else "amber",
            ),
            render_kpi(
                "请求完成率",
                fmt_pct(overview.get("reliability")),
                f"{fmt_int(overview.get('completed'))}/{fmt_int(overview.get('started'))} completed",
                "green" if (overview.get("reliability") or 0) >= 0.99 else "amber",
            ),
            render_kpi(
                "TTFT P50",
                fmt_seconds(overview.get("ttft_p50_median")),
                "跨 Runner 中位数",
                "blue",
            ),
            render_kpi(
                "输出吞吐",
                fmt(overview.get("output_tps_median"), 2, " tok/s"),
                "跨 Runner 中位数",
                "blue",
            ),
            render_kpi(
                "Warm Cache 命中",
                fmt_pct(overview.get("weighted_hit_ratio")),
                "完整 Token 计数加权",
                "green",
            ),
            render_kpi(
                "Cache Speedup",
                fmt(overview.get("cache_speedup_median"), 3, "×"),
                "Prime/Warm TTFT 中位数",
                "violet",
            ),
        ]
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{css}</style></head><body><main class="page">
<header class="hero"><div class="eyebrow">LLMPerf · Reproducible Benchmark Report</div><h1>{escape(title)}</h1>
<div class="hero-meta"><span><b>ID</b> {escape(str(report_id))}</span><span><b>Provider</b> {escape(', '.join(providers) or '—')}</span>
<span><b>Model</b> {escape(', '.join(models) or '—')}</span><span><b>Generated</b> {generated}</span></div></header>
<section class="kpis">{kpis}</section>
<section class="section"><h2>执行摘要</h2><div class="panel"><ul class="findings">{finding_html}</ul></div></section>
<section class="section"><h2>趋势与证据</h2><p class="section-sub">图表按 Runner 时间顺序展示；缺失指标保留为空，不补零。</p>
<div class="charts">{''.join(charts)}</div></section>
<section class="section"><h2>Runner 明细</h2><p class="section-sub">S/C/E = started/completed/errors。</p>{render_runner_table(rows)}</section>
<section class="section"><h2>数据质量与失败诊断</h2><p class="section-sub">基础设施与网络问题不应直接归因于模型性能。</p>{render_diagnostics(rows)}</section>
<section class="section"><h2>方法与溯源</h2><div class="method">
<div><b>数据来源</b>{escape(source)}</div><div><b>导出契约</b>{escape(str(data['kind']))} schema {escape(str(data['version']))}</div>
<div><b>时间范围</b>{escape(compact_time(rows[0].get('scheduled_for') if rows else None))} → {escape(compact_time(rows[-1].get('finished_at') if rows else None))}</div>
<div><b>请求级记录</b>{overview['request_records']:,} 条（0 表示使用 summary-only Campaign 导出）</div>
<div><b>比较 Cohort</b>{overview['cohort_count']}（Provider/Model/并发/Token/Cache Mode）</div>
<div><b>隐私边界</b>未渲染 stdout、stderr、Prompt 文本、凭据或完整私有 Endpoint</div>
</div></section><footer class="footer">Generated by the project skill <span class="mono">$generate-llmperf-report</span> · {generated}</footer>
</main></body></html>"""
    return html, overview


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
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), str(path)
    forbidden = ("--token", "--private-key")
    if any(
        item == option or item.startswith(option + "=")
        for item in arguments.llmperfctl_arg
        for option in forbidden
    ):
        raise ValueError(
            "Do not pass credentials through --llmperfctl-arg; use environment or key discovery"
        )
    cli = find_llmperfctl(arguments.llmperfctl)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temporary:
        export_path = Path(temporary.name)
    try:
        command = [cli] + list(arguments.llmperfctl_arg)
        if arguments.campaign_id:
            command += ["campaign", "export", arguments.campaign_id]
            if arguments.include_requests:
                command.append("--include-requests")
        else:
            command += ["runner", "export", arguments.runner_id]
        command += ["-o", str(export_path)]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = bounded_message(completed.stderr or completed.stdout, 500)
            raise RuntimeError(f"llmperfctl export failed ({completed.returncode}): {detail}")
        with export_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if arguments.keep_json:
            keep = Path(arguments.keep_json).expanduser().resolve()
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(export_path, keep)
        source = (
            f"llmperfctl campaign export {arguments.campaign_id}"
            if arguments.campaign_id
            else f"llmperfctl runner export {arguments.runner_id}"
        )
        return document, source
    finally:
        export_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a professional self-contained HTML report from LLMPerf records"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-id", help="Export and report one Campaign")
    source.add_argument("--runner-id", help="Export and report one Runner")
    source.add_argument("--input", help="Read an existing Campaign/Runner export JSON")
    parser.add_argument("--output", required=True, help="Destination HTML file")
    parser.add_argument("--title", help="Override the report title")
    parser.add_argument(
        "--include-requests",
        action="store_true",
        help="Include Campaign request records in the export (larger report source)",
    )
    parser.add_argument("--keep-json", help="Keep the intermediate llmperfctl export JSON")
    parser.add_argument("--llmperfctl", help="Path to llmperfctl executable")
    parser.add_argument(
        "--llmperfctl-arg",
        action="append",
        default=[],
        help="Repeat for safe global llmperfctl options such as --url and its value",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.include_requests and not arguments.campaign_id:
            raise ValueError("--include-requests is only valid with --campaign-id")
        document, source = export_document(arguments)
        html, overview = report_html(document, source, arguments.title)
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "bytes": output.stat().st_size,
                    "runner_count": overview["runner_count"],
                    "requests_started": overview.get("started"),
                    "requests_completed": overview.get("completed"),
                    "request_errors": overview.get("errors"),
                    "timed_out_runners": overview.get("timed_out_count"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

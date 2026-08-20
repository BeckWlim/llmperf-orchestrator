#!/usr/bin/env python3
"""Inventory report evidence and review an evidence-bound render plan and HTML."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from llmperf.version import PROTOCOL_VERSION

RENDER_PLAN_VERSION = PROTOCOL_VERSION
ANALYSIS_VERSION = PROTOCOL_VERSION
EVIDENCE_POLICY = "normalized-analysis-only"
VISUAL_PRIORITIES = {"primary", "supporting"}
PATH_PART = re.compile(r"^(?P<key>[^.\[\]]+)(?P<array>\[\])?$")


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def collect_inventory(document: Mapping[str, Any], digest: str) -> Dict[str, Any]:
    observations: Dict[str, List[Any]] = defaultdict(list)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if not value and path:
                observations[path].append(value)
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, list):
            array_path = f"{path}[]"
            if not value:
                observations[array_path].append(value)
            for child in value:
                visit(child, array_path)
            return
        observations[path].append(value)

    visit(document, "")
    fields = []
    for path in sorted(observations):
        values = observations[path]
        non_null = [value for value in values if value is not None]
        distinct = {
            json.dumps(value, sort_keys=True, ensure_ascii=False) for value in non_null
        }
        numeric = [
            float(value)
            for value in non_null
            if not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ]
        field = {
            "path": path,
            "types": sorted({scalar_kind(value) for value in values}),
            "observations": len(values),
            "nulls": sum(value is None for value in values),
            "distinct_non_null": len(distinct),
        }
        if numeric:
            field["numeric_range"] = {"min": min(numeric), "max": max(numeric)}
        fields.append(field)
    return {
        "render_review_version": RENDER_PLAN_VERSION,
        "analysis_sha256": digest,
        "analysis_version": document.get("analysis_version"),
        "top_level_fields": sorted(document),
        "fields": fields,
    }


def resolve_path(document: Any, path: str) -> Tuple[bool, int]:
    if not isinstance(path, str) or not path:
        return False, 0
    current = [document]
    for raw_part in path.split("."):
        match = PATH_PART.fullmatch(raw_part)
        if match is None:
            return False, 0
        following = []
        for value in current:
            if not isinstance(value, Mapping) or match.group("key") not in value:
                continue
            child = value[match.group("key")]
            if match.group("array"):
                if isinstance(child, list):
                    following.extend(child)
            else:
                following.append(child)
        if not following:
            return False, 0
        current = following
    return True, len(current)


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_paths(
    document: Mapping[str, Any], paths: Any, location: str, errors: List[str]
) -> None:
    if not isinstance(paths, list) or not paths:
        errors.append(f"{location} must be a non-empty list")
        return
    for index, path in enumerate(paths):
        exists, _ = resolve_path(document, path)
        if not exists:
            errors.append(f"{location}[{index}] does not resolve in analysis: {path!r}")


def review_plan(
    document: Mapping[str, Any], plan: Mapping[str, Any], digest: str
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if document.get("analysis_version") != ANALYSIS_VERSION:
        errors.append(f"analysis_version must be {ANALYSIS_VERSION}")
    if plan.get("render_plan_version") != RENDER_PLAN_VERSION:
        errors.append(f"render_plan_version must be {RENDER_PLAN_VERSION}")
    if plan.get("analysis_sha256") != digest:
        errors.append("analysis_sha256 does not match the current analysis artifact")
    if plan.get("evidence_policy") != EVIDENCE_POLICY:
        errors.append(f"evidence_policy must be {EVIDENCE_POLICY!r}")
    if not nonempty_string(plan.get("objective")):
        errors.append("objective must be a non-empty string")

    claims = plan.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a non-empty list")
        claims = []
    claim_ids: Set[str] = set()
    for index, claim in enumerate(claims):
        location = f"claims[{index}]"
        if not isinstance(claim, Mapping):
            errors.append(f"{location} must be an object")
            continue
        claim_id = claim.get("id")
        if not nonempty_string(claim_id):
            errors.append(f"{location}.id must be a non-empty string")
        elif claim_id in claim_ids:
            errors.append(f"duplicate claim id: {claim_id}")
        else:
            claim_ids.add(claim_id)
        if not nonempty_string(claim.get("statement")):
            errors.append(f"{location}.statement must be a non-empty string")
        validate_paths(
            document, claim.get("evidence_paths"), f"{location}.evidence_paths", errors
        )
        if not nonempty_string(claim.get("qualification")):
            warnings.append(f"{location}.qualification is empty")

    charts = plan.get("charts")
    if charts is None:
        charts = []
    if not isinstance(charts, list):
        errors.append("charts must be a list")
        charts = []
    chart_ids: Set[str] = set()
    required_brief = (
        "grain",
        "comparison_dimension",
        "metric_semantics",
        "uncertainty",
        "missingness",
    )
    for index, chart in enumerate(charts):
        location = f"charts[{index}]"
        if not isinstance(chart, Mapping):
            errors.append(f"{location} must be an object")
            continue
        chart_id = chart.get("id")
        if not nonempty_string(chart_id):
            errors.append(f"{location}.id must be a non-empty string")
        elif chart_id in chart_ids:
            errors.append(f"duplicate chart id: {chart_id}")
        else:
            chart_ids.add(chart_id)
        linked_claims = chart.get("claim_ids")
        if not isinstance(linked_claims, list) or not linked_claims:
            errors.append(f"{location}.claim_ids must be a non-empty list")
        else:
            for claim_id in linked_claims:
                if claim_id not in claim_ids:
                    errors.append(
                        f"{location} references unknown claim id: {claim_id!r}"
                    )
        validate_paths(
            document, chart.get("data_paths"), f"{location}.data_paths", errors
        )
        for field in required_brief:
            if not nonempty_string(chart.get(field)):
                errors.append(f"{location}.{field} must be a non-empty string")
        if chart.get("visual_priority") not in VISUAL_PRIORITIES:
            errors.append(
                f"{location}.visual_priority must be one of {sorted(VISUAL_PRIORITIES)}"
            )

    return {
        "render_review_version": RENDER_PLAN_VERSION,
        "analysis_sha256": digest,
        "claim_ids": sorted(claim_ids),
        "chart_ids": sorted(chart_ids),
        "errors": errors,
        "warnings": warnings,
    }


class ReportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.analysis_hashes: List[str] = []
        self.claim_ids: Set[str] = set()
        self.chart_ids: Set[str] = set()
        self.external_assets: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("name") == "llmperf-analysis-sha256":
            self.analysis_hashes.append(values.get("content") or "")
        if values.get("data-claim-id"):
            self.claim_ids.update(values["data-claim-id"].split())
        if values.get("data-chart-id"):
            self.chart_ids.update(values["data-chart-id"].split())
        asset = values.get("src") or (
            values.get("href")
            if tag == "link" and values.get("rel") == "stylesheet"
            else None
        )
        if asset and re.match(r"^(?:https?:)?//", asset):
            self.external_assets.append(asset)


def review_html(html_path: Path, review: Dict[str, Any], digest: str) -> None:
    html = html_path.read_text(encoding="utf-8")
    parser = ReportHTMLParser()
    parser.feed(html)
    if parser.analysis_hashes != [digest]:
        review["errors"].append(
            "HTML must contain exactly one llmperf-analysis-sha256 meta tag with the current hash"
        )
    missing_claims = set(review["claim_ids"]) - parser.claim_ids
    missing_charts = set(review["chart_ids"]) - parser.chart_ids
    if missing_claims:
        review["errors"].append(
            f"HTML is missing claim bindings: {sorted(missing_claims)}"
        )
    if missing_charts:
        review["errors"].append(
            f"HTML is missing chart bindings: {sorted(missing_charts)}"
        )
    if parser.external_assets:
        review["errors"].append(
            f"HTML references external assets and is not self-contained: {parser.external_assets}"
        )


def write_result(result: Mapping[str, Any], output: Optional[Path]) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", type=Path)
    mode.add_argument("--plan", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        analysis = load_json(arguments.analysis)
        digest = sha256(arguments.analysis)
        if arguments.inventory:
            if arguments.html:
                raise ValueError("--html requires --plan")
            result = collect_inventory(analysis, digest)
            write_result(result, arguments.inventory)
            return 0
        plan = load_json(arguments.plan)
        result = review_plan(analysis, plan, digest)
        if arguments.html:
            review_html(arguments.html, result, digest)
        write_result(result, arguments.output)
        return 1 if result["errors"] else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

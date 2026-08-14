#!/usr/bin/env python3
"""Validate an LLMPerf Runner or Campaign YAML without submitting it."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

from pydantic import ValidationError
import yaml

from llmperf_backend.models import BenchmarkCampaignStart, BenchmarkRunnerCreate


def load_document(path: Path) -> Dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read YAML {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("YAML document must be a mapping")
    return document


def validate_document(document: Dict[str, Any]) -> str:
    if "campaign" not in document:
        runner = BenchmarkRunnerCreate.model_validate(document)
        benchmark = runner.benchmark
        provider = benchmark.provider if benchmark is not None else "<backend-default>"
        model = benchmark.model if benchmark is not None else "<backend-default>"
        return f"valid runner workload: provider={provider} model={model}"

    payload = {
        key: value for key, value in document.items() if key not in {"wait", "export"}
    }
    campaign = BenchmarkCampaignStart.model_validate(payload)
    return (
        "valid campaign workload: "
        f"runners={len(campaign.runners)} "
        f"runner_plans={len(campaign.runner_plans)} "
        f"protocol_definitions={len(campaign.protocol_definitions)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Runner/Campaign YAML without Backend or Provider calls."
    )
    parser.add_argument("file", type=Path, metavar="FILE")
    arguments = parser.parse_args()
    try:
        print(validate_document(load_document(arguments.file.expanduser())))
    except (ValueError, ValidationError) as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

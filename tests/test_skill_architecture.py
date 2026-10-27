from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD_RUNTIME_NAMES = (
    "cache-retention/v1",
    "cache-residency/v1",
    "cache-promotion/v1",
    "protocol_definitions",
    "campaign-v5-protocol-to-task-graph/v1",
    "Campaign export v6",
)


def _text(*paths):
    return "\n".join((ROOT / path).read_text(encoding="utf-8") for path in paths)


def test_operation_skill():
    text = _text(
        ".codex/skills/operate-llmperf/SKILL.md",
        ".codex/skills/operate-llmperf/references/yaml.md",
        ".codex/skills/operate-llmperf/references/engineering.md",
        ".codex/skills/operate-llmperf/references/io.md",
        ".codex/skills/operate-llmperf/references/cli.md",
    )

    assert "task_definitions" in text
    assert all(term in text for term in ("matrix", "sequence", "repeat", "parallel"))
    assert "Each compiled invoke sends one request" in text
    assert "independent completion boundary" in text
    assert "required tables and columns" in text
    assert "dependencies and due times" in text
    assert "prompt_hash" in text
    assert "CLIProjection" in text
    assert "whitelist projector" in text
    assert "completed_bytes/total_bytes" in text
    assert "without a progress bar" in text
    assert "tqdm bar" not in text
    assert not any(name in text for name in OLD_RUNTIME_NAMES)


@pytest.mark.parametrize("enforce_campaign_limits", [False, True])
def test_workload_validation(tmp_path, enforce_campaign_limits):
    workload_path = tmp_path / "large.yaml"
    workload_path.write_text(
        'version: "1.0.0"\n'
        "provider: test\n"
        "model: test-model\n"
        "concurrent_workers: 1\n"
        "inputs:\n"
        "  default:\n"
        "    tokens: 819200\n"
        "worker:\n"
        "  repeat: 256\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / ".codex/skills/operate-llmperf/scripts/validate_workload.py"),
        str(workload_path),
    ]
    if enforce_campaign_limits:
        command.append("--enforce-campaign-limits")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if enforce_campaign_limits:
        assert result.returncode == 1
        assert "token_budget_limit" in result.stderr
    else:
        assert result.returncode == 0
        assert "campaign_admission=False within_limits=False" in result.stdout
        assert "advisory_risks=1" in result.stdout


def test_reporting_skill():
    text = _text(
        ".codex/skills/generate-llmperf-report/SKILL.md",
        ".codex/skills/generate-llmperf-report/references/report-contract.md",
        ".codex/skills/generate-llmperf-report/references/rendering-review.md",
    )

    assert "Campaign export version `1.0.0`" in text
    assert "evidence.task_graphs" in text
    assert "chart" in text.lower()
    assert "1×" in text
    assert "0–100%" in text
    assert not any(name in text for name in OLD_RUNTIME_NAMES)
    assert "generate_report.py" not in text
    assert "normalized-analysis-only" in text
    assert "explicitly" in text and "skill" in text
    assert (
        ROOT / ".codex/skills/generate-llmperf-report/references/rendering-review.md"
    ).is_file()
    assert (
        ROOT / ".codex/skills/generate-llmperf-report/scripts/review_render_plan.py"
    ).is_file()


def test_deployment_cutover():
    text = _text(".codex/skills/deploy-llmperf/references/deployment.md")

    assert "independent Worker metadata/file-reference layout" in text
    assert "No automatic compatibility migration" in text
    assert "separate database" in text and "service port" in text
    assert "clean 1.0.3 schema" in text

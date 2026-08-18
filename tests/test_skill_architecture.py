from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_RUNTIME_NAMES = (
    "cache-retention/v1",
    "cache-residency/v1",
    "cache-promotion/v1",
    "protocol_definitions",
)


def _text(*paths):
    return "\n".join((ROOT / path).read_text(encoding="utf-8") for path in paths)


def test_operation_skill():
    text = _text(
        ".codex/skills/operate-llmperf/SKILL.md",
        ".codex/skills/operate-llmperf/references/yaml.md",
        ".codex/skills/operate-llmperf/references/engineering.md",
    )

    assert "task_definitions" in text
    assert all(term in text for term in ("matrix", "sequence", "repeat", "parallel"))
    assert "single-request Runner" in text
    assert "Planner only handles" in text
    assert "prompt_hash" in text
    assert not any(name in text for name in OLD_RUNTIME_NAMES)


def test_reporting_skill():
    text = _text(
        ".codex/skills/generate-llmperf-report/SKILL.md",
        ".codex/skills/generate-llmperf-report/references/report-contract.md",
    )

    assert "Campaign export version 6" in text
    assert "evidence.task_graphs" in text
    assert "chart" in text.lower()
    assert "1×" in text
    assert "0–100%" in text
    assert not any(name in text for name in OLD_RUNTIME_NAMES)
    assert "generate_report.py" not in text


def test_deployment_cutover():
    text = _text(".codex/skills/deploy-llmperf/references/deployment.md")

    assert "pending" in text and "legacy\nDispatches" in text
    assert "separate database and service port" in text
    assert "explicit migration or rebuild" in text

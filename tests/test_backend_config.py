from pathlib import Path
import os

import pytest

from llmperf_backend.config import (
    ConfigError,
    ConfigStore,
    load_config,
    load_config_text,
)
from llmperf_backend.environment import load_environment
from llmperf_backend.models import BenchmarkCampaignStart, DatabaseConfig


VALID_CONFIG = """
version: 1
environment: test
server:
  host: 127.0.0.1
  port: 9000
benchmark:
  provider: test
  model: test-model
  llm_api: openai
  concurrent_requests: 2
"""


def test_defaults():
    config = load_config_text(VALID_CONFIG)

    assert config.environment == "test"
    assert config.server.port == 9000
    assert config.benchmark.concurrent_requests == 2
    assert config.benchmark.timeout_seconds == 90
    assert config.benchmark.tokenizer.selection == "global_default"
    assert config.benchmark.tokenizer.accuracy == "approximate"
    assert config.planner.enabled is True
    assert config.planner.batch_size == 20


def test_postgres_only():
    with pytest.raises(ValueError, match=r"postgresql\+asyncpg"):
        DatabaseConfig(url="mysql+aiomysql:///unsupported")


def test_campaign_workload():
    with pytest.raises(ValueError, match="protocol_definitions"):
        BenchmarkCampaignStart.model_validate({"campaign": {"name": "empty"}})

    campaign = BenchmarkCampaignStart.model_validate(
        {
            "campaign": {"name": "planned"},
            "runner_plans": [
                {
                    "name": "every-30s",
                    "timezone": "Asia/Shanghai",
                    "max_occurrences": 8,
                    "recurrence": {"kind": "interval", "every_seconds": 30},
                    "runner": {},
                }
            ],
        }
    )

    assert campaign.runners == []
    assert campaign.runner_plans[0].max_occurrences == 8
    assert campaign.runner_plans[0].starts_at is None

    sweep = BenchmarkCampaignStart.model_validate(
        {
            "campaign": {"name": "retention"},
            "protocol_definitions": [
                {
                    "name": "ttl",
                    "protocol": "cache-retention/v1",
                    "delay_seconds": [0, 60, 3600],
                    "trials_per_delay": 2,
                    "runner": {},
                }
            ],
        }
    )
    assert sweep.protocol_definitions[0].refresh_semantics == "independent_family"
    assert sweep.protocol_definitions[0].trials_per_delay == 2

    residency = BenchmarkCampaignStart.model_validate(
        {
            "campaign": {"name": "residency"},
            "protocol_definitions": [
                {
                    "name": "daily-hours",
                    "protocol": "cache-residency/v1",
                    "schedule": {
                        "kind": "geographic",
                        "timezone": "Asia/Shanghai",
                        "starts_at": "2026-08-15T00:00:00+08:00",
                        "every_seconds": 3600,
                        "duration_days": 1,
                    },
                    "mapping": "one_to_one",
                    "chains": 2,
                    "runner": {},
                }
            ],
        }
    )
    definition = residency.protocol_definitions[0]
    assert definition.protocol == "cache-residency/v1"
    assert definition.schedule.timezone == "Asia/Shanghai"
    assert definition.chains == 2

    with pytest.raises(ValueError, match="UTC offset"):
        BenchmarkCampaignStart.model_validate(
            {
                "campaign": {"name": "bad-residency"},
                "protocol_definitions": [
                    {
                        "name": "bad-hours",
                        "protocol": "cache-residency/v1",
                        "schedule": {
                            "kind": "geographic",
                            "timezone": "Asia/Shanghai",
                            "starts_at": "2026-08-15T00:00:00Z",
                            "every_seconds": 3600,
                            "duration_days": 1,
                        },
                        "mapping": "one_to_one",
                        "runner": {},
                    }
                ],
            }
        )


def test_runner_tokenizer():
    config = load_config_text(
        VALID_CONFIG
        + """
  tokenizer:
    id: Qwen/Qwen2.5-7B-Instruct
    revision: tokenizer-release
    use_fast: false
"""
    )

    assert config.benchmark.tokenizer.id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.benchmark.tokenizer.revision == "tokenizer-release"
    assert config.benchmark.tokenizer.use_fast is False
    assert config.benchmark.tokenizer.selection == "explicit"
    assert config.benchmark.tokenizer.accuracy == "compatible"


def test_probe_tokenizer_rejection():
    with pytest.raises(ConfigError, match="allow_approximate_tokenizer"):
        load_config_text(
            VALID_CONFIG
            + """
  cache_probe:
    mode: exact_repeat
    trials: 2
"""
        )


def test_probe_tokenizer_acceptance():
    config = load_config_text(
        VALID_CONFIG
        + """
  tokenizer:
    id: organization/model-tokenizer
    revision: release
  cache_probe:
    mode: exact_repeat
    trials: 2
"""
    )

    assert config.benchmark.cache_probe.trials == 2
    assert config.benchmark.tokenizer.selection == "explicit"


def test_environment_expansion(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_NAME", "glm-test")
    config = load_config_text(VALID_CONFIG.replace("test-model", "${TEST_MODEL_NAME}"))

    assert config.benchmark.model == "glm-test"


def test_dotenv_precedence(tmp_path, monkeypatch):
    environment_path = tmp_path / ".env"
    environment_path.write_text(
        "DOTENV_ONLY=loaded\nDOTENV_PRECEDENCE=from-file\n", encoding="utf-8"
    )
    monkeypatch.delenv("DOTENV_ONLY", raising=False)
    monkeypatch.setenv("DOTENV_PRECEDENCE", "from-process")

    loaded_path = load_environment(environment_path)

    assert loaded_path == environment_path.resolve()
    assert os.environ["DOTENV_ONLY"] == "loaded"
    assert os.environ["DOTENV_PRECEDENCE"] == "from-process"


def test_missing_dotenv(tmp_path):
    with pytest.raises(RuntimeError, match="Environment file does not exist"):
        load_environment(tmp_path / "missing.env")


def test_default_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LLMPERF_DEFAULT_MODEL=glm-from-dotenv\nLLMPERF_SERVER_PORT=8123\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLMPERF_ENV_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.delenv("LLMPERF_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("LLMPERF_SERVER_PORT", raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.benchmark.model == "glm-from-dotenv"
    assert config.server.port == 8123


def test_unknown_keys():
    with pytest.raises(ConfigError, match="extra_forbidden"):
        load_config_text(VALID_CONFIG + "unknown: true\n")


def test_atomic_reload(tmp_path: Path):
    config_path = tmp_path / "backend.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    store = ConfigStore(config_path)

    initial = store.snapshot()
    config_path.write_text("benchmark: [invalid", encoding="utf-8")

    with pytest.raises(ConfigError):
        store.reload()

    after_failure = store.snapshot()
    assert after_failure.generation == initial.generation
    assert after_failure.config == initial.config

    config_path.write_text(
        VALID_CONFIG.replace("port: 9000", "port: 9100"), encoding="utf-8"
    )
    after_success = store.reload()
    assert after_success.generation == 1
    assert after_success.config["server"]["port"] == 9100

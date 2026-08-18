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
from llmperf_backend.models import (
    AppConfig,
    BenchmarkCampaignStart,
    DatabaseConfig,
    PerformanceGuardConfig,
    SchedulerConfig,
    ServerConfig,
)


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
    assert config.scheduler.ray_num_cpus == 8
    assert config.scheduler.ray_actor_num_cpus == 1.0
    assert config.scheduler.ray_object_store_memory_bytes == 268_435_456
    assert config.scheduler.artifact_resolution_timeout_seconds == 60.0
    assert config.performance_guard.min_ray_object_store_available_ratio == 0.1
    assert config.performance_guard.resume_ray_object_store_available_ratio == 0.2


def test_ray_runtime_config():
    external = load_config_text(
        VALID_CONFIG
        + """
scheduler:
  max_concurrent_runners: 2
  ray_address: ray://127.0.0.1:10001
  ray_actor_num_cpus: 0.5
"""
    )
    assert external.scheduler.ray_address == "ray://127.0.0.1:10001"
    assert external.scheduler.ray_actor_num_cpus == 0.5
    assert SchedulerConfig(ray_address="").ray_address is None


def test_worker_config_compatibility():
    config = load_config_text(
        VALID_CONFIG
        + """
scheduler:
  working_directory: /legacy/worker/path
  worker_module: custom.worker.module
  cancel_grace_seconds: 7
  log_bytes_limit: 4096
"""
    )

    assert config.scheduler.working_directory == "/legacy/worker/path"
    assert config.scheduler.worker_module == "custom.worker.module"
    assert config.scheduler.cancel_grace_seconds == 7
    assert config.scheduler.log_bytes_limit == 4096


def test_ray_worker_limit():
    with pytest.raises(ValueError, match="embedded Ray requires server.workers=1"):
        AppConfig(
            server=ServerConfig(workers=2),
            scheduler=SchedulerConfig(),
            benchmark={"provider": "test", "model": "test"},
        )


def test_ray_slot_capacity():
    with pytest.raises(ValueError, match="Ray actor capacity"):
        AppConfig(
            scheduler=SchedulerConfig(
                max_concurrent_runners=3,
                ray_num_cpus=2,
                ray_actor_num_cpus=1,
            ),
            performance_guard=PerformanceGuardConfig(),
            benchmark={"provider": "test", "model": "test"},
        )


def test_postgres_only():
    with pytest.raises(ValueError, match=r"postgresql\+asyncpg"):
        DatabaseConfig(url="mysql+aiomysql:///unsupported")


def test_campaign_workload():
    with pytest.raises(ValueError, match="task_definitions"):
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

    compiled = BenchmarkCampaignStart.model_validate(
        {
            "campaign": {"name": "compiled"},
            "task_definitions": [
                {
                    "name": "replay",
                    "matrix": {"delay": [0, 60], "hits": [0, 2]},
                    "trials": 2,
                    "payloads": {
                        "replay": {"seed_namespace": "replay"},
                        "cold": {"seed_namespace": "cold"},
                    },
                    "sequence": [
                        {
                            "kind": "invoke",
                            "id": "prime",
                            "role": "prime",
                            "payload": "replay",
                        },
                        {
                            "kind": "repeat",
                            "id": "hits",
                            "count": {"dimension": "hits"},
                            "invoke": {
                                "kind": "invoke",
                                "id": "warm",
                                "role": "warm",
                                "payload": "replay",
                            },
                        },
                        {
                            "kind": "parallel",
                            "after_seconds": {"dimension": "delay"},
                            "invokes": [
                                {
                                    "kind": "invoke",
                                    "id": "probe",
                                    "role": "probe",
                                    "payload": "replay",
                                },
                                {
                                    "kind": "invoke",
                                    "id": "cold",
                                    "role": "cold_control",
                                    "payload": "cold",
                                },
                            ],
                        },
                    ],
                    "runner": {},
                }
            ],
        }
    )
    definition = compiled.task_definitions[0]
    assert definition.matrix["delay"] == [0, 60]
    assert definition.trials == 2

    with pytest.raises(ValueError, match="unknown payload"):
        BenchmarkCampaignStart.model_validate(
            {
                "campaign": {"name": "bad"},
                "task_definitions": [
                    {
                        "name": "bad",
                        "payloads": {"known": {"seed_namespace": "known"}},
                        "sequence": [
                            {
                                "kind": "invoke",
                                "id": "node",
                                "role": "warm",
                                "payload": "missing",
                            }
                        ],
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

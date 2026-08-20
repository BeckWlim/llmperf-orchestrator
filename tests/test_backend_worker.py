"""Worker/Ray execution, resource isolation, environment, and outcome tests."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llmperf import common_metrics
from llmperf.common import RAY_ACTOR_CPUS_ENV, _ray_client_class, construct_clients
import llmperf_backend.worker as worker_module
from llmperf_backend.models import BenchmarkConfig, dump_model
from llmperf_backend.persistence import (
    FAILED,
    SUCCEEDED,
    BenchmarkRunnerRecord,
    _runner_list_dict,
    json_safe,
)
from llmperf_backend.worker import (
    Worker,
    _calculate,
    benchmark_actor_count,
    runtime_environment,
    summarize_outcome,
)

OUTCOME_CASES = [
    pytest.param(
        {
            "summary": {
                "results": {
                    common_metrics.NUM_REQ_STARTED: 1,
                    common_metrics.NUM_COMPLETED_REQUESTS: 0,
                    common_metrics.NUM_ERRORS: 1,
                }
            },
            "requests": [
                {
                    common_metrics.ERROR_CODE: -1,
                    common_metrics.ERROR_MSG: "JSONDecodeError: invalid stream",
                }
            ],
            "status": FAILED,
            "outcome": "failed",
            "message": (
                "No benchmark requests completed (1 failed); first error [-1]: "
                "JSONDecodeError: invalid stream"
            ),
            "first_error": {
                "code": -1,
                "message": "JSONDecodeError: invalid stream",
            },
        },
        id="failed",
    ),
    pytest.param(
        {
            "summary": {
                "results": {
                    common_metrics.NUM_REQ_STARTED: 2,
                    common_metrics.NUM_COMPLETED_REQUESTS: 1,
                    common_metrics.NUM_ERRORS: 1,
                }
            },
            "requests": [],
            "status": SUCCEEDED,
            "outcome": "degraded",
            "message": "1 request completed; 1 request failed",
            "first_error": None,
        },
        id="degraded",
    ),
    pytest.param(
        {
            "summary": {
                "timed_out": True,
                "results": {
                    common_metrics.NUM_REQ_STARTED: 0,
                    common_metrics.NUM_COMPLETED_REQUESTS: 0,
                    common_metrics.NUM_ERRORS: 0,
                },
            },
            "requests": [],
            "status": FAILED,
            "outcome": "failed",
            "message": "No benchmark requests completed before timeout",
            "first_error": None,
        },
        id="timeout",
    ),
]


@pytest.mark.parametrize("case", OUTCOME_CASES)
def test_outcome(case):
    status, message = summarize_outcome(case["summary"], case["requests"])

    assert status == case["status"]
    assert message == case["message"]
    assert case["summary"]["outcome"]["status"] == case["outcome"]
    assert case["summary"]["outcome"]["first_error"] == case["first_error"]


def test_finite_float():
    assert json_safe(1.0) == 1.0


def test_shared_ray_options(monkeypatch):
    fake_benchmark = SimpleNamespace(
        get_token_throughput_latencies=lambda **options: ({"ok": True}, [])
    )
    import sys

    monkeypatch.setitem(sys.modules, "llmperf.token_benchmark_ray", fake_benchmark)

    benchmark = dump_model(BenchmarkConfig(provider="test", model="test"))
    benchmark["adapter"] = "openai"
    result = _calculate(
        benchmark,
        {
            "backend": "ray",
            "worker_kind": "ray_task",
            "ray_mode": "external",
            "ray_namespace": "llmperf-control",
        },
    )

    assert result[0]["ok"] is True
    assert result[0]["execution_runtime"] == {
        "backend": "ray",
        "worker_kind": "ray_task",
        "ray_mode": "external",
        "ray_namespace": "llmperf-control",
        "ray_actor_num_cpus": 1.0,
    }
    assert result[1] == []


def test_worker_environment():
    runtime = runtime_environment(
        {
            "OPENAI_API_KEY": "provider-key",
            "HTTPS_PROXY": "http://proxy-user:proxy-password@proxy.internal:3128",
            "NO_PROXY": "127.0.0.1,localhost",
            "RAY_grpc_enable_http_proxy": "0",
            "LLMPERF_PROXY": "must-not-propagate",
            "LLMPERF_PRIVATE_KEY": "must-not-propagate",
            "DATABASE_URL": "must-not-propagate",
            "LLMPERF_WORKER_RAY_ACTOR_CPUS": "0.5",
        }
    )

    assert runtime["OPENAI_API_KEY"] == "provider-key"
    assert runtime["HTTPS_PROXY"] == (
        "http://proxy-user:proxy-password@proxy.internal:3128"
    )
    assert runtime["NO_PROXY"] == "127.0.0.1,localhost"
    assert runtime["RAY_grpc_enable_http_proxy"] == "0"
    assert runtime["LLMPERF_WORKER_RAY_ACTOR_CPUS"] == "0.5"
    assert "LLMPERF_PROXY" not in runtime
    assert "LLMPERF_PRIVATE_KEY" not in runtime
    assert "DATABASE_URL" not in runtime


def test_worker_actor_count():
    benchmark = dump_model(BenchmarkConfig(provider="test", model="test"))
    benchmark["concurrent_requests"] = 8
    benchmark["max_completed_requests"] = 3
    assert benchmark_actor_count(benchmark) == 3
    benchmark["cache_probe"] = {"trials": 2}
    assert benchmark_actor_count(benchmark) == 2


def test_worker_handle():
    calls = {}

    class Reference:
        def __init__(self, value):
            self.value = value

        def task_id(self):
            return SimpleNamespace(hex=lambda: "task-1")

    class RemoteFunction:
        def options(self, **options):
            calls["options"] = options
            return self

        def remote(self, *arguments):
            calls["arguments"] = arguments
            return Reference({"ok": True})

    fake_ray = SimpleNamespace(
        wait=lambda refs, timeout=0: (refs, []),
        get=lambda ref: ref.value,
        cancel=lambda ref, force=False: calls.setdefault("cancel", force),
    )
    worker = Worker(fake_ray, RemoteFunction(), "runner-1", 2, 0.5)

    worker.start(
        {"model": "test"},
        {"OPENAI_API_KEY": "key"},
        {"backend": "ray"},
        1024,
    )
    assert worker.ready() is True
    assert worker.result() == {"ok": True}
    assert worker.task_id() == "task-1"
    assert calls["options"]["runtime_env"]["env_vars"] == {"OPENAI_API_KEY": "key"}
    assert "scheduling_strategy" not in calls["options"]
    worker.close()
    assert worker.task_ref is None


def test_worker_task_failure(monkeypatch):
    def fail(benchmark, execution_runtime):
        raise RuntimeError("benchmark failed")

    monkeypatch.setattr(worker_module, "_calculate", fail)

    result = worker_module.execute_worker_task({}, {"backend": "ray"}, 4096)

    assert result["ok"] is False
    assert result["error"] == "RuntimeError: benchmark failed"
    assert "RuntimeError: benchmark failed" in result["stderr"]


def test_actor_resource_options(monkeypatch):
    import sys

    captured = {}

    class RemoteClass:
        @classmethod
        def options(cls, **options):
            captured.update(options)
            return cls

        @staticmethod
        def remote():
            return "actor"

    def remote(client_class):
        captured["client_class"] = client_class
        return RemoteClass

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=remote))
    monkeypatch.setenv(RAY_ACTOR_CPUS_ENV, "0.5")
    _ray_client_class.cache_clear()

    clients = construct_clients("openai", 2)

    assert clients == ["actor", "actor"]
    assert captured.pop("client_class").__name__ == "OpenAIChatCompletionsClient"
    assert captured == {
        "num_cpus": 0.5,
        "max_concurrency": 1,
        "max_restarts": 0,
        "max_task_retries": 0,
    }
    _ray_client_class.cache_clear()


def test_list_projection():
    created_at = datetime(2026, 8, 12, tzinfo=timezone.utc)
    runner = BenchmarkRunnerRecord(
        id="runner-1",
        campaign_id=None,
        runner_plan_id=None,
        plan_occurrence=None,
        scheduled_for=None,
        label="smoke",
        created_by="tester",
        status=FAILED,
        benchmark_config={"provider": "aliyun", "model": "glm-5.2"},
        summary={
            "results": {"error_rate": 1.0},
            "outcome": {
                "requests_started": 1,
                "requests_completed": 0,
                "requests_failed": 1,
                "message": "No benchmark requests completed",
            },
        },
        error_message="large worker error",
        created_at=created_at,
        started_at=None,
        finished_at=created_at,
        scheduler_id="scheduler-1",
        process_id=62791,
        exit_code=1,
        stdout="large stdout",
        stderr="large stderr",
        user_metadata={},
        request_count=0,
        cancel_requested=False,
    )

    listed = _runner_list_dict(runner)

    assert listed["provider"] == "aliyun"
    assert listed["model"] == "glm-5.2"
    assert listed["requests"] == {
        "started": 1,
        "completed": 0,
        "failed": 1,
        "error_rate": 1.0,
    }
    assert listed["scheduler_id"] == "scheduler-1"
    assert listed["worker"] == {"process_id": 62791, "exit_code": 1}
    assert "summary" not in listed
    assert "stdout" not in listed
    assert "stderr" not in listed

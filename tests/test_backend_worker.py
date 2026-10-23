"""Worker/Ray execution, resource isolation, environment, and outcome tests."""

import asyncio
from datetime import datetime, timezone
import threading
from types import SimpleNamespace

import pytest

from llmperf.metrics import constants as common_metrics
from llmperf.clients.factory import (
    RAY_ACTOR_CPUS_ENV,
    _ray_client_class,
    construct_clients,
)
import llmperf_backend.runtime.worker as worker_module
from llmperf_backend.configuration.models import BenchmarkConfig, dump_model
from llmperf_backend.storage.repository import (
    FAILED,
    SUCCEEDED,
    BenchmarkRunnerRecord,
    _runner_list_dict,
    json_safe,
)
import llmperf_backend.runtime.preparation as preparation_module
from llmperf_backend.runtime.preparation import calculate
from llmperf_backend.runtime.worker import (
    Worker,
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


@pytest.mark.parametrize("count_mode", ["provider", "local"])
def test_shared_ray_options(monkeypatch, count_mode):
    received_options = {}

    def capture_options(**options):
        received_options.update(options)
        return {"ok": True}, []

    fake_benchmark = SimpleNamespace(get_token_throughput_latencies=capture_options)
    import sys

    monkeypatch.setitem(sys.modules, "llmperf.benchmark.throughput", fake_benchmark)

    benchmark = dump_model(BenchmarkConfig(provider="test", model="test"))
    benchmark["adapter"] = "openai"
    benchmark["output_token_count"] = count_mode
    result = calculate(
        benchmark,
        {
            "backend": "ray",
            "worker_kind": "ray_task",
            "ray_mode": "external",
            "ray_namespace": "llmperf-control",
        },
    )

    assert result[0]["ok"] is True
    assert received_options["output_token_count"] == count_mode
    assert result[0]["execution_runtime"] == {
        "backend": "ray",
        "worker_kind": "ray_task",
        "ray_mode": "external",
        "ray_namespace": "llmperf-control",
        "ray_actor_num_cpus": 1.0,
        "ray_actor_max_concurrency": 1,
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
            "LLMPERF_WORKER_RAY_ACTOR_CONCURRENCY": "8",
        }
    )

    assert runtime["OPENAI_API_KEY"] == "provider-key"
    assert runtime["HTTPS_PROXY"] == (
        "http://proxy-user:proxy-password@proxy.internal:3128"
    )
    assert runtime["NO_PROXY"] == "127.0.0.1,localhost"
    assert runtime["RAY_grpc_enable_http_proxy"] == "0"
    assert runtime["LLMPERF_WORKER_RAY_ACTOR_CPUS"] == "0.5"
    assert runtime["LLMPERF_WORKER_RAY_ACTOR_CONCURRENCY"] == "8"
    assert "LLMPERF_PROXY" not in runtime
    assert "LLMPERF_PRIVATE_KEY" not in runtime
    assert "DATABASE_URL" not in runtime


def test_worker_actor_count():
    benchmark = dump_model(BenchmarkConfig(provider="test", model="test"))
    benchmark["concurrent_workers"] = 8
    benchmark["repeat"] = 3
    assert benchmark_actor_count(benchmark) == 3
    benchmark["cache_probe"] = {"trials": 2}
    assert benchmark_actor_count(benchmark) == 2
    assert benchmark_actor_count(benchmark, 8) == 1
    benchmark["cache_probe"] = None
    benchmark["concurrent_workers"] = 64
    benchmark["repeat"] = 640
    assert benchmark_actor_count(benchmark, 8) == 8
    benchmark["concurrent_workers"] = 65
    assert benchmark_actor_count(benchmark, 8) == 9


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
    assert asyncio.run(worker.result_async()) == {"ok": True}
    assert worker.task_id() == "task-1"
    assert calls["options"]["runtime_env"]["env_vars"] == {"OPENAI_API_KEY": "key"}
    assert "scheduling_strategy" not in calls["options"]
    worker.close()
    assert worker.task_ref is None


def test_worker_task_failure(monkeypatch):
    def fail(benchmark, execution_runtime):
        raise RuntimeError("benchmark failed")

    monkeypatch.setattr(preparation_module, "calculate", fail)

    result = worker_module.execute_worker_task({}, {"backend": "ray"}, 4096)

    assert result["ok"] is False
    assert result["error"] == "RuntimeError: benchmark failed"
    assert "RuntimeError: benchmark failed" in result["stderr"]


@pytest.mark.parametrize("fails", [False, True])
def test_listener_shutdown(monkeypatch, fails):
    listener_started = threading.Event()
    release_listener = threading.Event()
    listeners = []
    thread_errors = []
    monkeypatch.setattr(threading, "excepthook", thread_errors.append)

    def get_result(_task_ref):
        listeners.append(threading.current_thread())
        listener_started.set()
        release_listener.wait(timeout=5)
        if fails:
            raise RuntimeError("Ray task failed")
        return {"ok": True}

    worker = Worker(SimpleNamespace(get=get_result), object(), "runner-1", 1, 1.0)
    worker.task_ref = object()

    async def exercise():
        result_task = asyncio.create_task(worker.result_async())
        while not listener_started.is_set():
            await asyncio.sleep(0.001)
        result_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await result_task

    try:
        asyncio.run(asyncio.wait_for(exercise(), timeout=2))
    finally:
        release_listener.set()
        for listener in listeners:
            listener.join(timeout=2)

    assert listeners and not listeners[0].is_alive()
    assert thread_errors == []


def test_async_result_failure():
    error = RuntimeError("Ray task failed")

    def get_result(_task_ref):
        raise error

    worker = Worker(SimpleNamespace(get=get_result), object(), "runner-1", 1, 1.0)
    with pytest.raises(RuntimeError, match="has not started"):
        asyncio.run(worker.result_async())

    worker.task_ref = object()
    with pytest.raises(RuntimeError, match="Ray task failed") as caught:
        asyncio.run(worker.result_async())
    assert caught.value is error


@pytest.mark.parametrize("concurrency", [1, 8])
def test_actor_resource_options(monkeypatch, concurrency):
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
    monkeypatch.setenv("LLMPERF_WORKER_RAY_ACTOR_CONCURRENCY", str(concurrency))
    _ray_client_class.cache_clear()

    clients = construct_clients("openai", 2)

    assert clients == ["actor", "actor"]
    assert captured.pop("client_class").__name__ == "OpenAIChatCompletionsClient"
    assert captured == {
        "num_cpus": 0.5,
        "max_concurrency": concurrency,
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
        exit_code=1,
        log_key="ab/" + "ab" * 32 + ".json",
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
    assert listed["worker"] == {"process_id": None, "exit_code": 1}
    assert "summary" not in listed
    assert "stdout" not in listed
    assert "stderr" not in listed

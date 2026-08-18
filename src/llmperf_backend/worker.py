"""Ray-backed execution abstraction for one durable benchmark Runner."""

from contextlib import redirect_stderr, redirect_stdout
import os
import threading
import traceback
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from llmperf import common_metrics
from llmperf.common import RAY_ACTOR_CPUS_ENV
from llmperf.utils import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.datasets import WORKER_DATASET_PATH
from llmperf_backend.persistence import FAILED, SUCCEEDED


class _TailBuffer:
    """Thread-safe text stream that retains only its configured tail."""

    encoding = "utf-8"

    def __init__(self, limit: int):
        self.limit = limit
        self._value = ""
        self._lock = threading.Lock()

    def write(self, value: str) -> int:
        text = str(value)
        with self._lock:
            self._value = (self._value + text)[-self.limit :]
        return len(text)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        with self._lock:
            return self._value


def benchmark_actor_count(benchmark: Mapping[str, Any]) -> int:
    """Return the exact maximum number of request actors created by a Runner."""

    concurrent = int(benchmark["concurrent_requests"])
    cache_probe = benchmark.get("cache_probe")
    if cache_probe:
        return max(1, min(concurrent, int(cache_probe["trials"])))
    return max(1, min(concurrent, int(benchmark["max_completed_requests"])))


def runtime_environment(environment: Mapping[str, str]) -> Dict[str, str]:
    """Keep Provider credentials but exclude Backend control-plane secrets."""

    allowed_llmperf_variables = {
        RAY_ACTOR_CPUS_ENV,
        TOKENIZER_FAST,
        TOKENIZER_PATH,
        WORKER_DATASET_PATH,
    }
    return {
        name: value
        for name, value in environment.items()
        if name != "DATABASE_URL"
        and (not name.startswith("LLMPERF_") or name in allowed_llmperf_variables)
    }


def _calculate(
    benchmark: Dict[str, Any], execution_runtime: Optional[Mapping[str, Any]] = None
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    """Run one benchmark inside an already connected Ray worker process."""

    from llmperf.token_benchmark_ray import get_token_throughput_latencies

    dataset = benchmark.get("dataset")
    dataset_path = None
    dataset_format = "sharegpt"
    if dataset is not None:
        dataset_path = os.environ.get(WORKER_DATASET_PATH)
        if not dataset_path:
            raise RuntimeError(
                f"{WORKER_DATASET_PATH} is required for dataset workloads"
            )
        dataset_format = dataset["format"]
    summary, requests = get_token_throughput_latencies(
        model=benchmark["model"],
        llm_api=benchmark["llm_api"],
        test_timeout_s=benchmark["timeout_seconds"],
        max_num_completed_requests=benchmark["max_completed_requests"],
        num_concurrent_requests=benchmark["concurrent_requests"],
        mean_input_tokens=benchmark["mean_input_tokens"],
        stddev_input_tokens=benchmark["stddev_input_tokens"],
        shared_prefix_tokens=benchmark.get("shared_prefix_tokens", 0),
        dataset_path=dataset_path,
        dataset_format=dataset_format,
        dataset_repeat_count=benchmark.get("dataset_repeat_count", 1),
        dataset_seed=benchmark.get("dataset_seed", 11111),
        mean_output_tokens=benchmark["mean_output_tokens"],
        stddev_output_tokens=benchmark["stddev_output_tokens"],
        additional_sampling_params=benchmark["additional_sampling_params"],
        cache_probe=benchmark.get("cache_probe"),
        tokenizer_provenance=benchmark.get("tokenizer"),
        task_request=benchmark.get("task_request"),
    )
    runtime = dict(execution_runtime or {})
    runtime.setdefault("backend", "ray")
    runtime.setdefault("worker_kind", "ray_task")
    runtime.setdefault(
        "ray_actor_num_cpus", float(os.environ.get(RAY_ACTOR_CPUS_ENV, "1.0"))
    )
    summary["execution_runtime"] = runtime
    return summary, requests


def execute_worker_task(
    benchmark: Dict[str, Any],
    execution_runtime: Mapping[str, Any],
    log_limit: int,
) -> Dict[str, Any]:
    """Ray task entrypoint; it never connects to PostgreSQL or initializes Ray."""

    stdout = _TailBuffer(log_limit)
    stderr = _TailBuffer(log_limit)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            summary, requests = _calculate(benchmark, execution_runtime)
        return {
            "ok": True,
            "summary": summary,
            "requests": list(requests),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    except Exception as exc:
        stderr.write(traceback.format_exc())
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }


class Worker:
    """Scheduler-owned handle around one retry-free Ray task/ObjectRef."""

    def __init__(
        self,
        ray_module: Any,
        remote_function: Any,
        runner_id: str,
        actor_count: int,
        actor_num_cpus: float,
    ):
        self.ray = ray_module
        self.remote_function = remote_function
        self.runner_id = runner_id
        self.actor_count = actor_count
        self.actor_num_cpus = actor_num_cpus
        self.task_ref: Any = None
        self._closed = False

    @classmethod
    def remote(cls, ray_module: Any) -> Any:
        """Build the retry-free Ray task used by all Worker handles."""

        return ray_module.remote(num_cpus=0, max_retries=0)(execute_worker_task)

    def start(
        self,
        benchmark: Dict[str, Any],
        environment: Mapping[str, str],
        execution_runtime: Mapping[str, Any],
        log_limit: int,
    ) -> None:
        if self.task_ref is not None:
            raise RuntimeError(f"Worker {self.runner_id} has already started")
        self.task_ref = self.remote_function.options(
            name=f"llmperf-worker-{self.runner_id}",
            runtime_env={"env_vars": runtime_environment(environment)},
        ).remote(dict(benchmark), dict(execution_runtime), log_limit)

    def ready(self) -> bool:
        if self.task_ref is None:
            return False
        ready, _ = self.ray.wait([self.task_ref], timeout=0)
        return bool(ready)

    def result(self) -> Dict[str, Any]:
        if self.task_ref is None:
            raise RuntimeError(f"Worker {self.runner_id} has not started")
        return self.ray.get(self.task_ref)

    def task_id(self) -> Optional[str]:
        if self.task_ref is None:
            return None
        try:
            return self.task_ref.task_id().hex()
        except (AttributeError, TypeError):
            return None

    def cancel(self, force: bool = False) -> None:
        if self.task_ref is not None and not self.ready():
            self.ray.cancel(self.task_ref, force=force)

    def close(self) -> None:
        if self._closed:
            return
        # Dropping the final local ObjectRef lets Ray release task result state and,
        # after task completion, the Runner-owned actor handles created inside it.
        self.task_ref = None
        self._closed = True


def summarize_outcome(
    summary: Dict[str, Any], requests: Sequence[Dict[str, Any]]
) -> Tuple[str, str]:
    """Attach a machine-readable outcome and choose the durable Runner status."""

    results = summary.get("results") or {}
    started = int(results.get(common_metrics.NUM_REQ_STARTED) or len(requests))
    completed = int(results.get(common_metrics.NUM_COMPLETED_REQUESTS) or 0)
    errors = int(results.get(common_metrics.NUM_ERRORS) or 0)
    first_error = None
    for request in requests:
        code = request.get(common_metrics.ERROR_CODE)
        if code is None:
            continue
        error_message = str(request.get(common_metrics.ERROR_MSG) or "Unknown error")
        if len(error_message) > 2000:
            error_message = f"{error_message[:1997]}..."
        first_error = {"code": code, "message": error_message}
        break

    if completed == 0:
        outcome = "failed"
        if summary.get("timed_out") and not errors:
            message = "No benchmark requests completed before timeout"
        else:
            message = f"No benchmark requests completed ({errors or started} failed)"
        if first_error is not None:
            message = (
                f"{message}; first error [{first_error['code']}]: "
                f"{first_error['message']}"
            )
        status = FAILED
    elif errors:
        outcome = "degraded"
        completed_label = "request" if completed == 1 else "requests"
        failed_label = "request" if errors == 1 else "requests"
        message = (
            f"{completed} {completed_label} completed; "
            f"{errors} {failed_label} failed"
        )
        status = SUCCEEDED
    else:
        outcome = "succeeded"
        request_label = "request" if completed == 1 else "requests"
        message = f"{completed} benchmark {request_label} completed"
        status = SUCCEEDED

    summary["outcome"] = {
        "status": outcome,
        "requests_started": started,
        "requests_completed": completed,
        "requests_failed": errors,
        "first_error": first_error,
        "message": message,
    }
    return status, message

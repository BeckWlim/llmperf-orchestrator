"""Ray-backed execution abstraction for one durable benchmark Runner."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, Future
from contextlib import redirect_stderr, redirect_stdout
import os
import threading
import traceback
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

from llmperf import common_metrics
from llmperf.preparation import PreparationHandle
from llmperf.common import (
    RAY_ACTOR_CPUS_ENV,
    RAY_ACTOR_CONCURRENCY_ENV,
    actor_concurrency,
    request_actor_count,
)
from llmperf.utils import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.persistence import FAILED, SUCCEEDED
from llmperf_backend.request_pool import RouterHandle, SharedRequestExecutor

if TYPE_CHECKING:
    from llmperf.token_benchmark_ray import TokenizerEncoder

WORKER_DATASET_PATH = "LLMPERF_DATASET_PATH"


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


def benchmark_actor_count(
    benchmark: Mapping[str, Any], requests_per_actor: int = 1
) -> int:
    """Return the exact maximum number of request actors created by a Runner."""

    concurrent = int(benchmark["concurrent_requests"])
    cache_probe = benchmark.get("cache_probe")
    if cache_probe:
        request_slots = max(1, min(concurrent, int(cache_probe["trials"])))
    else:
        request_slots = max(
            1, min(concurrent, int(benchmark["max_completed_requests"]))
        )
    return request_actor_count(request_slots, requests_per_actor)


def runtime_environment(environment: Mapping[str, str]) -> Dict[str, str]:
    """Keep Provider credentials but exclude Backend control-plane secrets."""

    allowed_llmperf_variables = {
        RAY_ACTOR_CPUS_ENV,
        RAY_ACTOR_CONCURRENCY_ENV,
        TOKENIZER_FAST,
        TOKENIZER_PATH,
        WORKER_DATASET_PATH,
        "LLMPERF_REQUEST_API_BASE",
        "LLMPERF_REQUEST_API_KEY",
    }
    return {
        name: value
        for name, value in environment.items()
        if name != "DATABASE_URL"
        and (not name.startswith("LLMPERF_") or name in allowed_llmperf_variables)
    }


def _calculate(
    benchmark: Dict[str, Any],
    execution_runtime: Optional[Mapping[str, Any]] = None,
    request_router: Optional[RouterHandle] = None,
    execution_environment: Optional[Mapping[str, str]] = None,
    tokenizer_override: Optional[TokenizerEncoder] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    """Run one benchmark inside an already connected Ray worker process."""

    from llmperf.prompt_datasets import PromptDatasetSource
    from llmperf.token_benchmark_ray import get_token_throughput_latencies

    worker_environment = (
        os.environ if execution_environment is None else execution_environment
    )

    dataset = benchmark.get("dataset")
    if dataset is None:
        prompt_dataset_source = PromptDatasetSource.builtin_sonnet()
    else:
        dataset_path = worker_environment.get(WORKER_DATASET_PATH)
        if not dataset_path:
            raise RuntimeError(
                f"{WORKER_DATASET_PATH} is required for dataset workloads"
            )
        prompt_dataset_source = PromptDatasetSource.external(
            dataset["adapter"], dataset_path, filename=dataset["filename"]
        )
    summary, requests = get_token_throughput_latencies(
        model=benchmark["model"],
        llm_api=benchmark["adapter"],
        test_timeout_s=benchmark["timeout_seconds"],
        max_num_completed_requests=benchmark["max_completed_requests"],
        num_concurrent_requests=benchmark["concurrent_requests"],
        mean_input_tokens=benchmark["mean_input_tokens"],
        stddev_input_tokens=benchmark["stddev_input_tokens"],
        shared_prefix_tokens=benchmark.get("shared_prefix_tokens", 0),
        prompt_dataset_source=prompt_dataset_source,
        dataset_prompt_mode=benchmark.get("dataset_prompt_mode", "sample"),
        dataset_repeat_count=benchmark.get("dataset_repeat_count", 1),
        dataset_seed=benchmark.get("dataset_seed", 11111),
        mean_output_tokens=benchmark["mean_output_tokens"],
        stddev_output_tokens=benchmark["stddev_output_tokens"],
        additional_sampling_params=benchmark["additional_sampling_params"],
        cache_probe=benchmark.get("cache_probe"),
        tokenizer_provenance=benchmark.get("tokenizer"),
        task_context=benchmark.get("task_context"),
        task_contexts=benchmark.get("task_contexts"),
        output_token_count=benchmark.get("output_token_count", "provider"),
        request_executor=(
            SharedRequestExecutor(
                request_router,
                str((execution_runtime or {})["runner_id"]),
                runtime_environment(worker_environment),
            )
            if request_router is not None
            else None
        ),
        conversation_messages=benchmark.get("conversation_messages"),
        tokenizer_override=tokenizer_override,
        preparation_actors=preparation_actors,
        preparation_chunk_size=int(
            (execution_runtime or {}).get("preparation_chunk_size", 16)
        ),
        preparation_buffer_chunks=int(
            (execution_runtime or {}).get("preparation_buffer_chunks", 8)
        ),
    )
    runtime = dict(execution_runtime or {})
    runtime.setdefault("backend", "ray")
    runtime.setdefault("worker_kind", "ray_task")
    runtime.setdefault(
        "ray_actor_num_cpus", float(os.environ.get(RAY_ACTOR_CPUS_ENV, "1.0"))
    )
    runtime.setdefault("ray_actor_max_concurrency", actor_concurrency())
    summary["execution_runtime"] = runtime
    return summary, requests


def execute_worker_task(
    benchmark: Dict[str, Any],
    execution_runtime: Mapping[str, Any],
    log_limit: int,
    request_router: Optional[RouterHandle] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
) -> Dict[str, Any]:
    """Ray task entrypoint; it never connects to PostgreSQL or initializes Ray."""

    stdout = _TailBuffer(log_limit)
    stderr = _TailBuffer(log_limit)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            summary, requests = (
                _calculate(
                    benchmark,
                    execution_runtime,
                    request_router,
                    preparation_actors=preparation_actors,
                )
                if preparation_actors
                else (
                    _calculate(benchmark, execution_runtime, request_router)
                    if request_router is not None
                    else _calculate(benchmark, execution_runtime)
                )
            )
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
        self.preparation_actors: Sequence[PreparationHandle] = ()
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
        request_router: Optional[RouterHandle] = None,
    ) -> None:
        if self.task_ref is not None:
            raise RuntimeError(f"Worker {self.runner_id} has already started")
        configured_function = self.remote_function.options(
            name=f"llmperf-worker-{self.runner_id}",
            runtime_env={"env_vars": runtime_environment(environment)},
        )
        if self.preparation_actors:
            self.task_ref = configured_function.remote(
                dict(benchmark),
                dict(execution_runtime),
                log_limit,
                request_router,
                self.preparation_actors,
            )
        else:
            self.task_ref = configured_function.remote(
                dict(benchmark), dict(execution_runtime), log_limit, request_router
            )

    def ready(self) -> bool:
        if self.task_ref is None:
            return False
        ready, _ = self.ray.wait([self.task_ref], timeout=0)
        return bool(ready)

    def result(self) -> Dict[str, Any]:
        if self.task_ref is None:
            raise RuntimeError(f"Worker {self.runner_id} has not started")
        return self.ray.get(self.task_ref)

    async def result_async(self) -> Dict[str, Any]:
        """Wait for the Ray result without blocking the Scheduler event loop."""

        task_ref = self.task_ref
        if task_ref is None:
            raise RuntimeError(f"Worker {self.runner_id} has not started")
        result_future: Future[Dict[str, Any]] = Future()

        def wait_for_result() -> None:
            if not result_future.set_running_or_notify_cancel():
                return
            try:
                result = self.ray.get(task_ref)
            except Exception as exc:
                result_future.set_exception(exc)
            else:
                result_future.set_result(result)

        result_listener = threading.Thread(
            target=wait_for_result,
            name=f"llmperf-worker-result-{self.runner_id}",
            daemon=True,
        )
        result_listener.start()
        # The bridge safely ignores late results after cancellation or loop closure.
        return await asyncio.wrap_future(result_future)

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


_TOKENIZER_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def _cached_tokenizer(path: str, use_fast: bool) -> TokenizerEncoder:
    from llmperf.utils import _load_tokenizer

    return _load_tokenizer(path, local_files_only=True, use_fast=use_fast)


def _prepared_tokenizer(path: str, use_fast: bool) -> TokenizerEncoder:
    with _TOKENIZER_LOCK:
        return _cached_tokenizer(path, use_fast)


class AtomicWorker(Worker):
    """Lightweight local handle for one atomic invoke; Ray owns only HTTP execution.

    Explicit environment and tokenizer inputs avoid per-invoke Ray processes and
    process-global configuration mutation. Output tokenization remains opt-in.
    """

    def __init__(
        self,
        ray_module: Any,
        remote_function: Any,
        runner_id: str,
        actor_count: int,
        actor_num_cpus: float,
        executor: Executor,
    ):
        super().__init__(
            ray_module, remote_function, runner_id, actor_count, actor_num_cpus
        )
        self.executor = executor
        self.local_task: Optional[asyncio.Future[Dict[str, Any]]] = None

    def start(
        self,
        benchmark: Dict[str, Any],
        environment: Mapping[str, str],
        execution_runtime: Mapping[str, Any],
        log_limit: int,
        request_router: Optional[RouterHandle] = None,
    ) -> None:
        if self.local_task is not None or request_router is None:
            raise RuntimeError(
                "Atomic Worker requires an unused handle and shared router"
            )

        def calculate() -> Dict[str, Any]:
            try:
                tokenizer = _prepared_tokenizer(
                    environment[TOKENIZER_PATH],
                    environment.get(TOKENIZER_FAST, "true").lower() == "true",
                )
                summary, requests = _calculate(
                    benchmark, execution_runtime, request_router, environment, tokenizer
                )
                return {
                    "ok": True,
                    "summary": summary,
                    "requests": list(requests),
                    "stdout": "",
                    "stderr": "",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stdout": "",
                    "stderr": traceback.format_exc()[-log_limit:],
                }

        self.local_task = asyncio.get_running_loop().run_in_executor(
            self.executor, calculate
        )

    def ready(self) -> bool:
        return self.local_task is not None and self.local_task.done()

    async def result_async(self) -> Dict[str, Any]:
        if self.local_task is None:
            raise RuntimeError("Atomic Worker has not started")
        return await asyncio.shield(self.local_task)

    def task_id(self) -> Optional[str]:
        return f"invoke:{self.runner_id}"

    def cancel(self, force: bool = False) -> None:
        # Scheduler cancels queued router calls. Executing calls retain physical
        # capacity until their HTTP operation ends, even if this handle is dropped.
        return None

    def close(self) -> None:
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

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

from llmperf_backend.compiler.requests import planned_worker_count
from llmperf.metrics import constants as common_metrics
from llmperf.datasets.preparation import PreparationHandle
from llmperf.clients.factory import (
    RAY_ACTOR_CPUS_ENV,
    RAY_ACTOR_CONCURRENCY_ENV,
    actor_concurrency,
    request_actor_count,
)
from llmperf.datasets.tokenizer import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.storage.repository import FAILED, SUCCEEDED
from llmperf_backend.runtime.request_pool import RouterHandle

if TYPE_CHECKING:
    from llmperf.benchmark.throughput import TokenizerEncoder

WORKER_DATASET_PATH = "LLMPERF_DATASET_PATH"

INPUT_DATASET_PATHS = "LLMPERF_INPUT_DATASET_PATHS"


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

    concurrent = int(
        benchmark.get("concurrent_workers") or planned_worker_count(benchmark)
    )
    cache_probe = benchmark.get("cache_probe")
    if cache_probe:
        request_slots = max(1, min(concurrent, int(cache_probe["trials"])))
    else:
        request_slots = max(1, min(concurrent, planned_worker_count(benchmark)))
    return request_actor_count(request_slots, requests_per_actor)


def runtime_environment(environment: Mapping[str, str]) -> Dict[str, str]:
    """Keep Provider credentials but exclude Backend control-plane secrets."""

    allowed_llmperf_variables = {
        RAY_ACTOR_CPUS_ENV,
        RAY_ACTOR_CONCURRENCY_ENV,
        TOKENIZER_FAST,
        TOKENIZER_PATH,
        WORKER_DATASET_PATH,
        INPUT_DATASET_PATHS,
        "LLMPERF_REQUEST_API_BASE",
        "LLMPERF_REQUEST_API_KEY",
    }
    return {
        name: value
        for name, value in environment.items()
        if name != "DATABASE_URL"
        and (not name.startswith("LLMPERF_") or name in allowed_llmperf_variables)
    }


def execute_worker_task(
    benchmark: Dict[str, Any],
    execution_runtime: Mapping[str, Any],
    log_limit: int,
    request_router: Optional[RouterHandle] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
) -> Dict[str, Any]:
    """Ray task entrypoint; it never connects to PostgreSQL or initializes Ray."""

    from llmperf_backend.runtime.preparation import calculate

    stdout = _TailBuffer(log_limit)
    stderr = _TailBuffer(log_limit)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            summary, requests = (
                calculate(
                    benchmark,
                    execution_runtime,
                    request_router,
                    preparation_actors=preparation_actors,
                )
                if preparation_actors
                else (
                    calculate(benchmark, execution_runtime, request_router)
                    if request_router is not None
                    else calculate(benchmark, execution_runtime)
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
    from llmperf.datasets.tokenizer import _load_tokenizer

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
                from llmperf_backend.runtime.preparation import calculate

                summary, requests = calculate(
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

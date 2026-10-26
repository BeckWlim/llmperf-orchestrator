"""Runner-owned preparation/aggregation bridge; it never executes Provider calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from concurrent.futures import Executor
from datetime import datetime, timezone
import threading
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from llmperf.benchmark.execution import (
    RequestStep,
    PreparedBatch,
    BenchmarkStages,
    BenchmarkResult,
)
from llmperf.clients.base import LLMResponse
from llmperf.clients.factory import actor_concurrency
from llmperf.datasets.preparation import PreparationHandle
from llmperf.datasets.tokenizer import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.runtime.dispatch import WorkerDispatcher
from llmperf_backend.runtime.admission import RunnerAdmission
from llmperf_backend.runtime.request_pool import RouterHandle
from llmperf_backend.runtime.runner import (
    RunnerExecutionLog,
    RunnerRuntime,
    WorkflowTaskLogic,
)
from llmperf_backend.runtime.worker import (
    WORKER_DATASET_PATH,
    INPUT_DATASET_PATHS,
    RAY_ACTOR_CPUS_ENV,
    _prepared_tokenizer,
    runtime_environment,
)
from llmperf_backend.compiler.requests import (
    compile_requests,
    request_plan,
    planned_worker_count,
)
from llmperf_backend.configuration.models import InputTemplateConfig
from llmperf.benchmark.throughput import TokenizerEncoder
from llmperf.benchmark.execution import BatchExecutor
from llmperf.configuration.duration import duration_seconds
from llmperf.datasets.prompts import PromptDatasetSource, prepare_prompt_requests
from llmperf_backend.runtime.request_pool import SharedRequestExecutor


PreparedInput = tuple[tuple[str, int], dict[str, Any]]


def prepare_named_inputs(
    benchmark: Mapping[str, Any],
    environment: Mapping[str, str],
    tokenizer: TokenizerEncoder,
) -> Sequence[PreparedInput] | None:
    definitions = benchmark.get("inputs") or {}
    if not definitions or benchmark.get("steps") is None:
        return None
    specifications = request_plan(benchmark).expand()
    names = [step.input or "default" for step in specifications]
    # Retain the existing bounded preparation path when every Worker uses one input.
    if len(set(names)) == 1 and names[0] == benchmark.get("input", "default"):
        return None
    dataset_paths = json.loads(environment.get(INPUT_DATASET_PATHS, "{}"))
    prepared_by_name: dict[str, list[PreparedInput]] = {}
    for name in dict.fromkeys(names):
        definition = InputTemplateConfig.model_validate(definitions[name])
        count = names.count(name)
        if definition.text is not None:
            text = definition.text
            prepared_by_name[name] = [
                (
                    (text, len(tokenizer.encode(text))),
                    {"source": "literal", "input_name": name},
                )
                for _ in range(count)
            ]
            continue
        dataset = definition.dataset
        source = (
            PromptDatasetSource.builtin_sonnet()
            if dataset is None
            else PromptDatasetSource.external(
                dataset.adapter, dataset_paths[name], filename=dataset.filename
            )
        )
        seed_material = (
            f"{benchmark.get('dataset_seed', 11111)}:{definition.dataset_seed}:{name}"
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode()).digest()[:8], "big"
        )
        family_prefix_seed = benchmark.get("shared_prefix_seed")
        named_prefix_seed = (
            int.from_bytes(
                hashlib.sha256(f"{family_prefix_seed}:{name}".encode()).digest()[:8],
                "big",
            )
            if family_prefix_seed is not None
            and (benchmark.get("task_context") or {}).get("prefix_seed_version")
            != "explicit-prefix-v1"
            else family_prefix_seed
        )
        prompts, evidence = prepare_prompt_requests(
            source=source,
            prompt_mode=definition.dataset_prompt_mode,
            num_requests=count,
            repeat_count=definition.dataset_repeat_count,
            mean_input_tokens=definition.tokens.mean,
            stddev_input_tokens=definition.tokens.stddev,
            mean_output_tokens=int(benchmark.get("mean_output_tokens", 150)),
            get_token_length=lambda value: len(tokenizer.encode(value)),
            seed=seed,
            shared_prefix_tokens=definition.shared_prefix_tokens,
            shared_prefix_seed=named_prefix_seed,
        )
        prepared_by_name[name] = [
            (prompt, {**item, "input_name": name})
            for prompt, item in zip(prompts, evidence)
        ]
    cursors = dict.fromkeys(names, 0)
    ordered: list[PreparedInput] = []
    for name in names:
        ordered.append(prepared_by_name[name][cursors[name]])
        cursors[name] += 1
    return ordered


def _calculation_options(
    benchmark: Dict[str, Any],
    execution_runtime: Optional[Mapping[str, Any]] = None,
    request_router: Optional[RouterHandle] = None,
    execution_environment: Optional[Mapping[str, str]] = None,
    tokenizer_override: Optional[TokenizerEncoder] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
    batch_executor: Optional[BatchExecutor] = None,
) -> Dict[str, Any]:
    """Resolve preparation inputs without executing Provider requests."""

    from llmperf.datasets.prompts import PromptDatasetSource

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
    named_inputs = (
        prepare_named_inputs(
            benchmark,
            worker_environment,
            tokenizer_override
            or _prepared_tokenizer(
                worker_environment[TOKENIZER_PATH],
                worker_environment.get(TOKENIZER_FAST, "true") == "true",
            ),
        )
        if benchmark.get("inputs") and benchmark.get("steps") is not None
        else None
    )
    return dict(
        model=benchmark["model"],
        llm_api=benchmark["adapter"],
        test_timeout_s=duration_seconds(benchmark["timeout"]),
        max_num_completed_requests=planned_worker_count(benchmark),
        num_concurrent_requests=(
            benchmark.get("concurrent_workers") or planned_worker_count(benchmark)
        ),
        mean_input_tokens=benchmark["mean_input_tokens"],
        stddev_input_tokens=benchmark["stddev_input_tokens"],
        shared_prefix_tokens=benchmark.get("shared_prefix_tokens", 0),
        shared_prefix_seed=benchmark.get("shared_prefix_seed"),
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
        input_messages=benchmark.get("input_messages"),
        batch_executor=batch_executor,
        literal_prompts=(
            [benchmark["prompt"] for _ in range(planned_worker_count(benchmark))]
            if named_inputs is None and benchmark.get("prompt") is not None
            else None
        ),
        prepared_inputs=named_inputs,
        tokenizer_override=tokenizer_override,
        preparation_actors=preparation_actors,
        preparation_chunk_size=int(
            (execution_runtime or {}).get("preparation_chunk_size", 16)
        ),
        preparation_buffer_chunks=int(
            (execution_runtime or {}).get("preparation_buffer_chunks", 8)
        ),
    )


def _attach_runtime(
    summary: Dict[str, Any], execution_runtime: Optional[Mapping[str, Any]]
) -> None:
    runtime = dict(execution_runtime or {})
    runtime.setdefault("backend", "ray")
    runtime.setdefault("worker_kind", "ray_task")
    runtime.setdefault(
        "ray_actor_num_cpus", float(os.environ.get(RAY_ACTOR_CPUS_ENV, "1.0"))
    )
    runtime.setdefault("ray_actor_max_concurrency", actor_concurrency())
    summary["execution_runtime"] = runtime


def calculate(
    benchmark: Dict[str, Any],
    execution_runtime: Optional[Mapping[str, Any]] = None,
    request_router: Optional[RouterHandle] = None,
    execution_environment: Optional[Mapping[str, str]] = None,
    tokenizer_override: Optional[TokenizerEncoder] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
    batch_executor: Optional[BatchExecutor] = None,
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    from llmperf.benchmark.throughput import get_token_throughput_latencies

    options = _calculation_options(
        benchmark,
        execution_runtime,
        request_router,
        execution_environment,
        tokenizer_override,
        preparation_actors,
        batch_executor,
    )
    summary, requests = get_token_throughput_latencies(**options)
    _attach_runtime(summary, execution_runtime)
    return summary, requests


def aggregate_workers(
    benchmark: Mapping[str, Any],
    responses: List[LLMResponse],
    started_at: float,
    finished_at: float,
    timed_out: bool,
    cancelled_ids: Sequence[str] = (),
) -> BenchmarkResult:
    from llmperf.benchmark.throughput import metrics_summary
    from llmperf.metrics import constants
    from llmperf.metrics.cache import analyze_cache_probe

    metrics = [response[0] for response in responses]
    for item in metrics:
        item.setdefault(constants.ERROR_CODE, None)
        item.setdefault(constants.TTFT, None)
        item.setdefault(constants.E2E_LAT, 0.0)
        item.setdefault(constants.NUM_INPUT_TOKENS, None)
        item.setdefault(constants.NUM_OUTPUT_TOKENS, None)
        item.setdefault(constants.REQ_OUTPUT_THROUGHPUT, None)
    summary = {
        name: benchmark.get(name)
        for name in (
            "model",
            "mean_input_tokens",
            "stddev_input_tokens",
            "shared_prefix_tokens",
            "mean_output_tokens",
            "stddev_output_tokens",
            "additional_sampling_params",
            "tokenizer",
            "task_context",
            "task_contexts",
            "output_token_count",
            "input_token_count",
        )
    }
    summary.update(
        num_concurrent_requests=benchmark.get("concurrent_workers"),
        timed_out=timed_out,
        results=metrics_summary(metrics, started_at, finished_at),
        prompt_dataset={
            "seed": benchmark.get("dataset_seed", 11111),
            "prompt_mode": benchmark.get("dataset_prompt_mode", "sample"),
            "generation": "worker-index-v1",
        },
        duration_scope="worker_admission_preparation_and_execution",
    )
    probe = benchmark.get("cache_probe")
    if probe:
        analysis = analyze_cache_probe(
            metrics,
            bootstrap_samples=int(probe.get("bootstrap_samples", 2000)),
            confidence_level=float(probe.get("confidence_level", 0.95)),
            seed=int(benchmark.get("dataset_seed", 11111)),
            minimum_counter_coverage=float(probe.get("minimum_counter_coverage", 0.8)),
        )
        analysis["quality_flags"] = {
            "timed_out": timed_out,
            "skipped_dependency_requests": len(cancelled_ids),
            "tokenizer_mismatch_requests": sum(
                bool(item.get("tokenizer_mismatch")) for item in metrics
            ),
        }
        if analysis["quality_flags"]["tokenizer_mismatch_requests"]:
            analysis["verdict_before_quality_guard"] = analysis["verdict"]
            analysis["verdict"] = "inconclusive"
        summary.update(
            cache_probe=probe,
            cache_probe_analysis=analysis,
            cache_probe_plan={
                "requests": planned_worker_count(benchmark),
                "families": int(probe["trials"]),
                "prompt_hash_algorithm": "sha256",
                "persist_prompt_text": bool(probe.get("persist_prompt_text", False)),
                "skipped_request_ids": list(cancelled_ids),
            },
        )
    return summary, metrics


class RunnerPreparation:
    def __init__(
        self,
        runner_id: str,
        executor: Executor,
        dispatcher: WorkerDispatcher,
        admission: Optional[RunnerAdmission] = None,
    ):
        self.runner_id = runner_id
        self.executor = executor
        self.dispatcher = dispatcher
        self.admission = admission
        self.preparation_actors: Sequence[PreparationHandle] = ()
        self.local_task: Optional[asyncio.Future[Dict[str, Any]]] = None
        self._dispatch_future: Optional[asyncio.Task[List[LLMResponse]]] = None
        self._cancelled = threading.Event()
        self.phase = "creating"
        self.created_workers = 0
        self.execution_log: Optional[RunnerExecutionLog] = None

    def progress(self) -> Dict[str, Any]:
        phase = (
            "waiting"
            if self.phase == "submitting"
            and self.admission is not None
            and self.runner_id not in self.admission.active
            else self.phase
        )
        return {"phase": phase, "created": self.created_workers}

    def start(
        self,
        benchmark: Dict[str, Any],
        environment: Mapping[str, str],
        execution_runtime: Mapping[str, Any],
        log_limit: int,
        request_router: Optional[RouterHandle] = None,
    ) -> None:
        if self.local_task is not None:
            raise RuntimeError("Runner preparation already started")
        execution_log = RunnerExecutionLog(self.runner_id, log_limit)
        self.execution_log = execution_log
        loop = asyncio.get_running_loop()
        bus = self.dispatcher.register(
            self.runner_id,
            runtime_environment(environment),
            runtime_limits=execution_runtime.get("runtime_limits"),
        )

        async def execute_workers() -> Dict[str, Any]:
            from llmperf_backend.runtime.worker_input import worker_steps
            import time

            try:
                execution_log.record("runner.started")
                if self._cancelled.is_set():
                    raise asyncio.CancelledError
                self.phase = "creating"
                # Only recipes and dependencies are built here. CPU preparation is
                # performed by the request actor after the router grants its slot.
                steps = worker_steps(benchmark, calibrated=True)
                self.created_workers = len(steps)
                execution_log.record("workers.created", count=self.created_workers)
                self.phase = "submitting"
                runtime = RunnerRuntime(
                    self.runner_id,
                    WorkflowTaskLogic(steps, datetime.now(timezone.utc)),
                    self.dispatcher,
                    bus,
                    duration_seconds(benchmark["timeout"]),
                    max_outstanding_workers=benchmark.get("concurrent_workers"),
                    admission=self.admission,
                    execution_log=execution_log,
                    cooldown_seconds=(
                        duration_seconds(benchmark["every"])
                        if benchmark.get("every") is not None
                        else None
                    ),
                )
                started_at = time.monotonic()
                self._dispatch_future = asyncio.create_task(runtime.run())
                responses = await self._dispatch_future
                finished_at = time.monotonic()
                self.phase = "aggregating"
                execution_log.record("runner.aggregating", completed=len(responses))
                summary, requests = await loop.run_in_executor(
                    self.executor,
                    aggregate_workers,
                    benchmark,
                    responses,
                    started_at,
                    finished_at,
                    runtime.timed_out,
                    tuple(runtime.cancelled_steps),
                )
                _attach_runtime(summary, execution_runtime)
                summary["worker_creation"] = {"created": self.created_workers}
                self.phase = "finished"
                execution_log.record(
                    "runner.finished",
                    succeeded=sum(
                        result.succeeded for result in runtime.results.values()
                    ),
                    failed=sum(
                        not result.succeeded for result in runtime.results.values()
                    ),
                    cancelled=len(runtime.cancelled_steps),
                    timed_out=runtime.timed_out,
                    elapsed_seconds=finished_at - started_at,
                )
                return {
                    "ok": True,
                    "summary": summary,
                    "requests": requests,
                    "stdout": execution_log.text(),
                    "stderr": "",
                }
            except asyncio.CancelledError:
                self.phase = "cancelled"
                execution_log.record("runner.cancelled")
                raise
            except Exception as exc:
                self.phase = "failed"
                execution_log.record("runner.failed", error_type=type(exc).__name__)
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stdout": execution_log.text(),
                    "stderr": traceback.format_exc()[-log_limit:],
                }

        self.local_task = asyncio.create_task(execute_workers())

    def captured_stdout(self) -> str:
        return self.execution_log.text() if self.execution_log is not None else ""

    def ready(self) -> bool:
        return self.local_task is not None and self.local_task.done()

    async def result_async(self) -> Dict[str, Any]:
        if self.local_task is None:
            raise RuntimeError("Runner preparation not started")
        return await asyncio.shield(self.local_task)

    def task_id(self) -> str:
        return f"runner:{self.runner_id}"

    def cancel(self, force: bool = False) -> None:
        self._cancelled.set()
        if self._dispatch_future is not None:
            self._dispatch_future.cancel()
        if self.local_task is not None:
            self.local_task.cancel()

    def close(self) -> None:
        self.cancel()

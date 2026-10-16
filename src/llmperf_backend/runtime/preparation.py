"""Runner-owned preparation/aggregation bridge; it never executes Provider calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from concurrent.futures import Executor, Future
from datetime import datetime, timezone
import threading
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.base import LLMResponse
from llmperf.clients.factory import actor_concurrency
from llmperf.datasets.preparation import PreparationHandle
from llmperf.datasets.tokenizer import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.runtime.dispatch import WorkerDispatcher
from llmperf_backend.runtime.request_pool import RouterHandle
from llmperf_backend.runtime.runner import RunnerRuntime, WorkflowTaskLogic
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


def calculate(
    benchmark: Dict[str, Any],
    execution_runtime: Optional[Mapping[str, Any]] = None,
    request_router: Optional[RouterHandle] = None,
    execution_environment: Optional[Mapping[str, str]] = None,
    tokenizer_override: Optional[TokenizerEncoder] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
    batch_executor: Optional[BatchExecutor] = None,
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    """Run one benchmark inside an already connected Ray worker process."""

    from llmperf.datasets.prompts import PromptDatasetSource
    from llmperf.benchmark.throughput import get_token_throughput_latencies

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
    summary, requests = get_token_throughput_latencies(
        model=benchmark["model"],
        llm_api=benchmark["adapter"],
        test_timeout_s=duration_seconds(benchmark["timeout"]),
        max_num_completed_requests=planned_worker_count(benchmark),
        num_concurrent_requests=(
            benchmark.get("concurrent_requests") or planned_worker_count(benchmark)
        ),
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
    runtime = dict(execution_runtime or {})
    runtime.setdefault("backend", "ray")
    runtime.setdefault("worker_kind", "ray_task")
    runtime.setdefault(
        "ray_actor_num_cpus", float(os.environ.get(RAY_ACTOR_CPUS_ENV, "1.0"))
    )
    runtime.setdefault("ray_actor_max_concurrency", actor_concurrency())
    summary["execution_runtime"] = runtime
    return summary, requests


class RunnerPreparation:
    def __init__(
        self, runner_id: str, executor: Executor, dispatcher: WorkerDispatcher
    ):
        self.runner_id = runner_id
        self.executor = executor
        self.dispatcher = dispatcher
        self.preparation_actors: Sequence[PreparationHandle] = ()
        self.local_task: Optional[asyncio.Future[Dict[str, Any]]] = None
        self._dispatch_future: Optional[Future[List[LLMResponse]]] = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

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
        loop = asyncio.get_running_loop()
        bus = self.dispatcher.register(self.runner_id, runtime_environment(environment))

        async def run_steps(
            steps: Sequence[RequestStep], window: int, timeout_seconds: float
        ) -> List[LLMResponse]:
            resolved_steps = compile_requests(benchmark, steps)
            runtime = RunnerRuntime(
                self.runner_id,
                WorkflowTaskLogic(resolved_steps, datetime.now(timezone.utc)),
                self.dispatcher,
                bus,
                timeout_seconds,
                # Existing explicit workload concurrency becomes the optional limit.
                max_outstanding_workers=benchmark.get("concurrent_requests"),
            )
            try:
                return await runtime.run()
            except asyncio.CancelledError:
                failure = self.dispatcher.errors.get(self.runner_id)
                if failure is not None:
                    raise RuntimeError("Worker result publication failed") from failure
                raise

        def execute_batch(
            steps: Sequence[RequestStep], window: int, timeout_seconds: float
        ) -> List[LLMResponse]:
            with self._lock:
                if self._cancelled.is_set():
                    raise RuntimeError("Runner cancelled before submission")
                dispatch_future = asyncio.run_coroutine_threadsafe(
                    run_steps(steps, window, timeout_seconds), loop
                )
                self._dispatch_future = dispatch_future
            return dispatch_future.result()

        def prepare_and_aggregate() -> Dict[str, Any]:
            try:
                tokenizer = _prepared_tokenizer(
                    environment[TOKENIZER_PATH],
                    environment.get(TOKENIZER_FAST, "true").lower() == "true",
                )
                summary, requests = calculate(
                    benchmark,
                    execution_runtime,
                    request_router,
                    environment,
                    tokenizer,
                    preparation_actors=self.preparation_actors,
                    batch_executor=execute_batch,
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

        self.local_task = loop.run_in_executor(self.executor, prepare_and_aggregate)

    def ready(self) -> bool:
        return self.local_task is not None and self.local_task.done()

    async def result_async(self) -> Dict[str, Any]:
        if self.local_task is None:
            raise RuntimeError("Runner preparation not started")
        return await asyncio.shield(self.local_task)

    def task_id(self) -> str:
        return f"runner:{self.runner_id}"

    def cancel(self, force: bool = False) -> None:
        with self._lock:
            self._cancelled.set()
            if self._dispatch_future is not None:
                self._dispatch_future.cancel()

    def close(self) -> None:
        self.cancel()

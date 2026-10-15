"""Token benchmark orchestration used by the packaged Worker and module CLI."""

import threading
import argparse
from datetime import datetime, timezone
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import re
import time
import random
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Sequence

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.cache_analysis import analyze_cache_probe, summarize_cache_counters
from llmperf.cache_probe import (
    CacheProbeRequest,
    DependentPlanQueue,
    build_cache_probe_plan,
)
from llmperf.common import (
    SUPPORTED_APIS,
    actor_concurrency,
    construct_clients,
    request_actor_count,
)

from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMResponse
from llmperf.prompt_datasets import PromptDatasetSource, prepare_prompt_requests
from llmperf.requests_launcher import RequestsLauncher
from llmperf.preparation import PreparationHandle, PreparedStream, calibrate


class TokenizerEncoder(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]: ...


from llmperf.utils import (
    get_tokenizer,
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm


def normalize_request_metrics(
    request_metrics: Dict[str, Any],
    generated_text: str,
    get_token_length: Optional[Callable[[str], int]] = None,
    tokenizer_divergence_warning_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Use server counts by default; an explicit callback enables local output counts.

    Call only after the timed execution phase. Missing output usage is unknown,
    never a stream-event count or the requested output budget.
    """

    local_output_tokens = (
        get_token_length(generated_text) if get_token_length is not None else None
    )
    num_output_tokens = (
        local_output_tokens
        if get_token_length is not None
        else request_metrics.get(common_metrics.PROVIDER_OUTPUT_TOKENS)
    )
    request_metrics[common_metrics.LOCAL_OUTPUT_TOKENS] = local_output_tokens
    request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
    provider_input = request_metrics.get(common_metrics.PROVIDER_INPUT_TOKENS)
    local_input = request_metrics.get(
        common_metrics.LOCAL_INPUT_TOKENS,
        request_metrics.get(common_metrics.NUM_INPUT_TOKENS),
    )
    num_input_tokens = provider_input if provider_input is not None else local_input
    request_metrics[common_metrics.LOCAL_INPUT_TOKENS] = local_input
    request_metrics[common_metrics.NUM_INPUT_TOKENS] = num_input_tokens
    request_metrics[common_metrics.TOKEN_COUNT_SOURCES] = {
        "input": (
            "provider"
            if provider_input is not None
            else "local_prompt" if local_input is not None else "unavailable"
        ),
        "output": (
            "local"
            if get_token_length is not None
            else "provider" if num_output_tokens is not None else "unavailable"
        ),
    }
    request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
        num_input_tokens + num_output_tokens
        if num_input_tokens is not None and num_output_tokens is not None
        else None
    )
    end_to_end_latency = request_metrics.get(common_metrics.E2E_LAT, 0)
    request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
        (num_output_tokens / end_to_end_latency if end_to_end_latency > 0 else 0)
        if num_output_tokens is not None
        else None
    )
    if isinstance(provider_input, (int, float)) and isinstance(
        local_input, (int, float)
    ):
        divergence = (
            abs(provider_input - local_input) / provider_input
            if provider_input
            else (0 if local_input == 0 else 1)
        )
        request_metrics[common_metrics.TOKENIZER_DIVERGENCE] = divergence
        if divergence > tokenizer_divergence_warning_ratio:
            request_metrics["tokenizer_mismatch"] = True
    if num_output_tokens is None or num_output_tokens <= 1:
        request_metrics[common_metrics.TPOT] = None
    elif get_token_length is not None or common_metrics.TPOT not in request_metrics:
        ttft = request_metrics.get(common_metrics.TTFT)
        e2e = request_metrics.get(common_metrics.E2E_LAT)
        if (
            isinstance(ttft, (int, float))
            and ttft >= 0
            and isinstance(e2e, (int, float))
        ):
            request_metrics[common_metrics.TPOT] = max(0, e2e - ttft) / (
                num_output_tokens - 1
            )
        else:
            request_metrics[common_metrics.TPOT] = None
    return request_metrics


def _execute_cache_probe(
    plan: List[CacheProbeRequest],
    model: str,
    llm_api: str,
    num_concurrent_requests: int,
    num_output_tokens_list: List[int],
    additional_sampling_params: Dict[str, Any],
    test_timeout_s: float,
    get_token_length,
    cache_probe: Dict[str, Any],
    dataset_seed: int,
    tokenizer_provenance: Optional[Dict[str, Any]],
    benchmark_metadata: Dict[str, Any],
    prompt_evidence: Optional[List[Dict[str, Any]]] = None,
    output_token_count: str = "provider",
    request_executor: Optional[Callable[[RequestConfig], LLMResponse]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute a dependency-aware probe without changing the global Scheduler."""

    started = time.monotonic()
    deadline = started + test_timeout_s
    queue = DependentPlanQueue(plan)
    completed: List[Dict[str, Any]] = []
    pending_metrics: List[Tuple[Dict[str, Any], str]] = []
    errors = []
    lock = threading.Lock()
    dispatch_counter = 0
    completion_counter = 0
    pbar = tqdm(total=len(plan))
    thread_count = min(max(1, num_concurrent_requests), int(cache_probe["trials"]))
    requests_per_actor = actor_concurrency()
    clients = (
        []
        if request_executor is not None
        else construct_clients(
            llm_api=llm_api,
            num_clients=request_actor_count(thread_count, requests_per_actor),
        )
    )

    def execute_slot(slot_index: int) -> None:
        nonlocal dispatch_counter, completion_counter
        try:
            actor_index = slot_index // requests_per_actor
            launcher = RequestsLauncher(
                clients[actor_index : actor_index + 1], request_executor
            )
            while time.monotonic() < deadline:
                planned = queue.claim(deadline=deadline)
                if planned is None:
                    return
                with lock:
                    actual_dispatch_index = dispatch_counter
                    dispatch_counter += 1
                dispatched_at = time.monotonic()
                metadata = planned.metadata()
                if prompt_evidence:
                    family_index = int(planned.family_id.rsplit("-", 1)[-1])
                    metadata["prompt_evidence"] = dict(prompt_evidence[family_index])
                metadata["plan_index"] = planned.dispatch_index
                metadata["dispatch_index"] = actual_dispatch_index
                metadata["scheduled_monotonic"] = started
                metadata["dispatched_monotonic"] = dispatched_at
                sampling_params = {
                    "max_tokens": num_output_tokens_list[planned.dispatch_index]
                }
                sampling_params.update(additional_sampling_params)
                config = RequestConfig(
                    model=model,
                    prompt=(planned.prompt, planned.local_input_tokens),
                    sampling_params=sampling_params,
                    llm_api=llm_api,
                    metadata=metadata,
                    timeout_seconds=test_timeout_s,
                )
                launcher.launch_requests(config)
                outputs = launcher.get_next_ready(block=True)
                if not outputs:
                    raise RuntimeError("Cache probe request completed without a result")
                request_succeeded = True
                for request_metrics, generated_text, returned_config in outputs:
                    request_metrics[common_metrics.REQUEST_METADATA] = dict(
                        request_metrics.get(common_metrics.REQUEST_METADATA)
                        or returned_config.metadata
                        or metadata
                    )
                    with lock:
                        request_metrics[common_metrics.REQUEST_METADATA][
                            "completion_index"
                        ] = completion_counter
                        completion_counter += 1
                        completed.append(request_metrics)
                        pending_metrics.append(
                            (
                                request_metrics,
                                generated_text if output_token_count == "local" else "",
                            )
                        )
                        pbar.update(1)
                    request_succeeded = request_succeeded and (
                        request_metrics.get(common_metrics.ERROR_CODE) is None
                    )
                queue.complete(planned, request_succeeded)
        except Exception as exc:
            queue.close()
            with lock:
                if not errors:
                    errors.append((exc, exc.__traceback__))

    threads = [
        threading.Thread(target=execute_slot, args=(slot_index,))
        for slot_index in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    pbar.close()
    if errors:
        error, traceback = errors[0]
        raise error.with_traceback(traceback)

    finished = time.monotonic()
    for request_metrics, generated_text in pending_metrics:
        normalize_request_metrics(
            request_metrics,
            generated_text,
            get_token_length if output_token_count == "local" else None,
            float(cache_probe.get("tokenizer_divergence_warning_ratio", 0.05)),
        )
    completed.sort(
        key=lambda request: request.get(common_metrics.REQUEST_METADATA, {}).get(
            "completion_index", 0
        )
    )
    results = metrics_summary(completed, started, finished)
    analysis = analyze_cache_probe(
        completed,
        bootstrap_samples=int(cache_probe.get("bootstrap_samples", 2_000)),
        confidence_level=float(cache_probe.get("confidence_level", 0.95)),
        seed=dataset_seed,
        minimum_counter_coverage=float(
            cache_probe.get("minimum_counter_coverage", 0.8)
        ),
    )
    mismatches = sum(bool(item.get("tokenizer_mismatch")) for item in completed)
    analysis["quality_flags"] = {
        "tokenizer_mismatch_requests": mismatches,
        "skipped_dependency_requests": len(queue.skipped),
        "timed_out": finished >= deadline and len(completed) < len(plan),
    }
    if mismatches:
        analysis["verdict_before_quality_guard"] = analysis["verdict"]
        analysis["verdict"] = "inconclusive"

    provenance: Dict[str, Any] = dict(tokenizer_provenance or {})
    if not provenance:
        provenance = {
            "id": "hf-internal-testing/llama-tokenizer",
            "selection": "global_default",
            "accuracy": "approximate",
        }
    provenance["warning"] = (
        "Tokenizer was not explicitly selected for this model"
        if provenance.get("accuracy") == "approximate"
        else None
    )
    metadata = {
        "model": model,
        **benchmark_metadata,
        "output_token_count": output_token_count,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
        "cache_probe": cache_probe,
        "cache_probe_plan": {
            "requests": len(plan),
            "families": int(cache_probe["trials"]),
            "prompt_hash_algorithm": "sha256",
            "persist_prompt_text": bool(cache_probe.get("persist_prompt_text", False)),
            "skipped_request_ids": [item.request_id for item in queue.skipped],
        },
        "tokenizer": provenance,
        "timed_out": finished >= deadline and len(completed) < len(plan),
        "cache_probe_analysis": analysis,
        "results": results,
    }
    return metadata, completed


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    prompt_dataset_source: PromptDatasetSource,
    shared_prefix_tokens: int = 0,
    dataset_prompt_mode: str = "sample",
    dataset_repeat_count: int = 1,
    dataset_seed: int = 11111,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s: float = 90,
    llm_api="openai",
    cache_probe: Optional[Dict[str, Any]] = None,
    tokenizer_provenance: Optional[Dict[str, Any]] = None,
    task_context: Optional[Dict[str, Any]] = None,
    output_token_count: str = "provider",
    task_contexts: Optional[List[Dict[str, Any]]] = None,
    request_executor: Optional[Callable[[RequestConfig], LLMResponse]] = None,
    conversation_messages: Optional[List[Dict[str, str]]] = None,
    tokenizer_override: Optional[TokenizerEncoder] = None,
    preparation_actors: Sequence[PreparationHandle] = (),
    preparation_chunk_size: int = 16,
    preparation_buffer_chunks: int = 8,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        shared_prefix_tokens: Number of leading prompt tokens reused by every request.
        prompt_dataset_source: Explicit prompt adapter and optional artifact path.
        dataset_prompt_mode: Select whole prompts or concatenate diverse records to the
            requested token budget.
        dataset_repeat_count: Number of times to issue every sampled dataset prompt.
        dataset_seed: Seed used to select and order dataset prompts.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    if output_token_count not in {"provider", "local"}:
        raise ValueError("output_token_count must be provider or local")
    if task_contexts is not None and (
        task_context
        or cache_probe
        or len(task_contexts) != max_num_completed_requests
        or len(task_contexts) != num_concurrent_requests
        or len(task_contexts) < 2
    ):
        raise ValueError(
            "grouped task contexts require one concurrent request per node"
        )
    atomic_context = (
        task_context
        if task_context and task_context.get("execution_kind", "atomic") == "atomic"
        else None
    )
    task_payload_seed = atomic_context.get("payload_seed") if atomic_context else None
    effective_dataset_seed = (
        int(task_payload_seed) if task_payload_seed is not None else dataset_seed
    )
    random.seed(
        effective_dataset_seed
        if cache_probe or task_payload_seed is not None
        else 11111
    )

    tokenizer = (
        tokenizer_override if tokenizer_override is not None else get_tokenizer()
    )
    get_token_length = lambda text: len(
        tokenizer.encode(text, add_special_tokens=False)
    )

    sampling_parameters = dict(additional_sampling_params or {})

    completed_requests_lock = threading.Lock()
    completed_requests = []
    pending_metrics: List[Tuple[Dict[str, Any], str]] = []
    num_completed_requests = 0
    request_threads_stop = threading.Event()
    request_thread_errors = []
    request_thread_errors_lock = threading.Lock()
    if shared_prefix_tokens >= mean_input_tokens:
        raise ValueError("shared_prefix_tokens must be less than mean_input_tokens")
    if (
        prompt_dataset_source.is_external
        and shared_prefix_tokens
        and dataset_prompt_mode != "concatenate"
    ):
        raise ValueError(
            "external dataset shared_prefix_tokens requires concatenate mode"
        )
    if dataset_prompt_mode not in {"sample", "concatenate"}:
        raise ValueError(f"Unsupported dataset prompt mode: {dataset_prompt_mode}")
    if dataset_repeat_count < 1:
        raise ValueError("dataset_repeat_count must be at least 1")
    if dataset_prompt_mode == "concatenate" and dataset_repeat_count != 1:
        raise ValueError("concatenated dataset prompts require repeat_count=1")

    # Make up prompts outside of the send loop for faster benchmarking. A shared
    # prefix creates a controlled provider KV-cache workload while each suffix
    # remains unique.
    probe_request_count = (
        int(cache_probe["trials"])
        * (1 + int(cache_probe.get("repeats_after_prime", 1)))
        if cache_probe
        else max_num_completed_requests
    )
    base_prompt_count = (
        int(cache_probe["trials"]) if cache_probe else max_num_completed_requests
    )
    if atomic_context and base_prompt_count != 1:
        raise ValueError("compiled task nodes must contain exactly one request")
    num_output_tokens_list = [
        sample_random_positive_int(mean_output_tokens, stddev_output_tokens)
        for _ in range(probe_request_count)
    ]
    preparations = (
        [(int(context["payload_seed"]), 1) for context in task_contexts]
        if task_contexts
        else [(effective_dataset_seed, base_prompt_count)]
    )
    prompt_stream: Optional[PreparedStream] = None
    if preparation_actors and not (
        (prompt_dataset_source.is_external and dataset_prompt_mode != "concatenate")
        or cache_probe
        or atomic_context
        or task_contexts
        or dataset_repeat_count != 1
    ):
        preparation_plan = calibrate(
            tokenizer,
            effective_dataset_seed,
            mean_input_tokens,
            stddev_input_tokens,
            shared_prefix_tokens,
            mean_output_tokens,
            source=prompt_dataset_source,
        )
        prompt_stream = PreparedStream(
            preparation_plan,
            preparation_actors,
            max_num_completed_requests,
            preparation_chunk_size,
            preparation_buffer_chunks,
        )
    prompts: List[Tuple[str, int]] = []
    prompt_evidence: List[Dict[str, Any]] = []
    for prompt_seed, prompt_count in [] if prompt_stream is not None else preparations:
        prepared_prompts, prepared_evidence = prepare_prompt_requests(
            source=prompt_dataset_source,
            prompt_mode=dataset_prompt_mode,
            num_requests=prompt_count,
            repeat_count=(1 if cache_probe else dataset_repeat_count),
            mean_input_tokens=mean_input_tokens,
            stddev_input_tokens=stddev_input_tokens,
            mean_output_tokens=mean_output_tokens,
            get_token_length=get_token_length,
            seed=prompt_seed,
            shared_prefix_tokens=shared_prefix_tokens,
        )
        prompts.extend(prepared_prompts)
        prompt_evidence.extend(prepared_evidence)
    prompt_source = (
        prompt_dataset_source.adapter
        if prompt_stream is not None
        else prompt_evidence[0]["source"]
    )
    if atomic_context and atomic_context.get("prompt_text") is not None:
        literal_prompt = str(atomic_context["prompt_text"])
        prompts = [(literal_prompt, get_token_length(literal_prompt))]
        prompt_evidence = [
            {
                "source": "literal",
                "sha256": hashlib.sha256(literal_prompt.encode()).hexdigest(),
            }
        ]
        prompt_source = "literal"
    effective_prompt_mode = (
        "calibrated-concatenate"
        if prompt_stream is not None
        else prompt_evidence[0].get("mode", dataset_prompt_mode)
    )
    prompt_dataset_metadata = {
        "adapter": prompt_dataset_source.adapter,
        "source": prompt_source,
        "prompt_mode": effective_prompt_mode,
        "repeat_count": dataset_repeat_count,
        "seed": effective_dataset_seed,
    }
    if cache_probe:
        probe_config = dict(cache_probe)
        if probe_config.get("shared_prefix_tokens") is None:
            probe_config["shared_prefix_tokens"] = shared_prefix_tokens
        plan = build_cache_probe_plan(
            prompts,
            probe_config,
            tokenizer,
            effective_dataset_seed,
        )
        return _execute_cache_probe(
            plan=plan,
            model=model,
            llm_api=llm_api,
            num_concurrent_requests=num_concurrent_requests,
            num_output_tokens_list=num_output_tokens_list,
            additional_sampling_params=sampling_parameters,
            test_timeout_s=test_timeout_s,
            get_token_length=get_token_length,
            cache_probe=probe_config,
            dataset_seed=effective_dataset_seed,
            tokenizer_provenance=tokenizer_provenance,
            benchmark_metadata={
                "mean_input_tokens": mean_input_tokens,
                "stddev_input_tokens": stddev_input_tokens,
                "shared_prefix_tokens": shared_prefix_tokens,
                "prompt_dataset": dict(prompt_dataset_metadata),
                "mean_output_tokens": mean_output_tokens,
                "stddev_output_tokens": stddev_output_tokens,
            },
            prompt_evidence=prompt_evidence,
            output_token_count=output_token_count,
            request_executor=request_executor,
        )
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests)
    worker_count = min(num_concurrent_requests, max_num_completed_requests)
    requests_per_actor = actor_concurrency()
    clients = (
        []
        if request_executor is not None
        else construct_clients(
            llm_api=llm_api,
            num_clients=request_actor_count(worker_count, requests_per_actor),
        )
    )
    next_request_index = 0

    def launch_request(thread_index):
        nonlocal num_completed_requests, next_request_index
        try:
            actor_index = thread_index // requests_per_actor
            # Each request slot owns its launcher and ObjectRefs. Several slots
            # may call the same actor, up to its configured concurrency.
            req_launcher = RequestsLauncher(
                clients[actor_index : actor_index + 1], request_executor
            )

            while (
                not request_threads_stop.is_set()
                and time.monotonic() - start_time < test_timeout_s
            ):
                if prompt_stream is not None:
                    try:
                        request_index, (request_prompt, request_evidence) = (
                            prompt_stream.take()
                        )
                    except StopIteration:
                        return
                else:
                    with completed_requests_lock:
                        if next_request_index >= max_num_completed_requests:
                            return
                        request_index = next_request_index
                        next_request_index += 1
                    request_prompt = prompts[request_index]
                    request_evidence = prompt_evidence[request_index]

                if (
                    request_threads_stop.is_set()
                    or time.monotonic() - start_time >= test_timeout_s
                ):
                    return

                default_sampling_params = dict(sampling_parameters)
                default_sampling_params.setdefault(
                    "max_tokens", num_output_tokens_list[request_index]
                )
                request_metadata: Dict[str, Any] = {
                    "prompt_evidence": dict(request_evidence)
                }
                invocation_context = (
                    task_contexts[request_index] if task_contexts else task_context
                )
                if invocation_context:
                    request_metadata.update(
                        {
                            **dict(invocation_context),
                            "prompt_hash": "sha256:"
                            + hashlib.sha256(
                                request_prompt[0].encode("utf-8")
                            ).hexdigest(),
                            "local_input_tokens": request_prompt[1],
                        }
                    )
                    payload_evidence = request_evidence
                    request_metadata["payload_evidence"] = dict(payload_evidence)
                    request_metadata.pop("prompt_text", None)
                if conversation_messages is not None:
                    default_sampling_params["messages"] = [
                        *conversation_messages,
                        {"role": "user", "content": request_prompt[0]},
                    ]
                    request_metadata["history_version"] = len(conversation_messages)
                    request_metadata["messages_hash"] = (
                        "sha256:"
                        + hashlib.sha256(
                            json.dumps(
                                default_sampling_params["messages"],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                    )
                request_config = RequestConfig(
                    model=model,
                    prompt=request_prompt,
                    sampling_params=default_sampling_params,
                    llm_api=llm_api,
                    metadata=request_metadata,
                    timeout_seconds=test_timeout_s,
                )
                invocation_started_at = datetime.now(timezone.utc).isoformat()
                invocation_started_monotonic = time.monotonic()
                try:
                    req_launcher.launch_requests(request_config)
                    outs = req_launcher.get_next_ready()
                    if task_contexts and not outs:
                        raise RuntimeError("Task request completed without a result")
                except Exception as exc:
                    if not task_contexts:
                        raise
                    outs = [
                        (
                            {
                                common_metrics.ERROR_CODE: "worker_request_error",
                                common_metrics.ERROR_MSG: str(exc),
                                common_metrics.TTFT: None,
                                common_metrics.E2E_LAT: time.monotonic()
                                - invocation_started_monotonic,
                                common_metrics.NUM_INPUT_TOKENS: request_prompt[1],
                            },
                            "",
                            request_config,
                        )
                    ]
                for out in outs:
                    request_metrics, gen_text, _ = out
                    if conversation_messages is not None:
                        request_metrics[common_metrics.LOCAL_INPUT_TOKENS] = None
                        request_metrics["conversation_turn"] = {
                            "user": request_prompt[0],
                            "assistant": gen_text,
                            "parent_version": len(conversation_messages),
                        }
                    request_metrics.setdefault(
                        common_metrics.REQUEST_METADATA, request_metadata
                    )
                    request_metrics.setdefault(
                        common_metrics.REQUEST_TIMING,
                        {
                            "client_start_utc": invocation_started_at,
                            "completed_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    with completed_requests_lock:
                        if num_completed_requests < max_num_completed_requests:
                            completed_requests.append(request_metrics)
                            pending_metrics.append(
                                (
                                    request_metrics,
                                    gen_text if output_token_count == "local" else "",
                                )
                            )
                            pbar.update(1)
                            num_completed_requests += 1
        except Exception as exc:
            # Python does not propagate exceptions from child threads to their
            # caller. Retain the first failure and stop launching more paid API
            # requests; the main benchmark thread re-raises it after joining all
            # request threads so the Worker exits non-zero.
            with request_thread_errors_lock:
                if not request_thread_errors:
                    request_thread_errors.append((exc, exc.__traceback__))
            request_threads_stop.set()

    threads = []
    for i in range(worker_count):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if prompt_stream is not None:
        prompt_stream.close()

    pbar.close()
    if request_thread_errors:
        error, traceback = request_thread_errors[0]
        raise error.with_traceback(traceback)

    end_time = time.monotonic()
    for request_metrics, generated_text in pending_metrics:
        normalize_request_metrics(
            request_metrics,
            generated_text,
            get_token_length if output_token_count == "local" else None,
        )
    timed_out = (
        end_time - start_time >= test_timeout_s
        and num_completed_requests < max_num_completed_requests
    )
    if timed_out:
        print("Test timed out before all requests could be completed.")

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "output_token_count": output_token_count,
        "stddev_input_tokens": stddev_input_tokens,
        "shared_prefix_tokens": shared_prefix_tokens,
        "prompt_dataset": dict(prompt_dataset_metadata),
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": sampling_parameters,
        "tokenizer": dict(
            tokenizer_provenance
            or {
                "id": "hf-internal-testing/llama-tokenizer",
                "selection": "global_default",
                "accuracy": "approximate",
                "warning": "Tokenizer was not explicitly selected for this model",
            }
        ),
        "task_context": dict(task_context or {}),
        "task_contexts": list(task_contexts or []),
        "timed_out": timed_out,
    }

    metadata["results"] = ret

    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], started_at: float, finished_at: float
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        started_at: Monotonic timestamp when the measurement started.
        finished_at: Monotonic timestamp when the measurement finished.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret: Dict[str, Any] = {}
    metric_keys = [
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]

    if not metrics:
        for key in metric_keys:
            ret[key] = {
                "quantiles": {
                    name: None for name in ("p25", "p50", "p75", "p90", "p95", "p99")
                },
                "mean": None,
                "min": None,
                "max": None,
                "stddev": None,
            }
        ret[common_metrics.NUM_REQ_STARTED] = 0
        ret[common_metrics.ERROR_RATE] = 0
        ret[common_metrics.NUM_ERRORS] = 0
        ret[common_metrics.ERROR_CODE_FREQ] = {}
        ret[common_metrics.OUTPUT_THROUGHPUT] = 0
        ret[common_metrics.OUTPUT_TOKEN_COVERAGE] = {
            "counted_requests": 0,
            "successful_requests": 0,
            "ratio": None,
        }
        ret[common_metrics.NUM_COMPLETED_REQUESTS] = 0
        ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = 0
        ret[common_metrics.KV_CACHE] = summarize_cache_counters([])
        return ret

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in metric_keys:
        print(key)
        ret[key] = {}
        series = pd.Series(
            list(flatten(df_without_errored_req[key])), dtype=float
        ).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = (
                float(value) if pd.notna(value) else None
            )
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = float(mean) if pd.notna(mean) else None
        print(f"    min = {series.min()}")
        ret[key]["min"] = float(series.min()) if not series.empty else None
        print(f"    max = {series.max()}")
        ret[key]["max"] = float(series.max()) if not series.empty else None
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std() if len(series) > 1 else None

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = {
        str(code): int(count) for code, count in error_codes.value_counts().items()
    }
    if num_errors:
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = error_code_frequency

    duration = finished_at - started_at
    output_counts = pd.Series(
        df_without_errored_req[common_metrics.NUM_OUTPUT_TOKENS], dtype=float
    )
    counted_requests = int(output_counts.notna().sum())
    successful_requests = len(df_without_errored_req)
    ret[common_metrics.OUTPUT_TOKEN_COVERAGE] = {
        "counted_requests": counted_requests,
        "successful_requests": successful_requests,
        "ratio": (
            counted_requests / successful_requests if successful_requests else None
        ),
    }
    # A partial sum would understate throughput while looking like a full measurement.
    overall_output_throughput = (
        (float(output_counts.sum()) / duration if duration > 0 else 0)
        if counted_requests == successful_requests
        else None
    )

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / duration * 60 if duration > 0 else 0
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    cache_records: List[Dict[str, Any]] = [
        {
            str(column): value
            for column, value in zip(df_without_errored_req.columns, row)
        }
        for row in df_without_errored_req.itertuples(index=False, name=None)
    ]
    ret[common_metrics.KV_CACHE] = summarize_cache_counters(cache_records)

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    output_token_count: str = "provider",
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        output_token_count=output_token_count,
        prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
    )

    if results_dir:
        filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_directory = Path(results_dir)
        if not results_directory.exists():
            results_directory.mkdir(parents=True)
        elif not results_directory.is_dir():
            raise ValueError(f"{results_directory} is not a directory")

        summary_path = results_directory / f"{summary_filename}.json"
        with summary_path.open("w", encoding="utf-8") as summary_stream:
            json.dump(results.to_dict(), summary_stream, indent=4, default=str)

        responses_path = results_directory / f"{individual_responses_filename}.json"
        with responses_path.open("w", encoding="utf-8") as responses_stream:
            json.dump(individual_responses, responses_stream, indent=4)


parser = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

parser.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
parser.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
parser.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
parser.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
parser.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
parser.add_argument(
    "--output-token-count",
    choices=("provider", "local"),
    default="provider",
    help="Use server usage (default), or tokenize outputs locally after the timed test.",
)
parser.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
parser.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
parser.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
parser.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
parser.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
parser.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
parser.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars}, include_dashboard=False)
    arguments = parser.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if arguments.metadata:
        for item in arguments.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        llm_api=arguments.llm_api,
        model=arguments.model,
        test_timeout_s=arguments.timeout,
        max_num_completed_requests=arguments.max_num_completed_requests,
        mean_input_tokens=arguments.mean_input_tokens,
        stddev_input_tokens=arguments.stddev_input_tokens,
        mean_output_tokens=arguments.mean_output_tokens,
        stddev_output_tokens=arguments.stddev_output_tokens,
        num_concurrent_requests=arguments.num_concurrent_requests,
        additional_sampling_params=arguments.additional_sampling_params,
        results_dir=arguments.results_dir,
        user_metadata=user_metadata,
        output_token_count=arguments.output_token_count,
    )

import threading
import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import time
import random
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    get_tokenizer,
    randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm

def normalize_request_metrics(
    request_metrics: Dict[str, Any], generated_text: str, get_token_length
) -> Dict[str, Any]:
    """Reconcile provider counters with tokenizer-derived output metrics safely."""

    num_output_tokens = get_token_length(generated_text)
    inter_token_latency_sum = request_metrics.get(common_metrics.INTER_TOKEN_LAT, 0)
    request_metrics[common_metrics.INTER_TOKEN_LAT] = (
        inter_token_latency_sum / num_output_tokens if num_output_tokens else 0
    )
    request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
    request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
        request_metrics.get(common_metrics.NUM_INPUT_TOKENS, 0) + num_output_tokens
    )
    end_to_end_latency = request_metrics.get(common_metrics.E2E_LAT, 0)
    request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
        num_output_tokens / end_to_end_latency if end_to_end_latency > 0 else 0
    )
    return request_metrics


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    repeat_count: int,
    min_input_tokens: int,
    max_input_tokens: int,
    get_token_length,
    seed: int = 11111,
) -> List[Tuple[str, int]]:
    """Sample and repeat first-turn ShareGPT prompts like vLLM's APC benchmark."""

    path = Path(dataset_path).expanduser()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read ShareGPT dataset {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ShareGPT dataset is not valid JSON: {path}") from exc
    if not isinstance(document, list):
        raise ValueError("ShareGPT dataset must be a JSON array")

    candidates = []
    for record in document:
        if not isinstance(record, dict):
            continue
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        first_turn = conversations[0]
        if not isinstance(first_turn, dict):
            continue
        prompt = first_turn.get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        candidates.append(prompt)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    unique_request_count = math.ceil(num_requests / repeat_count)
    sampled = []
    for prompt in candidates:
        prompt_len = get_token_length(prompt)
        if min_input_tokens <= prompt_len <= max_input_tokens:
            sampled.append((prompt, prompt_len))
            if len(sampled) == unique_request_count:
                break
    if len(sampled) < unique_request_count:
        raise ValueError(
            "ShareGPT dataset has only "
            f"{len(sampled)} matching prompts; {unique_request_count} required in "
            f"token range {min_input_tokens}:{max_input_tokens}"
        )

    repeated = (sampled * repeat_count)[:num_requests]
    rng.shuffle(repeated)
    return repeated


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    shared_prefix_tokens: int = 0,
    dataset_path: Optional[str] = None,
    dataset_format: str = "sharegpt",
    dataset_repeat_count: int = 1,
    dataset_seed: int = 11111,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        shared_prefix_tokens: Number of leading prompt tokens reused by every request.
        dataset_path: Optional path to a ShareGPT JSON dataset.
        dataset_format: Dataset schema. Currently only ``sharegpt`` is supported.
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
    random.seed(11111)

    tokenizer = get_tokenizer()
    get_token_length = lambda text: len(
        tokenizer.encode(text, add_special_tokens=False)
    )
    
    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    if shared_prefix_tokens >= mean_input_tokens:
        raise ValueError("shared_prefix_tokens must be less than mean_input_tokens")
    if dataset_path and shared_prefix_tokens:
        raise ValueError("dataset_path and shared_prefix_tokens cannot be used together")
    if dataset_path and dataset_format != "sharegpt":
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    if dataset_repeat_count < 1:
        raise ValueError("dataset_repeat_count must be at least 1")

    # Make up prompts outside of the send loop for faster benchmarking. A shared
    # prefix creates a controlled provider KV-cache workload while each suffix
    # remains unique.
    num_output_tokens_list = [
        sample_random_positive_int(mean_output_tokens, stddev_output_tokens)
        for _ in range(max_num_completed_requests)
    ]
    if dataset_path:
        min_input_tokens = max(1, mean_input_tokens - stddev_input_tokens)
        max_input_tokens = mean_input_tokens + stddev_input_tokens
        prompts = sample_sharegpt_requests(
            dataset_path=dataset_path,
            num_requests=max_num_completed_requests,
            repeat_count=dataset_repeat_count,
            min_input_tokens=min_input_tokens,
            max_input_tokens=max_input_tokens,
            get_token_length=get_token_length,
            seed=dataset_seed,
        )
    else:
        prompts = []
        shared_prefix = ""
        if shared_prefix_tokens:
            shared_prefix, _ = randomly_sample_sonnet_lines_prompt(
                prompt_tokens_mean=shared_prefix_tokens,
                prompt_tokens_stddev=0,
                expect_output_tokens=mean_output_tokens,
                tokenizer=tokenizer,
            )
        for i in range(max_num_completed_requests):
            suffix = randomly_sample_sonnet_lines_prompt(
                prompt_tokens_mean=mean_input_tokens - shared_prefix_tokens,
                prompt_tokens_stddev=stddev_input_tokens,
                expect_output_tokens=num_output_tokens_list[i],
                tokenizer=tokenizer
            )[0]
            prompt = shared_prefix + suffix
            prompts.append((prompt, get_token_length(prompt)))
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests)

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % max_num_completed_requests

        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
        ):

            default_sampling_params = {"max_tokens": num_output_tokens_list[request_index] }
            default_sampling_params.update(additional_sampling_params)
            remaining_seconds = max(
                0.1, test_timeout_s - (time.monotonic() - start_time) - 1
            )
            request_config = RequestConfig(
                model=model,
                prompt=prompts[request_index],
                sampling_params=default_sampling_params,
                llm_api=llm_api,
                timeout_seconds=remaining_seconds,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                request_metrics = normalize_request_metrics(
                    request_metrics, gen_text, get_token_length
                )
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        all_metrics.append(request_metrics)
                        completed_requests.extend(all_metrics)
                        pbar.update(len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = (request_index + num_concurrent_requests) % max_num_completed_requests

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()
    end_time = time.monotonic()
    timed_out = end_time - start_time >= test_timeout_s
    if timed_out:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        request_metrics = normalize_request_metrics(
            request_metrics, gen_text, get_token_length
        )
        with completed_requests_lock:
            if num_completed_requests < max_num_completed_requests:
                completed_requests.append(request_metrics)
                num_completed_requests += 1

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "shared_prefix_tokens": shared_prefix_tokens,
        "dataset_path": dataset_path,
        "dataset_format": dataset_format if dataset_path else None,
        "dataset_repeat_count": dataset_repeat_count if dataset_path else None,
        "dataset_seed": dataset_seed if dataset_path else None,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
        "timed_out": timed_out,
    }

    metadata["results"] = ret
        
    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}
    metric_keys = [
        common_metrics.INTER_TOKEN_LAT,
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
        ret[common_metrics.NUM_COMPLETED_REQUESTS] = 0
        ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = 0
        ret[common_metrics.KV_CACHE] = {
            "measured_requests": 0,
            "hit_tokens": 0,
            "miss_tokens": 0,
            "hit_ratio": None,
        }
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
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

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

    duration = end_time - start_time
    overall_output_throughput = (
        df_without_errored_req[common_metrics.NUM_OUTPUT_TOKENS].sum() / duration
        if duration > 0
        else 0
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

    cache_metrics = (
        df_without_errored_req.dropna(
            subset=[common_metrics.KV_CACHE_HIT_TOKENS],
        )
        if common_metrics.KV_CACHE_HIT_TOKENS in df_without_errored_req
        else pd.DataFrame()
    )
    cache_hit_tokens = (
        cache_metrics[common_metrics.KV_CACHE_HIT_TOKENS].sum()
        if not cache_metrics.empty
        else 0
    )
    cache_miss_tokens = (
        cache_metrics[common_metrics.KV_CACHE_MISS_TOKENS].sum()
        if not cache_metrics.empty
        and common_metrics.KV_CACHE_MISS_TOKENS in cache_metrics
        else 0
    )
    cache_tokens = cache_hit_tokens + cache_miss_tokens
    ret[common_metrics.KV_CACHE] = {
        "measured_requests": len(cache_metrics),
        "hit_tokens": cache_hit_tokens,
        "miss_tokens": cache_miss_tokens,
        "hit_ratio": cache_hit_tokens / cache_tokens if cache_tokens else None,
    }
    
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
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
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
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        mean_input_tokens=args.mean_input_tokens,
        stddev_input_tokens=args.stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
    )

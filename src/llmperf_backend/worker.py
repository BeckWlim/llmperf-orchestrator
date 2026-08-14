"""Calculation worker that commits benchmark results directly to PostgreSQL."""

import argparse
import asyncio
from functools import partial
import logging
import os
from typing import Any, Dict, Sequence, Tuple

from llmperf import common_metrics
from llmperf.logging import configure_logging
from llmperf_backend.models import DatabaseConfig
from llmperf_backend.datasets import WORKER_DATASET_PATH
from llmperf_backend.persistence import (
    Database,
    FAILED,
    RunnerRepository,
    RUNNING,
    SUCCEEDED,
)
from llmperf_backend.scheduler import WORKER_DATABASE_URL


LOGGER = logging.getLogger(__name__)


def _calculate(
    benchmark: Dict[str, Any],
) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]:
    import ray

    from llmperf.token_benchmark_ray import get_token_throughput_latencies

    runtime_environment = dict(os.environ)
    runtime_environment.pop(WORKER_DATABASE_URL, None)
    ray.init(runtime_env={"env_vars": runtime_environment})
    try:
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
        return get_token_throughput_latencies(
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
            protocol_request=benchmark.get("protocol_request"),
        )
    finally:
        ray.shutdown()


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
        first_error = {
            "code": code,
            "message": error_message,
        }
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


async def execute_runner(runner_id: str, database_url: str) -> None:
    LOGGER.info("Worker executing Runner %s", runner_id)
    database = Database(DatabaseConfig(url=database_url, auto_create_schema=False))
    repository = RunnerRepository(database)
    try:
        runner = await repository.get_runner(runner_id)
        if runner is None:
            raise RuntimeError(f"Runner does not exist: {runner_id}")
        if runner["status"] != RUNNING:
            raise RuntimeError(f"Runner {runner_id} is not running: {runner['status']}")
        loop = asyncio.get_running_loop()
        summary, requests = await loop.run_in_executor(
            None, partial(_calculate, runner["benchmark"])
        )
        summary["runner_metadata"] = runner["metadata"]
        terminal_status, message = summarize_outcome(summary, requests)
        await repository.complete_runner(
            runner_id,
            summary,
            requests,
            0,
            "",
            "",
            terminal_status=terminal_status,
            error_message=message if terminal_status == FAILED else None,
        )
        LOGGER.info("Worker completed Runner %s: %s", runner_id, terminal_status)
    finally:
        await database.dispose()


def main() -> None:
    configure_logging(
        os.environ.get("LLMPERF_LOG_LEVEL", "info"),
        color=os.environ.get("LLMPERF_LOG_COLOR", "auto").lower(),
    )
    parser = argparse.ArgumentParser(description="Execute one persisted LLMPerf Runner")
    parser.add_argument("--runner-id", required=True)
    arguments = parser.parse_args()
    database_url = os.environ.get(WORKER_DATABASE_URL)
    if not database_url:
        raise RuntimeError(f"{WORKER_DATABASE_URL} is not configured")
    try:
        asyncio.run(execute_runner(arguments.runner_id, database_url))
    except Exception:
        LOGGER.exception("Worker failed while executing Runner %s", arguments.runner_id)
        raise


if __name__ == "__main__":
    main()

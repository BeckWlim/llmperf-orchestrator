"""Opt-in real Ray/HTTP verification; never contacts a configured Provider."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import tempfile
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import ray

from llmperf.metrics import constants as common_metrics
from llmperf.benchmark import throughput as benchmark_module
from llmperf.clients.factory import RAY_ACTOR_CONCURRENCY_ENV, RAY_ACTOR_CPUS_ENV
from llmperf.datasets.prompts import PromptDatasetSource
from llmperf.clients.models import RequestConfig, WorkerInput
from llmperf.datasets.preparation import create_preparation_actor
from llmperf_backend.runtime.request_pool import (
    create_request_actor,
    create_request_router,
    SharedRequestExecutor,
    RequestActor,
    RequestActorHandle,
)
from llmperf_backend.runtime.worker import AtomicWorker
import llmperf_backend.runtime.worker as worker_module


pytestmark = pytest.mark.skipif(
    os.environ.get("LLMPERF_TEST_RAY") != "1",
    reason="Set LLMPERF_TEST_RAY=1 to start an isolated local Ray runtime",
)


@pytest.mark.parametrize(
    "mode",
    ["ordinary", "grouped", "shared", "atomic", "prepared", "sharegpt", "recipe"],
)
def test_concurrent_http_streams(monkeypatch, mode, tmp_path):
    grouped = mode == "grouped"
    prepared = mode in {"prepared", "sharegpt"}
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Question {index} about topic {index % 13}.",
                        }
                    ]
                }
                for index in range(300)
            ]
        )
    )
    request_count = 64
    barrier = threading.Barrier(request_count, timeout=60)
    seen_prompts: set[str] = set()
    request_lock = threading.Lock()
    credentials: set[str] = set()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            document = json.loads(self.rfile.read(content_length))
            with request_lock:
                seen_prompts.add(document["messages"][-1]["content"])
                credentials.add(str(self.headers["Authorization"]))
            # No request may complete before all 64 have reached the server.
            barrier.wait()
            stream = (
                b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n' b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(stream)))
            self.end_headers()
            self.wfile.write(stream)

    class Server(ThreadingHTTPServer):
        request_queue_size = 128

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    server = Server(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    provider_url = f"http://127.0.0.1:{server.server_port}/v1"
    runtime_environment = {
        "OPENAI_API_BASE": provider_url,
        "OPENAI_API_KEY": "local-test-only",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "RAY_grpc_enable_http_proxy": "0",
        RAY_ACTOR_CPUS_ENV: "1",
        RAY_ACTOR_CONCURRENCY_ENV: "8",
        "HF_DATASETS_CACHE": str(tmp_path / "indexes"),
    }
    for variable_name, variable_value in runtime_environment.items():
        monkeypatch.setenv(variable_name, variable_value)
    monkeypatch.setattr(benchmark_module, "get_tokenizer", Tokenizer)
    monkeypatch.setattr(
        benchmark_module,
        "prepare_prompt_requests",
        lambda **kwargs: (
            [
                (f"request-{kwargs['seed'] if grouped else index}", 1)
                for index in range(kwargs["num_requests"])
            ],
            [{"source": "test"} for _ in range(kwargs["num_requests"])],
        ),
    )
    ray_directory = tempfile.TemporaryDirectory(prefix="lpr-")
    try:
        ray.init(
            address="local",
            num_cpus=10 if prepared else 8,
            include_dashboard=False,
            object_store_memory=128 * 1024 * 1024,
            _temp_dir=ray_directory.name,
            runtime_env={"env_vars": runtime_environment},
        )
        if mode in {"shared", "atomic", "recipe"}:

            class RecipeActor(RequestActor):
                def __init__(self):
                    super().__init__()
                    import llmperf_backend.runtime.worker_input as inputs

                    inputs._prepared_tokenizer = lambda *args: Tokenizer()

            actors: list[RequestActorHandle] = []
            for _ in range(8):
                if mode == "recipe":
                    handle = (
                        ray.remote(RecipeActor)
                        .options(
                            num_cpus=1,
                            max_concurrency=8,
                            max_restarts=0,
                            max_task_retries=0,
                        )
                        .remote()
                    )
                    assert isinstance(handle, RequestActorHandle)
                    actors.append(handle)
                else:
                    actors.append(create_request_actor(1, 8))
            router = create_request_router(actors, 8, 128)

            def invoke(index):
                executor = SharedRequestExecutor(
                    router,
                    f"runner-{index % 2}",
                    {
                        "OPENAI_API_BASE": provider_url,
                        "OPENAI_API_KEY": f"key-{index % 2}",
                        "LLMPERF_TOKENIZER_PATH": "mock",
                    },
                )
                return executor(
                    RequestConfig(
                        model="test",
                        prompt=("", 1) if mode == "recipe" else (f"request-{index}", 1),
                        preparation=(
                            WorkerInput(text=f"request-{index}")
                            if mode == "recipe"
                            else None
                        ),
                        llm_api="openai",
                        timeout_seconds=120,
                    )
                )[0]

            with ThreadPoolExecutor(max_workers=64) as executor:
                if mode == "atomic":
                    monkeypatch.setattr(
                        worker_module, "_prepared_tokenizer", lambda *args: Tokenizer()
                    )

                    async def execute_atomic():
                        workers = [
                            AtomicWorker(ray, None, f"runner-{index}", 1, 1, executor)
                            for index in range(64)
                        ]
                        for index, worker in enumerate(workers):
                            worker.start(
                                {
                                    "provider": "test",
                                    "model": "test",
                                    "adapter": "openai",
                                    "timeout": 120,
                                    "repeat": 1,
                                    "concurrent_workers": 1,
                                    "mean_input_tokens": 1,
                                    "stddev_input_tokens": 0,
                                    "mean_output_tokens": 1,
                                    "stddev_output_tokens": 0,
                                    "additional_sampling_params": {},
                                    "task_context": {
                                        "payload_seed": index,
                                        "prompt_text": f"request-{index}",
                                    },
                                },
                                {
                                    "LLMPERF_TOKENIZER_PATH": "mock",
                                    "OPENAI_API_BASE": provider_url,
                                    "OPENAI_API_KEY": f"key-{index % 2}",
                                },
                                {"runner_id": worker.runner_id},
                                1024,
                                router,
                            )
                        completed = await asyncio.gather(
                            *(worker.result_async() for worker in workers)
                        )
                        assert all(item["ok"] for item in completed)
                        return [item["requests"][0] for item in completed]

                    shared_results = asyncio.run(execute_atomic())
                else:
                    shared_results = list(executor.map(invoke, range(64)))
            assert all(
                item[common_metrics.ERROR_CODE] is None for item in shared_results
            )
            assert len(seen_prompts) == 64
            assert credentials == {"Bearer key-0", "Bearer key-1"}
            assert ray.get(router.status.remote())["request_capacity"] == 64
            return
        _, requests = benchmark_module.get_token_throughput_latencies(
            model="local-test",
            prompt_dataset_source=(
                PromptDatasetSource.external("sharegpt-user", dataset_path)
                if mode == "sharegpt"
                else PromptDatasetSource.builtin_sonnet()
            ),
            dataset_prompt_mode="concatenate" if mode == "sharegpt" else "sample",
            shared_prefix_tokens=128 if mode == "sharegpt" else 0,
            mean_input_tokens=256 if prepared else 1,
            stddev_input_tokens=0,
            mean_output_tokens=1,
            stddev_output_tokens=0,
            num_concurrent_requests=request_count,
            max_num_completed_requests=request_count,
            test_timeout_s=120,
            preparation_actors=(
                [create_preparation_actor() for _ in range(2)] if prepared else []
            ),
            task_contexts=(
                [
                    {
                        "definition_id": "definition",
                        "instance_id": "instance",
                        "node_id": f"request-{index}",
                        "payload_id": f"payload-{index}",
                        "payload_seed": index,
                    }
                    for index in range(request_count)
                ]
                if grouped
                else None
            ),
        )
        assert len(seen_prompts) == request_count
        assert len(requests) == request_count
        assert all(item[common_metrics.ERROR_CODE] is None for item in requests)
        if grouped:
            assert {
                item[common_metrics.REQUEST_METADATA]["node_id"] for item in requests
            } == {f"request-{index}" for index in range(request_count)}
    finally:
        ray.shutdown()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        ray_directory.cleanup()

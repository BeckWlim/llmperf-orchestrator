"""Pure benchmark, request normalization, and cache-probe algorithm tests."""

import json
from collections import Counter

import pytest
import time
import threading
import sys
from types import SimpleNamespace

from llmperf.benchmark import throughput as benchmark_module
from llmperf.metrics import constants as common_metrics
from llmperf.clients.models import RequestConfig
from llmperf.datasets.prompts import (
    PromptDatasetSource,
    concatenate_prompt_requests,
    load_prompt_dataset,
    sample_prompt_requests,
)
from llmperf.clients.openai import (
    OpenAIStreamError,
    SSE_ITER_CHUNK_SIZE,
    StreamInactivityTimeout,
    cache_metrics_from_usage,
    decode_sse_line,
)
from llmperf.metrics.cache import analyze_cache_probe, summarize_cache_counters
from llmperf.datasets.probes import DependentPlanQueue, build_cache_probe_plan
from llmperf.metrics.usage import normalize_usage
from llmperf.benchmark.throughput import metrics_summary, normalize_request_metrics

NORMALIZATION_CASES = [
    pytest.param(
        {
            "metrics": {
                common_metrics.E2E_LAT: 2.0,
                common_metrics.NUM_INPUT_TOKENS: 10,
                common_metrics.NUM_OUTPUT_TOKENS: 0,
                common_metrics.ERROR_CODE: None,
            },
            "text": "generated",
            "tokens": 4,
            "expected": {
                common_metrics.NUM_OUTPUT_TOKENS: 4,
                common_metrics.NUM_TOTAL_TOKENS: 14,
                common_metrics.REQ_OUTPUT_THROUGHPUT: 2.0,
            },
        },
        id="generated-output",
    ),
    pytest.param(
        {
            "metrics": {
                common_metrics.E2E_LAT: 0,
                common_metrics.NUM_INPUT_TOKENS: 10,
                common_metrics.NUM_OUTPUT_TOKENS: 0,
                common_metrics.ERROR_CODE: -1,
            },
            "text": "",
            "tokens": 0,
            "expected": {
                common_metrics.REQ_OUTPUT_THROUGHPUT: 0,
            },
        },
        id="zero-output",
    ),
]


@pytest.fixture(autouse=True)
def isolated_dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets-cache"))


def _event(document):
    return b"data:" + json.dumps(document).encode("utf-8")


def test_sharegpt_sampling(tmp_path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": f"prompt number {index}"},
                        {"from": "gpt", "value": "answer"},
                    ]
                }
                for index in range(5)
            ]
        ),
        encoding="utf-8",
    )

    requests, _ = sample_prompt_requests(
        load_prompt_dataset(PromptDatasetSource.external("sharegpt", dataset_path)),
        num_requests=6,
        repeat_count=3,
        min_input_tokens=3,
        max_input_tokens=3,
        get_token_length=lambda text: len(text.split()),
        seed=7,
    )

    assert len(requests) == 6
    assert set(Counter(requests).values()) == {3}


def test_sharegpt_capacity(tmp_path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "short prompt"},
                        {"from": "gpt", "value": "answer"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matching prompts"):
        sample_prompt_requests(
            load_prompt_dataset(PromptDatasetSource.external("sharegpt", dataset_path)),
            num_requests=2,
            repeat_count=1,
            min_input_tokens=100,
            max_input_tokens=200,
            get_token_length=lambda text: len(text.split()),
        )


def test_sharegpt_concatenate(tmp_path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": character * 10},
                        {"from": "gpt", "value": "answer"},
                    ]
                }
                for character in "ABCDE"
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_prompt_dataset(
        PromptDatasetSource.external("sharegpt", dataset_path)
    )
    first = concatenate_prompt_requests(
        dataset,
        num_requests=2,
        mean_input_tokens=24,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        get_token_length=len,
        seed=7,
    )
    second = concatenate_prompt_requests(
        dataset,
        num_requests=2,
        mean_input_tokens=24,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        get_token_length=len,
        seed=7,
    )

    assert first == second
    prompts, evidence = first
    assert [length for _, length in prompts] == [24, 24]
    selected = [index for item in evidence for index in item["record_indices"]]
    assert len(selected[:5]) == len(set(selected[:5]))
    assert all(item["manifest_hash"].startswith("sha256:") for item in evidence)
    assert all("AAAA" not in json.dumps(item) for item in evidence)


def test_stream():
    assert decode_sse_line(_event({"choices": []})) == {"kind": "metadata"}
    assert decode_sse_line(b"event: message") == {"kind": "ignore"}
    assert decode_sse_line(b": keepalive") == {"kind": "ignore"}
    assert decode_sse_line(b"data: [DONE]") == {"kind": "done"}

    event = decode_sse_line(
        _event(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "think ",
                            "content": "answer",
                        }
                    }
                ]
            }
        )
    )
    assert event == {"kind": "text", "text": "think answer"}


def test_error():
    with pytest.raises(OpenAIStreamError) as error:
        decode_sse_line(
            _event({"error": {"code": 403, "message": "model not enabled"}})
        )

    assert error.value.code == 403
    assert str(error.value) == "model not enabled"


def test_timeout():
    config = RequestConfig(
        model="deepseek-v4-pro",
        prompt=("prompt", 1),
        timeout_seconds=29.0,
    )

    assert config.timeout_seconds == 29.0


def test_aliyun_usage():
    usage = {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 75},
    }

    assert decode_sse_line(_event({"choices": [], "usage": usage})) == {
        "kind": "metadata",
        "usage": usage,
    }
    assert cache_metrics_from_usage(usage) == {
        common_metrics.KV_CACHE_HIT_TOKENS: 75,
        common_metrics.KV_CACHE_MISS_TOKENS: 25,
        common_metrics.KV_CACHE_HIT_RATE: 0.75,
    }


def test_text_event_usage():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 75},
    }
    event = decode_sse_line(
        _event({"choices": [{"delta": {"content": "answer"}}], "usage": usage})
    )

    assert event == {"kind": "text", "text": "answer", "usage": usage}
    normalized = normalize_usage(usage)
    assert normalized.complete is True
    assert normalized.hit_tokens == 75
    assert normalized.miss_tokens == 25
    assert normalized.provider_output_tokens == 3


def test_deepseek_usage():
    metrics = cache_metrics_from_usage(
        {"prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20}
    )

    assert metrics[common_metrics.KV_CACHE_HIT_RATE] == 0.8


def test_summary():
    request_metrics = [
        {
            common_metrics.TTFT: 0.2,
            common_metrics.E2E_LAT: 1.0,
            common_metrics.REQ_OUTPUT_THROUGHPUT: 2.0,
            common_metrics.NUM_INPUT_TOKENS: 100,
            common_metrics.NUM_OUTPUT_TOKENS: 2,
            common_metrics.ERROR_CODE: None,
            common_metrics.KV_CACHE_HIT_TOKENS: hit,
            common_metrics.KV_CACHE_MISS_TOKENS: miss,
        }
        for hit, miss in ((0, 100), (75, 25))
    ]

    summary = metrics_summary(request_metrics, 1.0, 3.0)

    cache = summary[common_metrics.KV_CACHE]
    assert cache["measured_requests"] == 2
    assert cache["complete_hit_tokens"] == 75
    assert cache["complete_miss_tokens"] == 125
    assert cache["weighted_token_hit_ratio"] == 0.375
    assert cache["counter_coverage"] == 1.0


def test_incomplete_counters():
    cache = summarize_cache_counters([{common_metrics.KV_CACHE_HIT_TOKENS: 100}])

    assert cache["complete_hit_tokens"] == 0
    assert cache["complete_miss_tokens"] == 0
    assert cache["weighted_token_hit_ratio"] is None
    assert cache["complete_counter_requests"] == 0


def test_invalid_counters():
    normalized = normalize_usage(
        {
            "prompt_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 12},
        }
    )
    request = normalized.to_metrics()
    cache = summarize_cache_counters([request])

    assert normalized.valid is False
    assert (
        normalized.validation_error == "cache hit tokens exceed provider input tokens"
    )
    assert cache["invalid_counter_requests"] == 1
    assert cache["weighted_token_hit_ratio"] is None


@pytest.mark.parametrize("case", NORMALIZATION_CASES)
def test_normalization(case):
    normalized = normalize_request_metrics(
        case["metrics"], case["text"], lambda text: case["tokens"]
    )

    for name, expected in case["expected"].items():
        assert normalized[name] == expected


@pytest.mark.parametrize("provider_output", [None, 0, 7])
def test_provider_token_counts(provider_output):
    metrics = {
        common_metrics.NUM_INPUT_TOKENS: 10,
        common_metrics.NUM_OUTPUT_TOKENS: 3,  # SSE events are not token counts.
        common_metrics.PROVIDER_INPUT_TOKENS: 12,
        common_metrics.PROVIDER_OUTPUT_TOKENS: provider_output,
        common_metrics.TTFT: 0.5,
        common_metrics.E2E_LAT: 2.0,
    }
    normalized = normalize_request_metrics(metrics, "some generated text")
    assert normalized[common_metrics.NUM_OUTPUT_TOKENS] == provider_output
    assert normalized[common_metrics.LOCAL_OUTPUT_TOKENS] is None
    assert normalized[common_metrics.LOCAL_INPUT_TOKENS] == 10
    assert normalized[common_metrics.NUM_INPUT_TOKENS] == 12
    assert normalized[common_metrics.NUM_TOTAL_TOKENS] == (
        None if provider_output is None else 12 + provider_output
    )
    assert normalized[common_metrics.REQ_OUTPUT_THROUGHPUT] == (
        None if provider_output is None else provider_output / 2
    )
    assert normalized[common_metrics.TOKEN_COUNT_SOURCES]["output"] == (
        "unavailable" if provider_output is None else "provider"
    )
    assert normalized[common_metrics.TPOT] == (0.25 if provider_output == 7 else None)


@pytest.mark.parametrize("adapter", ["vertexai", "sagemaker"])
def test_actor_without_tokenizer(monkeypatch, adapter):
    def forbidden_tokenizer():
        raise AssertionError("Request actors must not load a tokenizer")

    monkeypatch.setattr("llmperf.datasets.tokenizer.get_tokenizer", forbidden_tokenizer)
    if adapter == "vertexai":
        from llmperf.clients.vertexai import VertexAIClient

        for name in (
            "GCLOUD_PROJECT_ID",
            "GCLOUD_REGION",
            "VERTEXAI_ENDPOINT_ID",
            "GCLOUD_ACCESS_TOKEN",
        ):
            monkeypatch.setenv(name, "test")
        monkeypatch.setattr(
            "llmperf.clients.vertexai.requests.post",
            lambda *args, **kwargs: SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"predictions": ["Input: prompt\nOutput:\nresponse"]},
            ),
        )
        metrics, text, _ = VertexAIClient().llm_request(
            RequestConfig(model="test", prompt=("prompt", 1))
        )
    else:
        from llmperf.clients.sagemaker import SageMakerClient

        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"):
            monkeypatch.setenv(name, "test")
        response_bytes = b'[{"generation":{"content":"response"}}]\n'
        runtime = SimpleNamespace(
            invoke_endpoint_with_response_stream=lambda **kwargs: {
                "Body": [{"PayloadPart": {"Bytes": response_bytes}}],
            }
        )
        monkeypatch.setattr(
            "llmperf.clients.sagemaker.Session",
            lambda **credentials: SimpleNamespace(
                client=lambda *args, **kwargs: runtime
            ),
        )
        metrics, text, _ = SageMakerClient().llm_request(
            RequestConfig(model="test", prompt=("prompt", 1))
        )
    assert text == "response"
    assert metrics[common_metrics.ERROR_CODE] is None
    assert metrics[common_metrics.NUM_OUTPUT_TOKENS] is None


def test_litellm_usage_chunk(monkeypatch):
    from llmperf.clients.litellm import LiteLLMClient

    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta={"content": "response"})],
            model_dump=lambda: {},
        ),
        SimpleNamespace(
            choices=[],
            model_dump=lambda: {"usage": {"prompt_tokens": 2, "completion_tokens": 7}},
        ),
    ]
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion=lambda **kwargs: iter(chunks),
            validate_environment=lambda model: {"missing_keys": []},
        ),
    )
    metrics, text, _ = LiteLLMClient().llm_request(
        RequestConfig(model="test", prompt=("prompt", 1), llm_api="litellm")
    )
    normalized = normalize_request_metrics(metrics, text)
    assert normalized[common_metrics.ERROR_CODE] is None
    assert normalized[common_metrics.NUM_OUTPUT_TOKENS] == 7
    assert normalized[common_metrics.NUM_INPUT_TOKENS] == 2


@pytest.mark.parametrize("counts", [[None, None], [4, None], [4, 0]])
def test_token_coverage(counts):
    requests = [
        normalize_request_metrics(
            {
                common_metrics.NUM_INPUT_TOKENS: 10,
                common_metrics.PROVIDER_OUTPUT_TOKENS: count,
                common_metrics.TTFT: 0.1,
                common_metrics.E2E_LAT: 1.0,
                common_metrics.ERROR_CODE: None,
            },
            "output",
        )
        for count in counts
    ]
    summary = metrics_summary(requests, 0.0, 2.0)
    known_counts = [count for count in counts if count is not None]
    assert summary[common_metrics.OUTPUT_TOKEN_COVERAGE] == {
        "counted_requests": len(known_counts),
        "successful_requests": 2,
        "ratio": len(known_counts) / 2,
    }
    assert summary[common_metrics.OUTPUT_THROUGHPUT] == (
        sum(known_counts) / 2 if len(known_counts) == 2 else None
    )
    if not known_counts:
        assert summary[common_metrics.NUM_OUTPUT_TOKENS]["mean"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("count_mode", ["provider", "local"])
@pytest.mark.parametrize("cache_probe", [False, True])
def test_tokenization_phase(monkeypatch, count_mode, cache_probe):
    completed = []
    encoded_outputs = []
    clock_seconds = [0.0]
    expected_requests = 4 if cache_probe else 2

    class PhaseTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "response":
                assert len(completed) == expected_requests
                encoded_outputs.append(text)
                clock_seconds[0] += 100.0
            else:
                assert not completed
            return [ord(character) for character in text]

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens)

    def prepare_prompts(**options):
        count = options["get_token_length"]("prompt")
        return [("prompt", count)] * 2, [{"source": "test"}] * 2

    class PhaseLauncher:
        def __init__(self, clients, request_executor=None):
            self.config = None

        def launch_requests(self, config):
            self.config = config

        def get_next_ready(self, block=False):
            completed.append(self.config)
            clock_seconds[0] += 1.0
            return [
                (
                    {
                        common_metrics.ERROR_CODE: None,
                        common_metrics.E2E_LAT: 1.0,
                        common_metrics.TTFT: 0.1,
                        common_metrics.NUM_INPUT_TOKENS: 6,
                        common_metrics.NUM_OUTPUT_TOKENS: 1,
                        common_metrics.PROVIDER_OUTPUT_TOKENS: 2,
                    },
                    "response",
                    self.config,
                )
            ]

    monkeypatch.setattr(benchmark_module, "get_tokenizer", PhaseTokenizer)
    monkeypatch.setattr(benchmark_module, "prepare_prompt_requests", prepare_prompts)
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **options: [0])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", PhaseLauncher)
    monkeypatch.setattr(benchmark_module.time, "monotonic", lambda: clock_seconds[0])
    summary, requests = benchmark_module.get_token_throughput_latencies(
        model="test",
        prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
        mean_input_tokens=6,
        stddev_input_tokens=0,
        mean_output_tokens=8,
        stddev_output_tokens=0,
        max_num_completed_requests=2,
        num_concurrent_requests=1,
        output_token_count=count_mode,
        cache_probe=(
            {"mode": "exact_repeat", "trials": 2, "bootstrap_samples": 100}
            if cache_probe
            else None
        ),
    )
    assert len(encoded_outputs) == (expected_requests if count_mode == "local" else 0)
    assert summary["output_token_count"] == count_mode
    assert summary["results"][common_metrics.COMPLETED_REQUESTS_PER_MIN] == 60.0
    assert summary["timed_out"] is False
    assert all(
        request[common_metrics.NUM_OUTPUT_TOKENS] == (8 if count_mode == "local" else 2)
        for request in requests
    )


def test_empty_summary():
    summary = metrics_summary([], 1.0, 2.0)

    assert summary[common_metrics.NUM_REQ_STARTED] == 0
    assert summary[common_metrics.NUM_COMPLETED_REQUESTS] == 0
    assert summary[common_metrics.NUM_ERRORS] == 0
    assert summary[common_metrics.ERROR_CODE_FREQ] == {}
    assert summary[common_metrics.OUTPUT_THROUGHPUT] == 0
    assert summary[common_metrics.KV_CACHE]["measured_requests"] == 0


def test_thread_error_propagation(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    class FailingLauncher:
        def __init__(self, clients, request_executor=None):
            pass

        def launch_requests(self, request_config):
            pass

        def get_next_ready(self):
            raise ValueError("Ray Actor did not receive OPENAI_API_BASE")

    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeTokenizer)
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FailingLauncher)

    with pytest.raises(ValueError, match="OPENAI_API_BASE"):
        benchmark_module.get_token_throughput_latencies(
            model="glm-5.2",
            prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
            mean_input_tokens=64,
            stddev_input_tokens=0,
            mean_output_tokens=1,
            stddev_output_tokens=0,
            num_concurrent_requests=1,
            max_num_completed_requests=1,
            test_timeout_s=10,
            llm_api="openai",
        )


@pytest.mark.parametrize("concurrency,per_actor", [(4, 1), (64, 8), (9, 4)])
def test_shared_actor_requests(monkeypatch, concurrency, per_actor):
    from llmperf.clients.factory import RAY_ACTOR_CONCURRENCY_ENV

    actor_count = (concurrency + per_actor - 1) // per_actor
    active = [0] * actor_count
    peak = [0] * actor_count
    launched = []
    created = []
    lock = threading.Lock()
    barrier = threading.Barrier(concurrency, timeout=10)

    def create_clients(llm_api, num_clients):
        created.append(num_clients)
        return list(range(num_clients))

    class ConcurrentLauncher:
        def __init__(self, clients, request_executor=None):
            self.actor_index = clients[0]
            self.config = None
            self.first_request = True

        def launch_requests(self, request_config):
            self.config = request_config
            with lock:
                launched.append(request_config.prompt[0])
                active[self.actor_index] += 1
                peak[self.actor_index] = max(
                    peak[self.actor_index], active[self.actor_index]
                )

        def get_next_ready(self):
            if self.first_request:
                barrier.wait()
                self.first_request = False
            with lock:
                active[self.actor_index] -= 1
            return [
                (
                    {
                        common_metrics.E2E_LAT: 0.1,
                        common_metrics.TTFT: 0.05,
                        common_metrics.NUM_INPUT_TOKENS: 1,
                        common_metrics.ERROR_CODE: None,
                    },
                    "x",
                    self.config,
                )
            ]

    request_budget = concurrency + 3
    monkeypatch.setenv(RAY_ACTOR_CONCURRENCY_ENV, str(per_actor))
    monkeypatch.setattr(benchmark_module, "construct_clients", create_clients)
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", ConcurrentLauncher)
    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeMutationTokenizer)
    monkeypatch.setattr(
        benchmark_module,
        "prepare_prompt_requests",
        lambda **kwargs: (
            [(f"request-{index}", 1) for index in range(request_budget)],
            [{"source": "test"} for _ in range(request_budget)],
        ),
    )
    _, requests = benchmark_module.get_token_throughput_latencies(
        model="test",
        prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
        mean_input_tokens=1,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        stddev_output_tokens=0,
        num_concurrent_requests=concurrency,
        max_num_completed_requests=request_budget,
        test_timeout_s=20,
    )
    assert created == [actor_count]
    assert peak == [
        min(per_actor, concurrency - index * per_actor) for index in range(actor_count)
    ]
    assert active == [0] * actor_count
    assert len(requests) == request_budget
    assert len(launched) == len(set(launched)) == request_budget


class FakeMutationTokenizer:
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("latin1"))

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return bytes(token_ids).decode("latin1")


def test_probe_plan_determinism():
    tokenizer = FakeMutationTokenizer()
    prompts = [("abcdefghijk", 11), ("mnopqrstuvw", 11)]
    config = {
        "mode": "late_mutation",
        "trials": 2,
        "repeats_after_prime": 2,
        "schedule": "randomized_family_blocks",
        "mutation_token_offset": 8,
    }

    first = build_cache_probe_plan(prompts, config, tokenizer, seed=7)
    second = build_cache_probe_plan(prompts, config, tokenizer, seed=7)

    assert [item.metadata() for item in first] == [item.metadata() for item in second]
    assert all("prompt" not in item.metadata() for item in first)
    for family_id in {item.family_id for item in first}:
        family = [item for item in first if item.family_id == family_id]
        assert [item.role for item in family] == ["prime", "warm", "warm"]
        assert family[0].prompt != family[1].prompt
        assert family[1].expected_shared_prefix_tokens == 8


def test_prime_warm_release():
    plan = build_cache_probe_plan(
        [("abcdefghijk", 11)],
        {"mode": "exact_repeat", "trials": 1, "repeats_after_prime": 1},
        FakeMutationTokenizer(),
        seed=1,
    )
    queue = DependentPlanQueue(plan)

    prime = queue.claim()
    assert prime is not None
    assert prime.role == "prime"
    queue.complete(prime, success=True)
    warm = queue.claim()
    assert warm is not None
    assert warm.role == "warm"
    queue.complete(warm, success=True)
    assert queue.claim() is None


def test_prime_failure_skip():
    plan = build_cache_probe_plan(
        [("abcdefghijk", 11)],
        {"mode": "exact_repeat", "trials": 1, "repeats_after_prime": 2},
        FakeMutationTokenizer(),
        seed=1,
    )
    queue = DependentPlanQueue(plan)

    prime = queue.claim()
    assert prime is not None
    queue.complete(prime, success=False)

    assert queue.claim() is None
    assert [request.role for request in queue.skipped] == ["warm", "warm"]


def test_paired_verdict():
    requests = []
    for family in ("a", "b", "c"):
        for role, ttft, hit, miss in (
            ("prime", 1.0, 0, 100),
            ("warm", 0.2, 80, 20),
        ):
            requests.append(
                {
                    common_metrics.REQUEST_METADATA: {
                        "family_id": family,
                        "role": role,
                    },
                    common_metrics.ERROR_CODE: None,
                    common_metrics.TTFT: ttft,
                    common_metrics.KV_CACHE_HIT_TOKENS: hit,
                    common_metrics.KV_CACHE_MISS_TOKENS: miss,
                }
            )

    analysis = analyze_cache_probe(requests, bootstrap_samples=200, seed=3)

    assert analysis["verdict"] == "confirmed_external"
    assert analysis["paired_samples"] == 3
    assert analysis["paired_ttft_delta_s"]["median"] == pytest.approx(0.8)
    assert analysis["cache"]["by_role"]["warm"][
        "weighted_token_hit_ratio"
    ] == pytest.approx(0.8)


@pytest.mark.parametrize("stream_options", ["default", None, {"include_usage": False}])
@pytest.mark.parametrize("api_key", ["secret", "", "   ", None])
def test_client_observability(monkeypatch, stream_options, api_key):
    sent_body = {}

    def post(*args, **kwargs):
        sent_body.update(kwargs["json"])
        expected_headers = (
            {"Authorization": "Bearer secret"} if api_key == "secret" else {}
        )
        assert kwargs["headers"] == expected_headers
        return FakeResponse()

    class FakeResponse:
        status_code = 200
        headers = {
            "x-request-id": "provider-request-1",
            "authorization": "must-not-be-recorded",
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, chunk_size=None):
            yield _event({"choices": [{"delta": {"content": "one"}}]})
            time.sleep(0.001)
            yield _event(
                {
                    "choices": [{"delta": {"content": " two"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 8},
                    },
                }
            )
            yield b"data: [DONE]"

    monkeypatch.setenv("OPENAI_API_BASE", "https://provider.invalid/v1")
    if api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setattr(
        "llmperf.clients.openai.requests.post",
        post,
    )
    from llmperf.clients.openai import (
        OpenAIChatCompletionsClient,
    )

    client = OpenAIChatCompletionsClient()
    metrics, text, _ = client.llm_request(
        RequestConfig(
            model="model",
            prompt=("prompt", 9),
            metadata={"family_id": "family-1", "role": "warm"},
            sampling_params=(
                {}
                if stream_options == "default"
                else {"stream_options": stream_options}
            ),
        )
    )

    assert text == "one two"
    if stream_options == "default":
        assert sent_body["stream_options"] == {"include_usage": True}
    elif stream_options is None:
        assert "stream_options" not in sent_body
    else:
        assert sent_body["stream_options"] == stream_options
    assert metrics[common_metrics.KV_CACHE_HIT_TOKENS] == 8
    assert metrics[common_metrics.KV_CACHE_MISS_TOKENS] == 2
    assert metrics[common_metrics.PROVIDER_INPUT_TOKENS] == 10
    assert metrics[common_metrics.PROVIDER_OUTPUT_TOKENS] == 2
    assert metrics[common_metrics.REQUEST_METADATA]["role"] == "warm"
    assert metrics[common_metrics.RESPONSE_HEADERS] == {
        "x-request-id": "provider-request-1"
    }
    assert len(metrics[common_metrics.INTER_SSE_CHUNK_LAT]) == 1
    assert metrics[common_metrics.TPOT] >= 0
    assert metrics[common_metrics.REQUEST_TIMING]["first_sse_monotonic"] is not None


def test_client_inactivity(monkeypatch):
    closed = False
    observed_chunk_size = None

    class FakeResponse:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            return False

        def close(self):
            nonlocal closed
            closed = True

        def raise_for_status(self):
            return None

        def iter_lines(self, chunk_size=None):
            nonlocal observed_chunk_size
            observed_chunk_size = chunk_size
            while not closed:
                time.sleep(0.005)
                yield b": keepalive"

    monkeypatch.setenv("OPENAI_API_BASE", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        "llmperf.clients.openai.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    from llmperf.clients.openai import (
        OpenAIChatCompletionsClient,
    )

    started = time.monotonic()
    metrics, text, _ = OpenAIChatCompletionsClient().llm_request(
        RequestConfig(
            model="model",
            prompt=("prompt", 1),
            timeout_seconds=0.03,
        )
    )

    assert time.monotonic() - started < 0.5
    assert closed is True
    assert observed_chunk_size == SSE_ITER_CHUNK_SIZE
    assert text == ""
    assert metrics[common_metrics.ERROR_CODE] == -1
    assert StreamInactivityTimeout.__name__ in metrics[common_metrics.ERROR_MSG]


def test_client_progress(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, chunk_size=None):
            for text in ("one", " two", " three"):
                time.sleep(0.02)
                yield _event({"choices": [{"delta": {"content": text}}]})
            yield b"data: [DONE]"

    monkeypatch.setenv("OPENAI_API_BASE", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        "llmperf.clients.openai.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    from llmperf.clients.openai import (
        OpenAIChatCompletionsClient,
    )

    started = time.monotonic()
    metrics, text, _ = OpenAIChatCompletionsClient().llm_request(
        RequestConfig(
            model="model",
            prompt=("prompt", 1),
            timeout_seconds=0.03,
        )
    )

    assert time.monotonic() - started > 0.05
    assert text == "one two three"
    assert metrics[common_metrics.ERROR_CODE] is None
    assert metrics[common_metrics.REQUEST_TIMING]["last_text_monotonic"] is not None


@pytest.mark.parametrize("per_actor", [1, 2])
def test_probe_execution_order(monkeypatch, per_actor):
    launches = []
    created = []
    prompt_counter = iter(("abcdefghijk", "mnopqrstuvw"))

    def create_clients(llm_api, num_clients):
        created.append(num_clients)
        return list(range(num_clients))

    class FakeLauncher:
        def __init__(self, clients, request_executor=None):
            self.config = None

        def launch_requests(self, request_config):
            self.config = request_config
            launches.append(dict(request_config.metadata))

        def get_next_ready(self, block=False):
            request_config = self.config
            if request_config is None:
                raise RuntimeError("launch_requests must run before get_next_ready")
            metadata = request_config.metadata
            warm = metadata["role"] == "warm"
            metrics = {
                common_metrics.TTFT: 0.2 if warm else 1.0,
                common_metrics.E2E_LAT: 1.1,
                common_metrics.REQ_OUTPUT_THROUGHPUT: 1.0,
                common_metrics.NUM_INPUT_TOKENS: 11,
                common_metrics.NUM_OUTPUT_TOKENS: 1,
                common_metrics.ERROR_CODE: None,
                common_metrics.ERROR_MSG: "",
                common_metrics.KV_CACHE_HIT_TOKENS: 8 if warm else 0,
                common_metrics.KV_CACHE_MISS_TOKENS: 3 if warm else 11,
                common_metrics.REQUEST_METADATA: metadata,
            }
            return [(metrics, "x", self.config)]

    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeMutationTokenizer)
    monkeypatch.setattr(
        benchmark_module,
        "prepare_prompt_requests",
        lambda **kwargs: (
            [(next(prompt_counter), 11) for _ in range(2)],
            [{"source": "test"}, {"source": "test"}],
        ),
    )
    monkeypatch.setenv("LLMPERF_WORKER_RAY_ACTOR_CONCURRENCY", str(per_actor))
    monkeypatch.setattr(benchmark_module, "construct_clients", create_clients)
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FakeLauncher)

    summary, requests = benchmark_module.get_token_throughput_latencies(
        model="model",
        prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
        mean_input_tokens=64,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        stddev_output_tokens=0,
        num_concurrent_requests=2,
        max_num_completed_requests=99,
        test_timeout_s=10,
        cache_probe={
            "mode": "exact_repeat",
            "trials": 2,
            "repeats_after_prime": 1,
            "bootstrap_samples": 200,
            "confidence_level": 0.95,
            "minimum_counter_coverage": 0.8,
        },
        tokenizer_provenance={
            "id": "organization/tokenizer",
            "selection": "explicit",
            "accuracy": "compatible",
        },
    )

    assert len(requests) == 4
    assert created == [2 // per_actor]
    assert summary["cache_probe_analysis"]["verdict"] == "confirmed_external"
    for family in {item["family_id"] for item in launches}:
        roles = [item["role"] for item in launches if item["family_id"] == family]
        assert roles == ["prime", "warm"]
    assert all(item["prompt_evidence"]["source"] == "test" for item in launches)
    assert {
        item[common_metrics.REQUEST_METADATA]["completion_index"] for item in requests
    } == {
        0,
        1,
        2,
        3,
    }


def test_task_payload_replay(monkeypatch):
    launched = []

    class FakeLauncher:
        def __init__(self, clients, request_executor=None):
            self.config = None

        def launch_requests(self, request_config):
            self.config = request_config
            launched.append(request_config)

        def get_next_ready(self, block=False):
            if self.config is None:
                return []
            metrics = {
                common_metrics.TTFT: 0.2,
                common_metrics.E2E_LAT: 0.3,
                common_metrics.REQ_OUTPUT_THROUGHPUT: 1.0,
                common_metrics.NUM_INPUT_TOKENS: 64,
                common_metrics.NUM_OUTPUT_TOKENS: 1,
                common_metrics.ERROR_CODE: None,
                common_metrics.ERROR_MSG: "",
                common_metrics.REQUEST_METADATA: self.config.metadata,
            }
            return [(metrics, "x", self.config)]

    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeMutationTokenizer)
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FakeLauncher)

    base = {
        "model": "model",
        "prompt_dataset_source": PromptDatasetSource.builtin_sonnet(),
        "mean_input_tokens": 64,
        "stddev_input_tokens": 0,
        "mean_output_tokens": 1,
        "stddev_output_tokens": 0,
        "num_concurrent_requests": 1,
        "max_num_completed_requests": 1,
        "test_timeout_s": 10,
    }
    context = {
        "definition_id": "definition",
        "instance_id": "instance",
        "payload_id": "replay",
        "payload_seed": 22,
        "trial_index": 0,
        "dimensions": {},
    }
    benchmark_module.get_token_throughput_latencies(
        **base, task_context={**context, "node_id": "prime", "role": "prime"}
    )
    prime_prompt = launched[0].prompt[0]
    launched.clear()
    summary, _ = benchmark_module.get_token_throughput_latencies(
        **base, task_context={**context, "node_id": "warm", "role": "warm"}
    )

    assert launched[0].prompt[0] == prime_prompt
    assert launched[0].metadata["payload_id"] == "replay"
    assert launched[0].metadata["payload_evidence"]["source"] == "builtin-sonnet"
    assert summary["task_context"]["node_id"] == "warm"


def test_grouped_request_failure(monkeypatch):
    from llmperf.clients.factory import RAY_ACTOR_CONCURRENCY_ENV

    launched = []
    prepared_seeds = []
    barrier = threading.Barrier(8, timeout=10)

    def prepare(**kwargs):
        prepared_seeds.append(kwargs["seed"])
        assert kwargs["num_requests"] == 1
        return [(f"payload-{kwargs['seed']}", 1)], [{"source": "test"}]

    class Launcher:
        def __init__(self, clients, request_executor=None):
            assert clients == [0]
            self.config = None

        def launch_requests(self, request_config):
            self.config = request_config
            launched.append(request_config)

        def get_next_ready(self):
            barrier.wait()
            assert self.config is not None
            if self.config.metadata["node_id"] == "node-3":
                raise RuntimeError("one failed invocation")
            return [
                (
                    {
                        common_metrics.ERROR_CODE: None,
                        common_metrics.E2E_LAT: 0.1,
                        common_metrics.TTFT: 0.05,
                    },
                    "x",
                    self.config,
                )
            ]

    contexts = [
        {
            "instance_id": "instance",
            "node_id": f"node-{index}",
            "payload_id": f"payload-{index}",
            "payload_seed": index,
        }
        for index in range(8)
    ]
    monkeypatch.setenv(RAY_ACTOR_CONCURRENCY_ENV, "8")
    monkeypatch.setattr(
        benchmark_module,
        "construct_clients",
        lambda **kwargs: list(range(kwargs["num_clients"])),
    )
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", Launcher)
    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeMutationTokenizer)
    monkeypatch.setattr(benchmark_module, "prepare_prompt_requests", prepare)
    _, requests = benchmark_module.get_token_throughput_latencies(
        model="test",
        prompt_dataset_source=PromptDatasetSource.builtin_sonnet(),
        mean_input_tokens=1,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        stddev_output_tokens=0,
        num_concurrent_requests=8,
        max_num_completed_requests=8,
        test_timeout_s=20,
        task_contexts=contexts,
    )
    assert prepared_seeds == list(range(8))
    assert len(launched) == len(requests) == 8
    assert {request.prompt[0] for request in launched} == {
        f"payload-{index}" for index in range(8)
    }
    by_node = {
        result[common_metrics.REQUEST_METADATA]["node_id"]: result
        for result in requests
    }
    assert set(by_node) == {f"node-{index}" for index in range(8)}
    assert by_node["node-3"][common_metrics.ERROR_CODE] == "worker_request_error"
    assert sum(result[common_metrics.ERROR_CODE] is None for result in requests) == 7
    assert all(
        result[common_metrics.REQUEST_TIMING]["completed_utc"] for result in requests
    )


def test_task_dataset_replay(monkeypatch, tmp_path):
    launched = []

    class FakeLauncher:
        def __init__(self, clients, request_executor=None):
            self.config = None

        def launch_requests(self, request_config):
            self.config = request_config
            launched.append(request_config)

        def get_next_ready(self, block=False):
            if self.config is None:
                return []
            metrics = {
                common_metrics.TTFT: 0.2,
                common_metrics.E2E_LAT: 0.3,
                common_metrics.REQ_OUTPUT_THROUGHPUT: 1.0,
                common_metrics.NUM_INPUT_TOKENS: 32,
                common_metrics.NUM_OUTPUT_TOKENS: 1,
                common_metrics.ERROR_CODE: None,
                common_metrics.ERROR_MSG: "",
                common_metrics.REQUEST_METADATA: self.config.metadata,
            }
            return [(metrics, "x", self.config)]

    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": character * 20},
                        {"from": "gpt", "value": "answer"},
                    ]
                }
                for character in "ABCDE"
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeMutationTokenizer)
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FakeLauncher)
    base = {
        "model": "model",
        "prompt_dataset_source": PromptDatasetSource.external("sharegpt", dataset_path),
        "mean_input_tokens": 32,
        "stddev_input_tokens": 0,
        "mean_output_tokens": 1,
        "stddev_output_tokens": 0,
        "num_concurrent_requests": 1,
        "max_num_completed_requests": 1,
        "test_timeout_s": 10,
        "dataset_prompt_mode": "concatenate",
    }
    context = {
        "definition_id": "definition",
        "instance_id": "instance",
        "payload_id": "replay",
        "payload_seed": 22,
        "trial_index": 0,
        "dimensions": {},
    }

    benchmark_module.get_token_throughput_latencies(
        **base, task_context={**context, "node_id": "prime", "role": "prime"}
    )
    prime = launched.pop()
    summary, _ = benchmark_module.get_token_throughput_latencies(
        **base, task_context={**context, "node_id": "warm", "role": "warm"}
    )
    warm = launched.pop()

    assert warm.prompt == prime.prompt
    assert warm.metadata["prompt_hash"] == prime.metadata["prompt_hash"]
    assert warm.metadata["payload_evidence"] == prime.metadata["payload_evidence"]
    assert warm.metadata["payload_evidence"]["seed"] == 22
    assert summary["prompt_dataset"] == {
        "adapter": "sharegpt",
        "source": "sharegpt",
        "prompt_mode": "concatenate",
        "repeat_count": 1,
        "seed": 22,
    }
    assert "dataset_path" not in summary

"""Pure benchmark, request normalization, and cache-probe algorithm tests."""

import json
from collections import Counter

import pytest
import time

from llmperf import token_benchmark_ray as benchmark_module
from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIStreamError,
    SSE_ITER_CHUNK_SIZE,
    StreamInactivityTimeout,
    cache_metrics_from_usage,
    decode_sse_line,
)
from llmperf.cache_analysis import analyze_cache_probe, summarize_cache_counters
from llmperf.cache_probe import DependentPlanQueue, build_cache_probe_plan
from llmperf.usage import normalize_usage
from llmperf.token_benchmark_ray import (
    metrics_summary,
    normalize_request_metrics,
    sample_sharegpt_requests,
)


NORMALIZATION_CASES = [
    pytest.param(
        {
            "metrics": {
                common_metrics.INTER_TOKEN_LAT: 1.2,
                common_metrics.E2E_LAT: 2.0,
                common_metrics.NUM_INPUT_TOKENS: 10,
                common_metrics.NUM_OUTPUT_TOKENS: 0,
                common_metrics.ERROR_CODE: None,
            },
            "text": "generated",
            "tokens": 4,
            "expected": {
                common_metrics.INTER_TOKEN_LAT: 0.3,
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
                common_metrics.INTER_TOKEN_LAT: 0,
                common_metrics.E2E_LAT: 0,
                common_metrics.NUM_INPUT_TOKENS: 10,
                common_metrics.NUM_OUTPUT_TOKENS: 0,
                common_metrics.ERROR_CODE: -1,
            },
            "text": "",
            "tokens": 0,
            "expected": {
                common_metrics.INTER_TOKEN_LAT: 0,
                common_metrics.REQ_OUTPUT_THROUGHPUT: 0,
            },
        },
        id="zero-output",
    ),
]


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

    requests = sample_sharegpt_requests(
        str(dataset_path),
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
        sample_sharegpt_requests(
            str(dataset_path),
            num_requests=2,
            repeat_count=1,
            min_input_tokens=100,
            max_input_tokens=200,
            get_token_length=lambda text: len(text.split()),
        )


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
            common_metrics.INTER_TOKEN_LAT: 0.1,
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

    summary = metrics_summary(request_metrics, 1, 3)

    cache = summary[common_metrics.KV_CACHE]
    assert cache["measured_requests"] == 2
    assert cache["hit_tokens"] == 75
    assert cache["miss_tokens"] == 125
    assert cache["hit_ratio"] == 0.375
    assert cache["counter_coverage"] == 1.0


def test_incomplete_counters():
    cache = summarize_cache_counters([{common_metrics.KV_CACHE_HIT_TOKENS: 100}])

    assert cache["hit_tokens"] == 100
    assert cache["miss_tokens"] is None
    assert cache["hit_ratio"] is None
    assert cache["requests_with_complete_cache_counters"] == 0


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
    assert cache["hit_ratio"] is None


@pytest.mark.parametrize("case", NORMALIZATION_CASES)
def test_normalization(case):
    normalized = normalize_request_metrics(
        case["metrics"], case["text"], lambda text: case["tokens"]
    )

    for name, expected in case["expected"].items():
        assert normalized[name] == expected


def test_empty_summary():
    summary = metrics_summary([], 1, 2)

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
        def __init__(self, clients):
            pass

        def launch_requests(self, request_config):
            pass

        def get_next_ready(self):
            raise ValueError("Ray Actor did not receive OPENAI_API_BASE")

    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeTokenizer)
    monkeypatch.setattr(
        benchmark_module,
        "randomly_sample_sonnet_lines_prompt",
        lambda **kwargs: ("synthetic prompt", 2),
    )
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FailingLauncher)

    with pytest.raises(ValueError, match="OPENAI_API_BASE"):
        benchmark_module.get_token_throughput_latencies(
            model="glm-5.2",
            mean_input_tokens=64,
            stddev_input_tokens=0,
            mean_output_tokens=1,
            stddev_output_tokens=0,
            num_concurrent_requests=1,
            max_num_completed_requests=1,
            test_timeout_s=10,
            llm_api="openai",
        )


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
    assert prime.role == "prime"
    queue.complete(prime, success=True)
    warm = queue.claim()
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
    assert analysis["cache"]["by_role"]["warm"]["hit_ratio"] == pytest.approx(0.8)


def test_client_observability(monkeypatch):
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
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        "llmperf.ray_clients.openai_chat_completions_client.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    from llmperf.ray_clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
    )

    client = OpenAIChatCompletionsClient()
    metrics, text, _ = client.llm_request(
        RequestConfig(
            model="model",
            prompt=("prompt", 9),
            metadata={"family_id": "family-1", "role": "warm"},
        )
    )

    assert text == "one two"
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
    assert (
        metrics[common_metrics.STREAM_TIMING_SEMANTICS]["legacy_inter_token_latency"]
        == "deprecated_inter_chunk_average"
    )


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
        "llmperf.ray_clients.openai_chat_completions_client.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    from llmperf.ray_clients.openai_chat_completions_client import (
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
        "llmperf.ray_clients.openai_chat_completions_client.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    from llmperf.ray_clients.openai_chat_completions_client import (
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


def test_probe_execution_order(monkeypatch):
    launches = []
    prompt_counter = iter(("abcdefghijk", "mnopqrstuvw"))

    class FakeLauncher:
        def __init__(self, clients):
            self.config = None

        def launch_requests(self, request_config):
            self.config = request_config
            launches.append(dict(request_config.metadata))

        def get_next_ready(self, block=False):
            metadata = self.config.metadata
            warm = metadata["role"] == "warm"
            metrics = {
                common_metrics.INTER_TOKEN_LAT: 0.1,
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
        "randomly_sample_sonnet_lines_prompt",
        lambda **kwargs: (next(prompt_counter), 11),
    )
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FakeLauncher)

    summary, requests = benchmark_module.get_token_throughput_latencies(
        model="model",
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
    assert summary["cache_probe_analysis"]["verdict"] == "confirmed_external"
    for family in {item["family_id"] for item in launches}:
        roles = [item["role"] for item in launches if item["family_id"] == family]
        assert roles == ["prime", "warm"]
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
        def __init__(self, clients):
            self.config = None

        def launch_requests(self, request_config):
            self.config = request_config
            launched.append(request_config)

        def get_next_ready(self, block=False):
            if self.config is None:
                return []
            metrics = {
                common_metrics.INTER_TOKEN_LAT: 0.1,
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

    def generated_prompt(**kwargs):
        text = f"prompt-{benchmark_module.random.random()}"
        return text, 64

    monkeypatch.setattr(benchmark_module, "get_tokenizer", FakeMutationTokenizer)
    monkeypatch.setattr(
        benchmark_module, "randomly_sample_sonnet_lines_prompt", generated_prompt
    )
    monkeypatch.setattr(benchmark_module, "construct_clients", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_module, "RequestsLauncher", FakeLauncher)

    base = {
        "model": "model",
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
        **base, task_request={**context, "node_id": "prime", "role": "prime"}
    )
    prime_prompt = launched[0].prompt[0]
    launched.clear()
    summary, _ = benchmark_module.get_token_throughput_latencies(
        **base, task_request={**context, "node_id": "warm", "role": "warm"}
    )

    assert launched[0].prompt[0] == prime_prompt
    assert launched[0].metadata["payload_id"] == "replay"
    assert summary["task_request"]["node_id"] == "warm"

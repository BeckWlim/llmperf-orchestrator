import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIStreamError,
    cache_metrics_from_usage,
    decode_sse_line,
)
from llmperf_backend.models import BenchmarkRunnerSpec
from token_benchmark_ray import (
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


def test_campaign():
    campaign_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "deepseek-v4-pro-kvcache-campaign.yaml"
    )
    plan = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))

    assert plan["campaign"]["name"] == "deepseek-v4-pro-kvcache-reality"
    assert len(plan["runners"]) == 2
    runners = [
        BenchmarkRunnerSpec.model_validate(item) for item in plan["runners"]
    ]
    assert runners[0].benchmark.dataset.format == "sharegpt"
    assert runners[0].benchmark.dataset.id.endswith("ShareGPT_Vicuna_unfiltered")
    assert runners[0].benchmark.dataset_repeat_count == 1
    assert runners[0].benchmark.dataset_seed != runners[1].benchmark.dataset_seed
    assert runners[1].benchmark.dataset_repeat_count == 4
    assert runners[1].benchmark.additional_sampling_params["stream_options"] == {
        "include_usage": True
    }


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


def test_sharegpt_sampling_requires_enough_matching_prompts(tmp_path):
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

    assert summary[common_metrics.KV_CACHE] == {
        "measured_requests": 2,
        "hit_tokens": 75,
        "miss_tokens": 125,
        "hit_ratio": 0.375,
    }


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

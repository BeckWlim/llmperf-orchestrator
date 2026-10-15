"""Unknown cache counters must not discard completed requests."""

import pytest

from llmperf import common_metrics
from llmperf.cache_analysis import summarize_cache_counters
from llmperf.token_benchmark_ray import metrics_summary
from llmperf.usage import normalize_counter, normalize_usage


@pytest.mark.parametrize("counter", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_counters(counter: float) -> None:
    assert normalize_counter(counter) is None
    cache = summarize_cache_counters(
        [
            {
                common_metrics.KV_CACHE_HIT_TOKENS: counter,
                common_metrics.KV_CACHE_MISS_TOKENS: counter,
            }
        ]
    )
    assert cache["requests_total"] == 1
    assert cache["measured_requests"] == 0
    assert cache["counter_coverage"] == 0
    assert cache["weighted_token_hit_ratio"] is None
    usage = normalize_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": counter},
        }
    )
    assert usage.provider_input_tokens == 100
    assert usage.hit_tokens is None


def test_mixed_missing_counters() -> None:
    request_metrics = [
        {
            common_metrics.TTFT: 0.2,
            common_metrics.E2E_LAT: 1.0,
            common_metrics.REQ_OUTPUT_THROUGHPUT: 2.0,
            common_metrics.NUM_INPUT_TOKENS: 100,
            common_metrics.NUM_OUTPUT_TOKENS: 2,
            common_metrics.ERROR_CODE: None,
            **counters,
        }
        for counters in (
            {
                common_metrics.KV_CACHE_HIT_TOKENS: 75,
                common_metrics.KV_CACHE_MISS_TOKENS: 25,
            },
            {},
        )
    ]
    summary = metrics_summary(request_metrics, 1.0, 3.0)
    assert summary[common_metrics.NUM_COMPLETED_REQUESTS] == 2
    assert summary[common_metrics.NUM_ERRORS] == 0
    cache = summary[common_metrics.KV_CACHE]
    assert cache["counter_coverage"] == 0.5
    assert cache["complete_hit_tokens"] == 75
    assert cache["complete_miss_tokens"] == 25
    assert cache["weighted_token_hit_ratio"] == 0.75

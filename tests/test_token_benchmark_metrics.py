from pathlib import Path
import sys

import pytest

from llmperf import common_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_benchmark_ray import metrics_summary, normalize_request_metrics


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


@pytest.mark.parametrize("case", NORMALIZATION_CASES)
def test_normalization(case):
    normalized = normalize_request_metrics(
        case["metrics"], case["text"], lambda text: case["tokens"]
    )

    for name, expected in case["expected"].items():
        assert normalized[name] == expected


def test_no_completed_requests():
    summary = metrics_summary([], 1, 2)

    assert summary[common_metrics.NUM_REQ_STARTED] == 0
    assert summary[common_metrics.NUM_COMPLETED_REQUESTS] == 0
    assert summary[common_metrics.NUM_ERRORS] == 0
    assert summary[common_metrics.ERROR_CODE_FREQ] == {}
    assert summary[common_metrics.OUTPUT_THROUGHPUT] == 0

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llmperf import common_metrics
from llmperf_backend.persistence import FAILED, SUCCEEDED, _runner_list_dict, json_safe
from llmperf_backend.worker import summarize_outcome


OUTCOME_CASES = [
    pytest.param(
        {
            "summary": {
                "results": {
                    common_metrics.NUM_REQ_STARTED: 1,
                    common_metrics.NUM_COMPLETED_REQUESTS: 0,
                    common_metrics.NUM_ERRORS: 1,
                }
            },
            "requests": [
                {
                    common_metrics.ERROR_CODE: -1,
                    common_metrics.ERROR_MSG: "JSONDecodeError: invalid stream",
                }
            ],
            "status": FAILED,
            "outcome": "failed",
            "message": (
                "No benchmark requests completed (1 failed); first error [-1]: "
                "JSONDecodeError: invalid stream"
            ),
            "first_error": {
                "code": -1,
                "message": "JSONDecodeError: invalid stream",
            },
        },
        id="failed",
    ),
    pytest.param(
        {
            "summary": {
                "results": {
                    common_metrics.NUM_REQ_STARTED: 2,
                    common_metrics.NUM_COMPLETED_REQUESTS: 1,
                    common_metrics.NUM_ERRORS: 1,
                }
            },
            "requests": [],
            "status": SUCCEEDED,
            "outcome": "degraded",
            "message": "1 request completed; 1 request failed",
            "first_error": None,
        },
        id="degraded",
    ),
    pytest.param(
        {
            "summary": {
                "timed_out": True,
                "results": {
                    common_metrics.NUM_REQ_STARTED: 0,
                    common_metrics.NUM_COMPLETED_REQUESTS: 0,
                    common_metrics.NUM_ERRORS: 0,
                },
            },
            "requests": [],
            "status": FAILED,
            "outcome": "failed",
            "message": "No benchmark requests completed before timeout",
            "first_error": None,
        },
        id="timeout",
    ),
]


@pytest.mark.parametrize("case", OUTCOME_CASES)
def test_outcome(case):
    status, message = summarize_outcome(case["summary"], case["requests"])

    assert status == case["status"]
    assert message == case["message"]
    assert case["summary"]["outcome"]["status"] == case["outcome"]
    assert case["summary"]["outcome"]["first_error"] == case["first_error"]


def test_finite_float():
    assert json_safe(1.0) == 1.0


def test_list_projection():
    created_at = datetime(2026, 8, 12, tzinfo=timezone.utc)
    runner = SimpleNamespace(
        id="runner-1",
        campaign_id=None,
        runner_plan_id=None,
        plan_occurrence=None,
        scheduled_for=None,
        label="smoke",
        created_by="tester",
        status=FAILED,
        benchmark_config={"provider": "aliyun", "model": "glm-5.2"},
        summary={
            "results": {"error_rate": 1.0},
            "outcome": {
                "requests_started": 1,
                "requests_completed": 0,
                "requests_failed": 1,
                "message": "No benchmark requests completed",
            },
        },
        error_message="large worker error",
        created_at=created_at,
        started_at=None,
        finished_at=created_at,
        scheduler_id="scheduler-1",
        process_id=62791,
        exit_code=1,
        stdout="large stdout",
        stderr="large stderr",
    )

    listed = _runner_list_dict(runner)

    assert listed["provider"] == "aliyun"
    assert listed["model"] == "glm-5.2"
    assert listed["requests"] == {
        "started": 1,
        "completed": 0,
        "failed": 1,
        "error_rate": 1.0,
    }
    assert listed["scheduler_id"] == "scheduler-1"
    assert listed["worker"] == {"process_id": 62791, "exit_code": 1}
    assert "summary" not in listed
    assert "stdout" not in listed
    assert "stderr" not in listed

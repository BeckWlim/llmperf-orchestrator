import pytest

from llmperf.cache_sweep import analyze_cache_protocols, analyze_cache_retention


def _runner(ttft, request_hit=None, token_hit=None):
    cache = {}
    if request_hit is not None:
        cache["request_hit_ratio"] = request_hit
    if token_hit is not None:
        cache["weighted_token_hit_ratio"] = token_hit
    return {
        "summary": {
            "results": {
                "ttft_s": {"quantiles": {"p50": ttft}},
                "kv_cache": cache,
            }
        }
    }


def test_cache_curve():
    instances = [
        {
            "protocol_instance_id": "instance-1",
            "protocol_definition_id": "definition-1",
            "protocol": "cache-retention/v1",
            "state": "completed",
            "spec": {"delay_seconds": 300, "trial_index": 0},
            "outcome": {"actual_delay_seconds": 302.5},
        }
    ]
    dispatches = [
        {
            "protocol_instance_id": "instance-1",
            "role": role,
            "runner_id": runner_id,
        }
        for role, runner_id in (
            ("prime", "prime-1"),
            ("warm", "warm-1"),
            ("cold_control", "control-1"),
        )
    ]
    runners = {
        "prime-1": _runner(1.0),
        "warm-1": _runner(0.2, request_hit=1.0, token_hit=0.8),
        "control-1": _runner(1.1),
    }

    analysis = analyze_cache_retention(instances, dispatches, runners)[0]

    point = analysis["curve"][0]
    assert point["verdict"] == "accounting_observed"
    assert point["actual_delay_seconds"]["median"] == 302.5
    assert point["cold_control_minus_warm_ttft_s"]["median"] == pytest.approx(0.9)
    assert analysis["last_accounting_observed_delay_seconds"] == 300


def test_residency_curve():
    instances = [
        {
            "protocol_instance_id": "chain-1",
            "protocol_definition_id": "definition-2",
            "protocol": "cache-residency/v1",
            "state": "completed",
            "spec": {
                "chain_index": 0,
                "schedule": {
                    "kind": "relative",
                    "offsets_seconds": [3600, 7200],
                },
                "observations": [
                    {
                        "observation_index": 0,
                        "offset_seconds": 3600,
                        "scheduled_at": None,
                    },
                    {
                        "observation_index": 1,
                        "offset_seconds": 7200,
                        "scheduled_at": None,
                    },
                ],
            },
            "outcome": {
                "observations": {
                    "0": {"warm_actual_delay_seconds": 3602.0},
                    "1": {"warm_actual_delay_seconds": 7203.0},
                }
            },
        }
    ]
    dispatches = [
        {
            "protocol_instance_id": "chain-1",
            "role": role,
            "runner_id": runner_id,
        }
        for role, runner_id in (
            ("prime", "prime-2"),
            ("warm:0", "warm-2-0"),
            ("cold_control:0", "control-2-0"),
            ("warm:1", "warm-2-1"),
            ("cold_control:1", "control-2-1"),
        )
    ]
    runners = {
        "prime-2": _runner(1.0),
        "warm-2-0": _runner(0.2, request_hit=1.0, token_hit=0.8),
        "control-2-0": _runner(1.1),
        "warm-2-1": _runner(0.3, request_hit=1.0, token_hit=0.7),
        "control-2-1": _runner(1.2),
    }

    analyses = analyze_cache_protocols(instances, dispatches, runners)

    assert len(analyses) == 1
    analysis = analyses[0]
    assert analysis["pairing"] == "bundled_prime_mapping"
    assert analysis["interpretation"] == "access_conditioned_residency"
    assert [point["planned_offset_seconds"] for point in analysis["curve"]] == [
        3600,
        7200,
    ]
    assert analysis["curve"][1]["actual_delay_seconds"]["median"] == 7203.0
    assert analysis["last_accounting_observed_offset_seconds"] == 7200

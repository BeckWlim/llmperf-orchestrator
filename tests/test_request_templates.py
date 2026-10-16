"""Worker template expansion and structured message reuse contracts."""

from datetime import datetime, timezone

import pytest

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.models import RequestConfig
from llmperf_backend.configuration.models import BenchmarkRunnerSpec
from llmperf_backend.compiler.requests import (
    RequestPlan,
    compile_requests,
    planned_worker_count,
    planned_token_budget,
)
from llmperf_backend.compiler.tasks import TaskRecipe, TaskCompiler
from llmperf_backend.runtime.messages import WorkerCompletion
from llmperf_backend.runtime.runner import WorkerResult, WorkflowTaskLogic


def test_template_defaults():
    runner = BenchmarkRunnerSpec.model_validate(
        {"request": {}, "provider": "test", "model": "test"}
    )
    execution = runner.execution_benchmark()
    assert planned_worker_count(execution) == 1
    assert runner.concurrent_requests is None
    assert "max_submissions" not in execution
    assert runner.request is not None
    assert "timeout" not in runner.request.model_dump()
    assert "concurrent_requests" not in runner.request.model_dump()


def test_omitted_request_defaults():
    defaults = {
        "provider": "test",
        "model": "test",
        "repeat": 7,
        "steps": [{"name": "configured"}],
        "mean_input_tokens": 128,
        "mean_output_tokens": 16,
    }
    omitted = BenchmarkRunnerSpec().execution_benchmark(defaults)
    explicit = BenchmarkRunnerSpec.model_validate({"request": {}}).execution_benchmark(
        defaults
    )
    assert omitted == explicit
    assert planned_worker_count(omitted) == 1
    assert omitted["steps"] is None
    assert omitted["input"] == "default"
    assert omitted["dataset_repeat_count"] == 1
    assert omitted["mean_input_tokens"] == 128
    assert omitted["mean_output_tokens"] == 16
    assert defaults["repeat"] == 7


@pytest.mark.parametrize(
    "removed",
    [
        "max_submissions",
        "request_steps",
        "timeout",
        "concurrent_requests",
        "provider",
        "model",
        "tokenizer",
        "mean_input_tokens",
        "mean_output_tokens",
        "additional_sampling_params",
        "prompt",
        "dataset",
    ],
)
def test_removed_template_fields(removed):
    with pytest.raises(ValueError, match="Extra inputs"):
        BenchmarkRunnerSpec.model_validate(
            {"request": {removed: 1}, "provider": "test", "model": "test"}
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        BenchmarkRunnerSpec.model_validate(
            {"benchmark": {"provider": "test", "model": "test"}}
        )


def test_template_expansion():
    runner = BenchmarkRunnerSpec.model_validate(
        {
            "request": {
                "repeat": 3,
                "steps": [
                    {"name": "first", "input": "first"},
                    {
                        "name": "next",
                        "depends_on": [{"step": "first", "reuse_messages": True}],
                        "after": {"value": 500, "unit": "ms"},
                        "input": "next",
                    },
                ],
            },
            "provider": "test",
            "model": "test",
            "inputs": {
                "first": {"text": "remember this"},
                "next": {"text": "repeat it"},
            },
        }
    )
    execution = runner.execution_benchmark()
    prepared = [
        RequestStep(
            str(index),
            RequestConfig(
                model="test",
                prompt=(str(index), 1),
                sampling_params={"temperature": 0, "top_p": 0.9},
            ),
        )
        for index in range(6)
    ]
    compiled = compile_requests(execution, prepared)
    assert planned_worker_count(execution) == 6
    assert [step.step_id for step in compiled] == [
        "0.first",
        "0.next",
        "1.first",
        "1.next",
        "2.first",
        "2.next",
    ]
    for repetition in range(3):
        opening, followup = compiled[repetition * 2 : repetition * 2 + 2]
        assert not opening.dependencies
        assert followup.dependencies == (opening.step_id,)
        assert followup.message_sources == (opening.step_id,)
        assert followup.delay_seconds == 0.5
        assert followup.request.sampling_params == {"temperature": 0, "top_p": 0.9}
    assert prepared[1].request.sampling_params == {"temperature": 0, "top_p": 0.9}
    with pytest.raises(ValueError, match="Prepared Workers"):
        compile_requests(execution, prepared[:1])


def test_definition_accounting():
    runner = BenchmarkRunnerSpec.model_validate(
        {
            "request": {
                "repeat": 3,
                "steps": [
                    {"name": "a"},
                    {
                        "name": "b",
                        "depends_on": [{"step": "a", "reuse_messages": True}],
                    },
                ],
            },
            "provider": "test",
            "model": "test",
            "mean_output_tokens": 2,
            "inputs": {"default": {"tokens": {"mean": 10, "stddev": 150}}},
        }
    )
    benchmark = runner.execution_benchmark()
    recipe = TaskRecipe.model_validate(
        {
            "name": "history",
            "instances": {"trials": 2},
            "workflow": [{"run": {"name": "requests"}}],
        }
    )
    estimate = TaskCompiler.estimate(recipe, benchmark=benchmark)
    assert estimate["runners"] == 2
    assert estimate["requests"] == 12
    assert planned_token_budget(benchmark) == 3 * (12 + 24)
    assert (
        TaskCompiler.estimate_token_budget(
            recipe.model_dump(mode="json"), 10, 2, benchmark
        )
        == 216
    )


@pytest.mark.parametrize(
    "steps",
    [
        [{"name": "a", "depends_on": [{"step": "missing", "reuse_messages": True}]}],
        [{"name": "a", "depends_on": ["b"]}, {"name": "b"}],
        [{"name": "a"}, {"name": "a"}],
        [
            {"name": "a"},
            {
                "name": "b",
                "depends_on": [{"step": "a", "reuse_messages": True}],
                "additional_sampling_params": {"messages": []},
            },
        ],
    ],
)
def test_invalid_history_graph(steps):
    with pytest.raises(ValueError):
        RequestPlan.model_validate({"steps": steps})


def test_message_structure():
    original_messages = [
        {"role": "system", "content": "Preserve structure."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "remember"}],
            "name": "client",
        },
    ]
    original_request = RequestConfig(
        model="test",
        prompt=("fallback", 1),
        sampling_params={"messages": original_messages},
    )
    completion = WorkerCompletion(
        step_id="first",
        metrics={},
        generated_text="remembered",
        request=original_request,
    )
    persisted = WorkerCompletion.model_validate_json(completion.model_dump_json())
    result = WorkerResult(
        "first",
        (persisted.metrics, persisted.generated_text, persisted.request),
        datetime.now(timezone.utc),
    )
    child = RequestStep(
        "next",
        RequestConfig(model="test", prompt=("what was it?", 2)),
        ("first",),
        message_sources=("first",),
    )
    logic = WorkflowTaskLogic(
        [RequestStep("first", original_request), child], datetime.now(timezone.utc)
    )
    resolved = logic.request(child, {"first": result})
    assert resolved.request.sampling_params is not None
    messages = resolved.request.sampling_params["messages"]
    assert messages == [
        *original_messages,
        {"role": "assistant", "content": "remembered"},
        {"role": "user", "content": "what was it?"},
    ]
    messages[1]["content"][0]["text"] = "changed"
    assert original_messages[1]["content"][0]["text"] == "remember"
    assert persisted.messages()[1]["content"][0]["text"] == "remember"


def test_message_fanin():
    now = datetime.now(timezone.utc)
    shared = [{"role": "system", "content": "shared context"}]
    left = RequestStep(
        "left",
        RequestConfig(
            model="test", prompt=("left", 1), sampling_params={"messages": shared}
        ),
    )
    right = RequestStep(
        "right",
        RequestConfig(
            model="test", prompt=("right", 1), sampling_params={"messages": shared}
        ),
    )
    gate = RequestStep("gate", RequestConfig(model="test", prompt=("gate", 1)))
    merge = RequestStep(
        "merge",
        RequestConfig(model="test", prompt=("combine", 1)),
        ("left", "right", "gate"),
        message_sources=("right", "left"),
    )
    logic = WorkflowTaskLogic([left, right, gate, merge], now)
    results = {
        "left": WorkerResult("left", ({}, "L", left.request), now),
        "right": WorkerResult("right", ({}, "R", right.request), now),
    }
    assert not logic.eligibility(merge, now, results).ready
    results["gate"] = WorkerResult("gate", ({}, "G", gate.request), now)
    assert logic.eligibility(merge, now, results).ready
    resolved = logic.request(merge, results)
    assert resolved.request.sampling_params == {
        "messages": [
            *shared,
            {"role": "assistant", "content": "R"},
            *shared,
            {"role": "assistant", "content": "L"},
            {"role": "user", "content": "combine"},
        ]
    }
    # A shared ancestor is intentionally preserved twice; content is never deduplicated.
    results["gate"] = WorkerResult(
        "gate", ({"error_code": "failed"}, "", gate.request), now
    )
    assert logic.eligibility(merge, now, results).blocked


def test_wait_only():
    plan = RequestPlan.model_validate(
        {"steps": [{"name": "a"}, {"name": "b", "depends_on": ["a"]}]}
    )
    assert plan.steps is not None
    assert plan.steps[1].dependencies == ("a",)
    assert plan.steps[1].message_sources == ()
    with pytest.raises(ValueError, match="duplicate"):
        RequestPlan.model_validate(
            {
                "steps": [
                    {"name": "a"},
                    {
                        "name": "b",
                        "depends_on": ["a", {"step": "a", "reuse_messages": True}],
                    },
                ]
            }
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        RequestPlan.model_validate({"steps": [{"name": "a", "history_from": "b"}]})


def test_named_input_preparation(monkeypatch):
    from llmperf_backend.runtime import preparation

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    calls = []

    def generate(**options):
        calls.append(options)
        count = options["num_requests"]
        size = options["mean_input_tokens"]
        return [("x" * size, size) for _ in range(count)], [
            {"source": "test"} for _ in range(count)
        ]

    monkeypatch.setattr(preparation, "prepare_prompt_requests", generate)
    runner = BenchmarkRunnerSpec.model_validate(
        {
            "provider": "test",
            "model": "test",
            "mean_output_tokens": 2,
            "inputs": {
                "opening": {"tokens": 128},
                "followup": {"tokens": {"mean": 16, "stddev": 2}},
                "literal": {"text": "summarize"},
            },
            "request": {
                "repeat": 2,
                "steps": [
                    {"name": "a", "input": "opening"},
                    {
                        "name": "b",
                        "input": "followup",
                        "depends_on": [{"step": "a", "reuse_messages": True}],
                    },
                    {"name": "c", "input": "literal", "depends_on": ["b"]},
                ],
            },
        }
    )
    execution = runner.execution_benchmark()
    prepared = preparation.prepare_named_inputs(execution, {}, Tokenizer())
    assert prepared is not None
    assert [entry[0][1] for entry in prepared] == [128, 16, 9, 128, 16, 9]
    assert [
        (call["mean_input_tokens"], call["stddev_input_tokens"], call["num_requests"])
        for call in calls
    ] == [(128, 0, 2), (16, 2, 2)]
    assert calls[0]["seed"] != calls[1]["seed"]
    assert preparation.prepare_named_inputs(execution, {}, Tokenizer()) == prepared
    assert (
        "tokens" not in runner.request.model_dump()
        if runner.request is not None
        else False
    )
    with pytest.raises(ValueError, match="unknown Runner input"):
        BenchmarkRunnerSpec.model_validate(
            {"inputs": {"known": {"tokens": 10}}, "request": {"input": "missing"}}
        )

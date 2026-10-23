"""Complete Worker execution starts input preparation only inside an acquired slot."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import threading
from unittest.mock import Mock

import pytest

from llmperf.clients.models import RequestConfig
from llmperf.datasets.tokenizer import TOKENIZER_PATH
from llmperf_backend.runtime.request_pool import RequestActor, RequestRouter
from llmperf_backend.runtime.worker import INPUT_DATASET_PATHS
from llmperf_backend.runtime.worker_input import WorkerInputPreparer, worker_steps
from llmperf_backend.compiler.rounds import expand_rounds, expand_runner_group


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("ascii"))


def benchmark(**overrides):
    return {
        "model": "test",
        "adapter": "openai",
        "timeout": 10,
        "repeat": 3,
        "mean_input_tokens": 200,
        "stddev_input_tokens": 0,
        "shared_prefix_tokens": 160,
        "shared_prefix_seed": 1234,
        "dataset_seed": 789,
        "dataset_prompt_mode": "concatenate",
        "mean_output_tokens": 2,
        "stddev_output_tokens": 0,
        **overrides,
    }


def test_actor_family(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "indexes"))
    tokenizer = Mock(wraps=Tokenizer())
    monkeypatch.setattr(
        "llmperf_backend.runtime.worker_input._prepared_tokenizer",
        lambda *args: tokenizer,
    )
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": f"record-{index:06d}"}]}
                for index in range(1000)
            ]
        )
    )
    environment = {
        TOKENIZER_PATH: "unused",
        INPUT_DATASET_PATHS: json.dumps({"default": str(corpus_path)}),
    }
    dataset = {"adapter": "sharegpt-user", "filename": corpus_path.name}
    first_actor = WorkerInputPreparer()
    other_actor = WorkerInputPreparer()
    first_trial = worker_steps(benchmark(dataset=dataset), calibrated=True)
    second_trial = worker_steps(
        benchmark(dataset=dataset, dataset_seed=987), calibrated=True
    )
    assert all(step.request.prompt[0] == "" for step in first_trial)
    first = first_actor.prepare(first_trial[0].request, environment)
    initial_calls = tokenizer.encode.call_count
    assert initial_calls == 128
    other_family = worker_steps(
        benchmark(dataset=dataset, shared_prefix_seed=4321), calibrated=True
    )[0].request
    first_actor.prepare(other_family, environment)
    # A new family renders a prefix but must not recalibrate the same corpus.
    assert tokenizer.encode.call_count == initial_calls
    replay = other_actor.prepare(first_trial[0].request, environment)
    next_request = first_actor.prepare(first_trial[1].request, environment)
    next_trial = other_actor.prepare(second_trial[0].request, environment)
    assert first.prompt == replay.prompt
    assert (
        first.prompt[0][:160]
        == next_request.prompt[0][:160]
        == next_trial.prompt[0][:160]
    )
    assert (
        len(
            {
                first.prompt[0][160:],
                next_request.prompt[0][160:],
                next_trial.prompt[0][160:],
            }
        )
        == 3
    )
    assert first.preparation is None
    assert first.metadata is not None and next_trial.metadata is not None
    assert (
        first.metadata["prompt_evidence"]["prefix_hash"]
        == next_trial.metadata["prompt_evidence"]["prefix_hash"]
    )


def test_round_inputs(monkeypatch, tmp_path):
    """A completion-gated round changes suffixes, including repeated hot inputs."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "indexes"))
    monkeypatch.setattr(
        "llmperf_backend.runtime.worker_input._prepared_tokenizer",
        lambda *args: Tokenizer(),
    )
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": f"record-{index:06d}"}]}
                for index in range(1000)
            ]
        )
    )
    dataset = {"adapter": "sharegpt-user", "filename": corpus_path.name}
    input_definition = {
        "tokens": 200,
        "shared_prefix_tokens": 160,
        "dataset": dataset,
        "dataset_prompt_mode": "concatenate",
    }
    template = {
        "benchmark": benchmark(
            dataset=dataset,
            repeat=1,
            inputs={"warm": input_definition, "hot": input_definition},
            steps=[
                {"name": "first", "input": "warm"},
                {"name": "second", "input": "hot"},
                {"name": "third", "input": "hot"},
            ],
        )
    }
    rounds = expand_rounds([template, template], 3, 900, "family")
    environment = {
        TOKENIZER_PATH: "unused",
        INPUT_DATASET_PATHS: json.dumps(
            {name: str(corpus_path) for name in ("warm", "hot")}
        ),
    }
    preparer = WorkerInputPreparer()
    prefixes = {}
    suffixes = {}
    for item in rounds:
        family = item["metadata"]["runner_round"]["family"]
        for step in worker_steps(item["benchmark"], calibrated=True):
            assert step.request.preparation is not None
            identity = (family, step.request.preparation.input_name)
            prepared = preparer.prepare(step.request, environment)
            prefixes.setdefault(identity, set()).add(prepared.prompt[0][:160])
            suffixes.setdefault(identity, set()).add(prepared.prompt[0][160:])
    assert len(prefixes) == 4
    assert all(len(values) == 1 for values in prefixes.values())
    assert len(set.union(*prefixes.values())) == 4
    assert all(
        len(values) == (6 if identity[1] == "hot" else 3)
        for identity, values in suffixes.items()
    )


def test_phase_inputs(monkeypatch, tmp_path):
    """Actual actor preparation preserves targets, changes suffixes and pressure."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "indexes"))
    monkeypatch.setattr(
        "llmperf_backend.runtime.worker_input._prepared_tokenizer",
        lambda *args: Tokenizer(),
    )
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": f"record-{index:06d}"}]}
                for index in range(1000)
            ]
        )
    )
    environment = {
        TOKENIZER_PATH: "unused",
        INPUT_DATASET_PATHS: json.dumps({"default": str(corpus_path)}),
    }
    dataset = {"adapter": "sharegpt-user", "filename": corpus_path.name}
    templates = [
        runner
        for phase in ("load", "pressure", "read")
        for runner in expand_runner_group(
            {
                "benchmark": benchmark(
                    dataset=dataset,
                    repeat=1,
                    shared_prefix_tokens=0 if phase == "pressure" else 160,
                )
            },
            "pressure" if phase == "pressure" else "target",
            {"member": [1, 2]},
            phase,
        )
    ]
    runners = expand_rounds(templates, 3, 42, "family", ("load", "pressure", "read"))
    target_prefixes = {}
    target_suffixes = {}
    pressure_prompts = set()
    actors = [WorkerInputPreparer(), WorkerInputPreparer()]
    for index, runner in enumerate(runners):
        step = worker_steps(runner["benchmark"], calibrated=True)[0]
        prompt = actors[index % 2].prepare(step.request, environment).prompt[0]
        group = runner["metadata"]["runner_group"]
        if group["name"] == "pressure":
            pressure_prompts.add(prompt)
        else:
            family = group["dimensions"]["member"]
            target_prefixes.setdefault(family, set()).add(prompt[:160])
            target_suffixes.setdefault(family, set()).add(prompt[160:])
    assert len(pressure_prompts) == 6
    assert len(target_prefixes) == 2
    assert all(len(values) == 1 for values in target_prefixes.values())
    assert len(set.union(*target_prefixes.values())) == 2
    assert all(len(values) == 6 for values in target_suffixes.values())


def test_slot_before_preparation(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    prepared = []
    actor = RequestActor()
    client = Mock()
    client.llm_request.side_effect = lambda request: (
        {"provider_output_tokens": 1},
        "ok",
        request,
    )
    monkeypatch.setattr(
        "llmperf_backend.runtime.request_pool._client_class",
        lambda _: lambda environment: client,
    )

    def prepare(request, environment):
        prepared.append(request.model)
        entered.set()
        assert release.wait(3)
        return request.model_copy(
            update={"preparation": None, "prompt": ("prepared", 8)}
        )

    monkeypatch.setattr(actor.preparer, "prepare", prepare)

    class Method:
        async def remote(self, request, environment):
            return await asyncio.to_thread(actor.execute, request, environment)

    class Handle:
        @property
        def execute(self):
            return Method()

    async def run():
        router = RequestRouter([Handle()], 1, 8)
        recipes = worker_steps(benchmark(repeat=2), calibrated=True)
        tasks = [
            asyncio.create_task(router.execute(str(index), step.request, {}))
            for index, step in enumerate(recipes)
        ]
        try:

            async def wait_preparing():
                while not entered.is_set():
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_preparing(), 2)
            assert len(prepared) == 1
            assert client.llm_request.call_count == 0
            assert router.status()["executing_requests"] == 1
            assert router.status()["queued_requests"] == 1
            release.set()
            responses = await asyncio.gather(*tasks)
            assert len(prepared) == client.llm_request.call_count == 2
            assert all(response[2].prompt == ("prepared", 8) for response in responses)
            assert router.status()["available_slots"] == 1
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_preparation_failure(monkeypatch):
    actor = RequestActor()
    provider = Mock()
    monkeypatch.setattr("llmperf_backend.runtime.request_pool._client_class", provider)
    prepare_failure = Mock(side_effect=ValueError("bad corpus"))
    monkeypatch.setattr(actor.preparer, "prepare", prepare_failure)
    request = worker_steps(benchmark(), calibrated=True)[0].request
    response = actor.execute(request, {})
    assert response[0]["error_code"] == "worker_preparation_error"
    assert "bad corpus" in response[0]["error_msg"]
    provider.assert_not_called()
    expired = request.model_copy(
        update={"deadline_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
    )
    prepare_failure.reset_mock()
    assert actor.execute(expired, {})[0]["error_code"] == "worker_preparation_error"
    prepare_failure.assert_not_called()


def test_literal_history(monkeypatch):
    monkeypatch.setattr(
        "llmperf_backend.runtime.worker_input._prepared_tokenizer",
        lambda *args: Tokenizer(),
    )
    descriptor = worker_steps(
        benchmark(
            prompt="question",
            input_messages=[
                {"role": "user", "content": "before"},
                {"role": "assistant", "content": "answer"},
            ],
        ),
        calibrated=True,
    )[0].request
    prepared = WorkerInputPreparer().prepare(descriptor, {TOKENIZER_PATH: "unused"})
    assert prepared.sampling_params is not None and prepared.metadata is not None
    assert [message["content"] for message in prepared.sampling_params["messages"]] == [
        "before",
        "answer",
        "question",
    ]
    assert prepared.metadata["history_version"] == 2


@pytest.mark.parametrize(
    "literal", [None, "a literal input that changes between requests"]
)
def test_default_estimation(monkeypatch, literal):
    from llmperf_backend.configuration.models import BenchmarkRunnerSpec

    assert BenchmarkRunnerSpec().input_token_count == "estimate"
    tokenizer = Mock(wraps=Tokenizer())
    monkeypatch.setattr(
        "llmperf_backend.runtime.worker_input._prepared_tokenizer",
        lambda *args: tokenizer,
    )
    environment = {TOKENIZER_PATH: "general-tokenizer"}
    preparer = WorkerInputPreparer()
    initial_benchmark = benchmark(prompt=literal)
    initial_request = worker_steps(initial_benchmark)[0].request
    first = preparer.prepare(initial_request, environment)
    initial_calls = tokenizer.encode.call_count
    assert initial_calls > 0
    later_benchmark = benchmark(
        prompt=literal, dataset_seed=321, shared_prefix_seed=4321
    )
    later = preparer.prepare(worker_steps(later_benchmark)[0].request, environment)
    assert tokenizer.encode.call_count == initial_calls
    assert first.metadata is not None and later.metadata is not None
    assert (
        first.metadata["prompt_evidence"]["input_count_source"] == "calibrated_estimate"
    )
    assert (
        later.metadata["prompt_evidence"]["input_count_source"] == "calibrated_estimate"
    )
    exact_benchmark = benchmark(prompt=literal, input_token_count="tokenizer")
    exact_request = worker_steps(exact_benchmark)[0].request
    exact = preparer.prepare(exact_request, environment)
    assert exact.metadata is not None
    after_exact = tokenizer.encode.call_count
    assert after_exact > initial_calls
    preparer.prepare(exact_request, environment)
    assert tokenizer.encode.call_count > after_exact
    assert exact.metadata["prompt_evidence"]["input_count_source"] == "tokenizer"

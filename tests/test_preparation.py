"""Bounded preparation, deterministic chunking and round admission contracts."""

from copy import deepcopy
import json
from typing import List

import pytest
import ray
from pydantic import ValidationError

from llmperf.preparation import (
    PreparedPrompt,
    PreparedStream,
    PromptPlan,
    calibrate,
    render_chunk,
)
from llmperf_backend.models import BenchmarkCampaignStart, SchedulerConfig
from llmperf_backend.rounds import expand_rounds, ready_round_runners
from llmperf.prompt_datasets import PromptDatasetSource


class Tokenizer:
    calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        self.calls += 1
        return [1] * len(text)


def test_calibrated_chunks():
    tokenizer = Tokenizer()
    plan = calibrate(tokenizer, 20261014, 32768, 0, 26214, 16)
    calibration_calls = tokenizer.calls
    together = render_chunk(plan, 0, 12)
    split = render_chunk(plan, 0, 3) + render_chunk(plan, 3, 9)
    assert together == split
    assert tokenizer.calls == calibration_calls
    assert len({prompt[0][0] for prompt in together}) == 12
    assert all(prompt[0][0].startswith(plan.prefix) for prompt in together)
    assert len(plan.prefix) == 26214
    assert all(len(prompt[0][0]) == 32768 for prompt in together)
    assert together[0][1]["input_count_source"] == "calibrated_estimate"


def test_sharegpt_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "indexes"))
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Question {index}: " + "diverse content " * 10,
                        }
                    ]
                }
                for index in range(1000)
            ]
        )
    )
    source = PromptDatasetSource.external("sharegpt-user", path)
    tokenizer = Tokenizer()
    plan = calibrate(tokenizer, 20261014, 1000, 0, 800, 16, source=source)
    assert tokenizer.calls == plan.calibration_records == 128
    assert plan.records == plan.lengths == ()
    assert len(plan.prefix) == 800
    calibration_calls = tokenizer.calls
    together = render_chunk(plan, 0, 64)
    assert together == render_chunk(plan, 0, 17) + render_chunk(plan, 17, 47)
    assert tokenizer.calls == calibration_calls
    assert len({prompt[0][0] for prompt in together}) == 64
    assert all(len(prompt[0][0]) == 1000 for prompt in together)
    assert all(prompt[0][0].startswith(plan.prefix) for prompt in together)
    assert all(prompt[1]["source"] == "sharegpt-user" for prompt in together)
    assert all(
        prompt[1]["prefix_record_indices"] == list(plan.prefix_record_indices)
        for prompt in together
    )
    other_plan = calibrate(tokenizer, 20261015, 1000, 0, 800, 16, source=source)
    assert other_plan.prefix != plan.prefix


def test_bounded_calibration(tmp_path, monkeypatch):
    from llmperf import preparation
    from llmperf.prompt_datasets import PromptDataset

    class Records:
        reads = 0

        def __len__(self) -> int:
            return 1_000_000

        def __getitem__(self, position: int) -> tuple[int, str]:
            self.reads += 1
            return position * 3, f"record {position} " * 1000

    records = Records()
    dataset = PromptDataset("sharegpt-user", records, "\n")
    monkeypatch.setattr(preparation, "load_prompt_dataset", lambda source: dataset)
    source = PromptDatasetSource.external("sharegpt-user", tmp_path / "large.json")
    tokenizer = Tokenizer()
    plan = calibrate(tokenizer, 1, 1000, 0, 800, 16, source=source)
    assert records.reads <= 129
    assert tokenizer.calls == 128
    rendered = render_chunk(plan, 0, 2)
    assert records.reads <= 131
    assert tokenizer.calls == 128
    assert all(
        index % 3 == 0 for item in rendered for index in item[1]["record_indices"]
    )


def test_bounded_prefetch(monkeypatch):
    outputs: dict[ray.ObjectRef, List[PreparedPrompt]] = {}
    cancelled: list[ray.ObjectRef] = []
    submitted: list[int] = []

    class Render:
        def remote(
            self, plan: PromptPlan, start: int, count: int
        ) -> ray.ObjectRef[List[PreparedPrompt]]:
            reference = ray.ObjectRef.from_random()
            outputs[reference] = render_chunk(plan, start, count)
            submitted.append(start)
            return reference

    class Actor:
        @property
        def render(self) -> Render:
            return Render()

    monkeypatch.setattr(ray, "get", lambda reference: outputs.pop(reference))
    monkeypatch.setattr(
        ray, "cancel", lambda reference, force: cancelled.append(reference)
    )
    plan = calibrate(Tokenizer(), 1, 256, 0, 128, 8)
    stream = PreparedStream(plan, [Actor(), Actor()], 100, 4, 2)
    assert submitted == [0, 4]
    observed = [stream.take() for _ in range(9)]
    assert [index for index, _ in observed] == list(range(9))
    assert [item for _, item in observed] == render_chunk(plan, 0, 9)
    assert len(outputs) <= 2
    assert max(submitted) < 100
    stream.close()
    assert len(cancelled) == 2
    with pytest.raises(StopIteration):
        stream.take()


def test_empty_preparation():
    plan = calibrate(Tokenizer(), 1, 256, 0, 128, 8)
    with pytest.raises(ValueError, match="actors"):
        PreparedStream(plan, [], 100, 4, 2)


def test_round_expansion():
    templates = [
        {"benchmark": {"dataset_seed": 1}, "label": str(index)} for index in range(8)
    ]
    original = deepcopy(templates)
    runners = expand_rounds(templates, 100, 20261014)
    assert templates == original
    assert len(runners) == 800
    assert len({runner["benchmark"]["dataset_seed"] for runner in runners}) == 800
    assert runners == expand_rounds(templates, 100, 20261014)
    assert runners != expand_rounds(templates, 100, 20261015)


def test_round_barrier():
    runners = expand_rounds([{"benchmark": {}} for _ in range(8)], 3, 1)
    for runner in runners:
        runner["campaign_id"] = "test"
    states = [
        {
            "campaign_id": "test",
            "round_index": runner["metadata"]["round"],
            "status": "queued",
        }
        for runner in runners
    ]
    states[0]["status"] = "running"
    assert ready_round_runners(runners[1:], states) == runners[1:8]
    for state in states[:8]:
        state["status"] = "succeeded"
    assert ready_round_runners(runners[8:], states) == runners[8:16]
    # Recovery uses durable queued/running state, not a volatile round cursor.
    states[8]["status"] = "running"
    assert ready_round_runners(runners[9:], states) == runners[9:16]


def test_round_validation():
    document = {
        "version": "1.0.0",
        "campaign": {"name": "rounds", "rounds": {"count": 100, "seed": 20261014}},
        "runners": [{"benchmark": {"provider": "test", "model": "test"}}] * 8,
    }
    model = BenchmarkCampaignStart.model_validate(document)
    assert model.campaign.rounds is not None
    assert model.campaign.rounds.count == 100
    document["campaign"]["rounds"]["count"] = 1001
    with pytest.raises(ValidationError):
        BenchmarkCampaignStart.model_validate(document)
    assert (
        SchedulerConfig(ray_num_cpus=16, preparation_workers=8).request_actor_capacity
        == 8
    )

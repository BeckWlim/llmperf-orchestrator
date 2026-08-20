import json

import pytest

from llmperf.prompt_datasets import (
    PromptDatasetSource,
    load_prompt_dataset,
    prepare_prompt_requests,
)


def test_builtin_pipeline():
    first = prepare_prompt_requests(
        source=PromptDatasetSource.builtin_sonnet(),
        prompt_mode="sample",
        num_requests=2,
        repeat_count=1,
        mean_input_tokens=180,
        stddev_input_tokens=0,
        mean_output_tokens=8,
        get_token_length=len,
        seed=17,
    )
    second = prepare_prompt_requests(
        source=PromptDatasetSource.builtin_sonnet(),
        prompt_mode="sample",
        num_requests=2,
        repeat_count=1,
        mean_input_tokens=180,
        stddev_input_tokens=0,
        mean_output_tokens=8,
        get_token_length=len,
        seed=17,
    )

    assert first == second
    prompts, evidence = first
    assert [length for _, length in prompts] == [180, 180]
    assert all(item["source"] == "builtin-sonnet" for item in evidence)
    assert all(item["mode"] == "concatenate" for item in evidence)


def test_candidate_cycles(tmp_path):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": character * 6},
                        {"from": "gpt", "value": "answer"},
                    ]
                }
                for character in "ABC"
            ]
        ),
        encoding="utf-8",
    )

    prompts, evidence = prepare_prompt_requests(
        source=PromptDatasetSource.external("sharegpt", path),
        prompt_mode="concatenate",
        num_requests=2,
        repeat_count=1,
        mean_input_tokens=20,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        get_token_length=len,
        seed=9,
    )
    _, different = prepare_prompt_requests(
        source=PromptDatasetSource.external("sharegpt", path),
        prompt_mode="concatenate",
        num_requests=2,
        repeat_count=1,
        mean_input_tokens=20,
        stddev_input_tokens=0,
        mean_output_tokens=1,
        get_token_length=len,
        seed=10,
    )

    first_cycle = [
        segment["record_index"]
        for item in evidence
        for segment in item["segments"]
        if segment["corpus_cycle"] == 0
    ]
    assert len(first_cycle) == len(set(first_cycle)) == 3
    assert all(length == 20 for _, length in prompts)
    assert evidence != different


def test_source_adapters(tmp_path):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "external prompt"},
                        {"from": "gpt", "value": "answer"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    builtin = load_prompt_dataset(PromptDatasetSource.builtin_sonnet())
    external = load_prompt_dataset(PromptDatasetSource.external("sharegpt", path))

    assert builtin.source == "builtin-sonnet"
    assert external.source == "sharegpt"
    assert builtin.records and external.records
    assert all(isinstance(index, int) and text for index, text in builtin.records)
    assert all(isinstance(index, int) and text for index, text in external.records)


def test_sharegpt_compatibility(tmp_path):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": "one turn"}]},
                {"conversations": [{"from": "human", "value": "   "}]},
                {"conversations": "invalid"},
                {
                    "conversations": [
                        {"from": "human", "value": "two turns"},
                        {"from": "gpt", "value": "answer"},
                    ]
                },
            ]
        ),
        encoding="utf-8-sig",
    )

    dataset = load_prompt_dataset(PromptDatasetSource.external("sharegpt", path))

    assert dataset.records == ((0, "one turn"), (3, "two turns"))


def test_source_validation():
    builtin = load_prompt_dataset(PromptDatasetSource.builtin_sonnet())

    assert builtin.source == "builtin-sonnet"
    assert isinstance(builtin.records, tuple)
    with pytest.raises(ValueError, match="Unsupported prompt dataset adapter"):
        PromptDatasetSource.external("jsonl", "dataset.jsonl")
    with pytest.raises(ValueError, match="requires a path"):
        PromptDatasetSource(adapter="text")


def test_text_adapter(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8-sig")

    dataset = load_prompt_dataset(PromptDatasetSource.external("text", path))

    assert dataset.source == "text"
    assert dataset.records == ((0, "first prompt"), (2, "second prompt"))

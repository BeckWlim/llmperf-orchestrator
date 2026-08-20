import json

from datasets import Dataset
import pytest

from llmperf import prompt_datasets as prompt_dataset_module
from llmperf.prompt_datasets import (
    HuggingFacePromptRecords,
    PromptDatasetSource,
    load_prompt_dataset,
    prepare_prompt_requests,
)


@pytest.fixture(autouse=True)
def isolated_dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets-cache"))


def dataset_records(dataset):
    return tuple(dataset.records[index] for index in range(len(dataset.records)))


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
    assert all(
        isinstance(index, int) and text for index, text in dataset_records(builtin)
    )
    assert all(
        isinstance(index, int) and text for index, text in dataset_records(external)
    )


def test_sharegpt_compatibility(tmp_path):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": "one turn"}]},
                {"conversations": [{"from": "human", "value": "   "}]},
                {"conversations": []},
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

    assert dataset_records(dataset) == ((0, "one turn"), (3, "two turns"))


def test_sharegpt_user_filter(tmp_path):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": "human prompt"}]},
                {"conversations": [{"from": "gpt", "value": "assistant text"}]},
                {"conversations": [{"from": "USER", "value": "user prompt"}]},
                {"conversations": [{"from": "system", "value": "system text"}]},
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_prompt_dataset(PromptDatasetSource.external("sharegpt-user", path))

    assert dataset.source == "sharegpt-user"
    assert dataset_records(dataset) == (
        (0, "human prompt"),
        (2, "user prompt"),
    )


def test_json_streaming(tmp_path, monkeypatch):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": "first"}]},
                {"conversations": [{"from": "human", "value": "second"}]},
            ]
        ),
        encoding="utf-8",
    )

    def reject_standard_loader(*args, **kwargs):
        raise AssertionError("top-level JSON arrays must not use the full-read loader")

    monkeypatch.setattr(prompt_dataset_module, "load_dataset", reject_standard_loader)
    first = load_prompt_dataset(PromptDatasetSource.external("sharegpt-user", path))

    assert isinstance(first.records, HuggingFacePromptRecords)
    assert dataset_records(first) == ((0, "first"), (1, "second"))
    assert first.records.dataset.cache_files[0]["filename"].endswith(".arrow")

    def reject_parser(*args, **kwargs):
        raise AssertionError("the normalized Arrow index should be reused")

    monkeypatch.setattr(prompt_dataset_module.ijson, "items", reject_parser)
    second = load_prompt_dataset(PromptDatasetSource.external("sharegpt-user", path))

    assert dataset_records(second) == dataset_records(first)
    assert not tuple((tmp_path / "datasets-cache").rglob("*.incomplete"))


def test_jsonl_streaming(tmp_path):
    path = tmp_path / "sharegpt.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"conversations": [{"from": "human", "value": "first"}]}),
                json.dumps({"conversations": [{"from": "user", "value": "second"}]}),
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_prompt_dataset(PromptDatasetSource.external("sharegpt-user", path))

    assert dataset_records(dataset) == ((0, "first"), (1, "second"))


def test_sharegpt_parquet(tmp_path):
    artifact_path = tmp_path / "artifact"
    Dataset.from_list(
        [
            {"conversations": [{"from": "human", "value": "parquet prompt"}]},
            {"conversations": [{"from": "gpt", "value": "assistant text"}]},
        ]
    ).to_parquet(str(artifact_path))

    dataset = load_prompt_dataset(
        PromptDatasetSource.external(
            "sharegpt-user", artifact_path, filename="train.parquet"
        )
    )

    assert dataset_records(dataset) == ((0, "parquet prompt"),)


def test_document_text_parquet(tmp_path):
    artifact_path = tmp_path / "fineweb-artifact"
    Dataset.from_list(
        [
            {
                "text": "First complete document.\nIt keeps its paragraphs.",
                "id": "document-1",
                "token_count": 8,
            },
            {"text": "   ", "id": "empty", "token_count": 0},
            {
                "text": "Second complete document.",
                "id": "document-2",
                "token_count": 4,
            },
        ]
    ).to_parquet(str(artifact_path))

    dataset = load_prompt_dataset(
        PromptDatasetSource.external(
            "document-text", artifact_path, filename="fineweb.parquet"
        )
    )

    assert dataset.source == "document-text"
    assert dataset_records(dataset) == (
        (0, "First complete document.\nIt keeps its paragraphs."),
        (2, "Second complete document."),
    )


def test_document_text_schema(tmp_path):
    artifact_path = tmp_path / "missing-text"
    Dataset.from_list([{"content": "wrong field"}]).to_parquet(str(artifact_path))

    with pytest.raises(ValueError, match="no usable non-empty documents"):
        load_prompt_dataset(
            PromptDatasetSource.external(
                "document-text", artifact_path, filename="documents.parquet"
            )
        )


def test_source_validation():
    builtin = load_prompt_dataset(PromptDatasetSource.builtin_sonnet())

    assert builtin.source == "builtin-sonnet"
    assert isinstance(builtin.records, tuple)
    with pytest.raises(ValueError, match="Unsupported prompt dataset adapter"):
        PromptDatasetSource.external("jsonl", "dataset.jsonl")
    with pytest.raises(ValueError, match="requires a path"):
        PromptDatasetSource(adapter="text")
    with pytest.raises(ValueError, match="supported formats: parquet, arrow"):
        load_prompt_dataset(
            PromptDatasetSource.external(
                "document-text", "documents.txt", filename="documents.txt"
            )
        )


def test_text_adapter(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8-sig")

    dataset = load_prompt_dataset(PromptDatasetSource.external("text", path))

    assert dataset.source == "text"
    assert dataset_records(dataset) == ((0, "first prompt"), (2, "second prompt"))

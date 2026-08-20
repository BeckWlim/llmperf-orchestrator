import hashlib
from unittest.mock import Mock

import pytest

from llmperf_backend.artifacts import (
    ArtifactCaches,
    ArtifactValidationError,
    DatasetResolution,
    validate_dataset_artifact,
    validate_tokenizer_artifact,
)
from llmperf_backend.outbound import OutboundPolicy
from llmperf_backend.artifacts import TokenizerResolution


REVISION = "a" * 40


@pytest.fixture(autouse=True)
def isolated_dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets-cache"))


def test_shared_policy(tmp_path, monkeypatch):
    policy = OutboundPolicy(
        source="explicit",
        proxy_url="http://proxy.internal:3128",
        standard_proxy_names=("HTTPS_PROXY",),
        no_proxy_names=(),
    )
    configure = Mock(return_value=policy)
    monkeypatch.setattr(
        "llmperf_backend.artifacts.configure_outbound_transport", configure
    )
    monkeypatch.setenv("LLMPERF_DATASET_CACHE", str(tmp_path / "datasets"))
    monkeypatch.setenv("LLMPERF_TOKENIZER_CACHE", str(tmp_path / "tokenizers"))

    caches = ArtifactCaches.from_environment()

    assert caches.dataset.outbound_policy is policy
    assert caches.tokenizer.outbound_policy is policy
    configure.assert_called_once()


def test_dataset_integrity(tmp_path):
    dataset_path = tmp_path / "sharegpt.json"
    content = b'[{"conversations": [{"from": "human", "value": "prompt"}]}]'
    dataset_path.write_bytes(content)
    resolution = DatasetResolution(
        source="huggingface",
        dataset_id="organization/sharegpt",
        filename="sharegpt.json",
        revision=REVISION,
        adapter="sharegpt",
        path=dataset_path,
        cached=True,
    )

    evidence = validate_dataset_artifact(resolution)

    assert evidence.file_count == 1
    assert evidence.size_bytes == len(content)
    assert evidence.sha256 == hashlib.sha256(content).hexdigest()
    assert evidence.immutable_revision is True
    assert evidence.cache_hit is True
    assert evidence.adapter == "sharegpt"
    assert evidence.record_count == 1
    assert "path" not in evidence.public_dict()


def test_incomplete_rejected(tmp_path):
    dataset_path = tmp_path / "sharegpt.json.incomplete"
    dataset_path.write_text("[]", encoding="utf-8")
    resolution = DatasetResolution(
        source="huggingface",
        dataset_id="organization/sharegpt",
        filename="sharegpt.json",
        revision=REVISION,
        adapter="sharegpt",
        path=dataset_path,
        cached=False,
    )

    with pytest.raises(ArtifactValidationError, match="still incomplete"):
        validate_dataset_artifact(resolution)


def test_tokenizer_integrity(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    tokenizer_path.joinpath("tokenizer.json").write_text(
        '{"version":"1.0"}', encoding="utf-8"
    )
    tokenizer_path.joinpath("config.json").write_text("{}", encoding="utf-8")
    resolution = TokenizerResolution(
        source="huggingface",
        tokenizer_id="organization/tokenizer",
        revision=REVISION,
        use_fast=True,
        path=tokenizer_path,
        cached=False,
    )

    evidence = validate_tokenizer_artifact(resolution)

    assert evidence.file_count == 2
    assert evidence.size_bytes > 0
    assert len(evidence.sha256) == 64
    assert evidence.cache_hit is False


def test_empty_tokenizer_rejected(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    resolution = TokenizerResolution(
        source="huggingface",
        tokenizer_id="organization/tokenizer",
        revision=REVISION,
        use_fast=True,
        path=tokenizer_path,
        cached=True,
    )

    with pytest.raises(ArtifactValidationError, match="contains no files"):
        validate_tokenizer_artifact(resolution)

import asyncio
from pathlib import Path
from unittest.mock import Mock

from llmperf_backend.tokenizers import TokenizerCache, TokenizerResolution
from llmperf_backend.tokenizers import TokenizerResolutionError
import pytest


class FakeTokenizer:
    _commit_hash = "resolved-commit"

    def save_pretrained(self, directory):
        Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")


def test_immutable_revision(tmp_path):
    resolution = TokenizerResolution(
        source="huggingface",
        tokenizer_id="organization/tokenizer",
        revision="a" * 40,
        use_fast=True,
        path=tmp_path,
        cached=True,
    )

    spec = resolution.benchmark_spec(
        selection="explicit",
        accuracy="compatible",
        requested_revision="main",
    )

    assert spec["immutable_revision"] is True
    assert spec["revision"] == "a" * 40
    assert spec["requested_revision"] == "main"


def test_cached_commit(tmp_path, monkeypatch):
    commit = "e" * 40
    cached_file = tmp_path / "snapshots" / commit / "tokenizer_config.json"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: str(cached_file),
    )
    cache = TokenizerCache(cache_directory=tmp_path, proxy_url="")

    resolved = cache._resolved_revision(None, "deepseek-ai/DeepSeek-V3", "main")

    assert resolved == commit


def test_legacy_artifact(tmp_path, monkeypatch):
    commit = "e" * 40
    cached_file = tmp_path / "snapshots" / commit / "tokenizer_config.json"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: str(cached_file),
    )
    cache = TokenizerCache(cache_directory=tmp_path, proxy_url="")
    legacy_target = cache.resolved_directory / cache._artifact_key(
        "deepseek-ai/DeepSeek-V3", "main", True
    )
    legacy_target.mkdir(parents=True)

    resolution = cache._resolve_sync("deepseek-ai/DeepSeek-V3", "main", True)

    assert resolution.revision == commit
    assert resolution.path == legacy_target
    assert resolution.benchmark_spec()["immutable_revision"] is True


def test_cache(tmp_path, monkeypatch):
    loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(
        cache_directory=tmp_path, local_files_only=False, proxy_url=""
    )
    spec = {
        "source": "huggingface",
        "id": "organization/model-tokenizer",
        "revision": "release",
        "use_fast": True,
    }

    first = cache._resolve_sync("organization/model-tokenizer", "release", True)
    second = cache._resolve_sync(first.tokenizer_id, first.revision, first.use_fast)

    assert first.revision == "resolved-commit"
    assert first.cached is False
    assert first.path.is_dir()
    assert (first.path / "tokenizer.json").is_file()
    assert second.path == first.path
    assert second.cached is True
    loader.assert_called_once_with(
        "organization/model-tokenizer",
        revision="release",
        use_fast=True,
        trust_remote_code=False,
        cache_dir=str(tmp_path / "downloads"),
        local_files_only=False,
    )


def test_offline(tmp_path, monkeypatch):
    loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=True)

    cache._resolve_sync("cached/tokenizer", "main", False)

    assert loader.call_args.kwargs["local_files_only"] is True
    assert loader.call_args.kwargs["use_fast"] is False


def test_backend_compatibility(tmp_path, monkeypatch):
    auto_loader = Mock(
        side_effect=ValueError(
            "Tokenizer class TokenizersBackend does not exist or is not currently "
            "imported."
        )
    )
    fast_loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", auto_loader
    )
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.PreTrainedTokenizerFast.from_pretrained",
        fast_loader,
    )
    cache = TokenizerCache(
        cache_directory=tmp_path, local_files_only=False, proxy_url=""
    )

    result = cache._resolve_sync("zai-org/GLM-5.2", "main", True)

    assert result.revision == "resolved-commit"
    assert (result.path / "tokenizer.json").is_file()
    fast_loader.assert_called_once_with(
        "zai-org/GLM-5.2",
        revision="main",
        trust_remote_code=False,
        cache_dir=str(tmp_path / "downloads"),
        local_files_only=False,
        extra_special_tokens={},
    )


def test_proxy(tmp_path, monkeypatch):
    loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(
        cache_directory=tmp_path,
        local_files_only=False,
        proxy_url="http://proxy.internal:3128",
    )

    cache._resolve_sync("organization/tokenizer", "main", True)

    assert loader.call_args.kwargs["proxies"] == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }


def test_shared_proxy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPERF_HUGGINGFACE_PROXY", "http://proxy.environment:8080")

    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=False)

    assert cache.proxy_url == "http://proxy.environment:8080"


def test_bad_proxy(tmp_path):
    with pytest.raises(
        TokenizerResolutionError, match="must be an HTTP\\(S\\) proxy URL"
    ):
        TokenizerCache(cache_directory=tmp_path, proxy_url="proxy.internal:3128")


def test_bad_path(tmp_path):
    cache = TokenizerCache(cache_directory=tmp_path)

    with pytest.raises(TokenizerResolutionError, match="repository ID"):
        asyncio.run(cache.resolve({"id": "/etc/tokenizer"}))

import asyncio
from pathlib import Path
from unittest.mock import Mock

from llmperf_backend.tokenizers import TokenizerCache
from llmperf_backend.tokenizers import TokenizerResolutionError
import pytest


class FakeTokenizer:
    _commit_hash = "resolved-commit"

    def save_pretrained(self, directory):
        Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")


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

    first = asyncio.run(cache.resolve(spec))
    second = asyncio.run(cache.resolve(first.benchmark_spec()))

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

    asyncio.run(cache.resolve({"id": "cached/tokenizer", "use_fast": False}))

    assert loader.call_args.kwargs["local_files_only"] is True
    assert loader.call_args.kwargs["use_fast"] is False


def test_tokenizers_backend_compatibility(tmp_path, monkeypatch):
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
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=False)

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

    asyncio.run(cache.resolve({"id": "organization/tokenizer"}))

    assert loader.call_args.kwargs["proxies"] == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }


def test_shared_huggingface_proxy_environment(tmp_path, monkeypatch):
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

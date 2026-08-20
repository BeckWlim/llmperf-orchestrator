import asyncio
from pathlib import Path
from unittest.mock import Mock

from llmperf_backend.tokenizers import (
    TOKENIZER_SNAPSHOT_ALLOW_PATTERNS,
    TokenizerCache,
    TokenizerResolution,
)
from llmperf_backend.tokenizers import TokenizerResolutionError
import pytest

SNAPSHOT_COMMIT = "d" * 40


class FakeTokenizer:
    _commit_hash = "resolved-commit"

    def save_pretrained(self, directory):
        Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")


def _snapshot(tmp_path, commit=SNAPSHOT_COMMIT):
    path = tmp_path / "downloads" / "models--test" / "snapshots" / commit
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return path


def _mock_download(tmp_path, monkeypatch, commit=SNAPSHOT_COMMIT):
    snapshot = _snapshot(tmp_path, commit)
    downloader = Mock(return_value=str(snapshot))
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    return snapshot, downloader


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


def test_cache(tmp_path, monkeypatch):
    snapshot, downloader = _mock_download(tmp_path, monkeypatch)
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

    assert first.revision == SNAPSHOT_COMMIT
    assert first.cached is False
    assert first.path.is_dir()
    assert (first.path / "tokenizer.json").is_file()
    assert second.path == first.path
    assert second.cached is True
    loader.assert_called_once_with(
        str(snapshot),
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    downloader.assert_called_once_with(
        repo_id="organization/model-tokenizer",
        revision="release",
        cache_dir=str(tmp_path / "downloads"),
        local_files_only=False,
        allow_patterns=list(TOKENIZER_SNAPSHOT_ALLOW_PATTERNS),
    )


def test_offline(tmp_path, monkeypatch):
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    loader = Mock(side_effect=AssertionError("unexpected tokenizer load"))
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=True)

    with pytest.raises(TokenizerResolutionError, match="not present in local cache"):
        cache._resolve_sync("cached/tokenizer", "main", False)

    loader.assert_not_called()
    downloader.assert_not_called()


def test_offline_hit(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: str(snapshot / "tokenizer_config.json"),
    )
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=True)

    result = cache._resolve_sync("cached/tokenizer", "main", False)

    assert result.revision == SNAPSHOT_COMMIT
    loader.assert_called_once_with(
        str(snapshot),
        use_fast=False,
        trust_remote_code=False,
        local_files_only=True,
    )
    downloader.assert_not_called()


def test_offline_fallback(tmp_path, monkeypatch):
    snapshot = (
        tmp_path
        / "downloads"
        / "models--cached--tokenizer"
        / "snapshots"
        / SNAPSHOT_COMMIT
    )
    snapshot.mkdir(parents=True)
    # A copied Hugging Face cache can retain the snapshot revision directory
    # while losing the relative blob target. The resolved artifact remains
    # self-contained and must still be directly usable offline.
    (snapshot / "tokenizer.json").symlink_to("../../blobs/missing")
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    loader = Mock(side_effect=AssertionError("unexpected tokenizer load"))
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=True)
    target = cache.resolved_directory / cache._artifact_key(
        "cached/tokenizer", SNAPSHOT_COMMIT, True
    )
    target.mkdir(parents=True)
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")

    result = cache._resolve_sync("cached/tokenizer", "main", True)

    assert result.cached is True
    assert result.revision == SNAPSHOT_COMMIT
    assert result.path == target
    loader.assert_not_called()
    downloader.assert_not_called()


def test_offline_snapshot(tmp_path, monkeypatch):
    snapshot = (
        tmp_path
        / "downloads"
        / "models--cached--tokenizer"
        / "snapshots"
        / SNAPSHOT_COMMIT
    )
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: None,
    )
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=True)

    result = cache._resolve_sync("cached/tokenizer", "main", True)

    assert result.revision == SNAPSHOT_COMMIT
    loader.assert_called_once_with(
        str(snapshot),
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    downloader.assert_not_called()


def test_offline_ambiguity(tmp_path, monkeypatch):
    repository = tmp_path / "downloads" / "models--cached--tokenizer" / "snapshots"
    for revision in ("a" * 40, "b" * 40):
        snapshot = repository / revision
        snapshot.mkdir(parents=True)
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: None,
    )
    cache = TokenizerCache(cache_directory=tmp_path, local_files_only=True)

    with pytest.raises(TokenizerResolutionError, match="multiple usable local"):
        cache._resolve_sync("cached/tokenizer", "main", True)


def test_backend_compatibility(tmp_path, monkeypatch):
    snapshot, _ = _mock_download(tmp_path, monkeypatch)
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

    assert result.revision == SNAPSHOT_COMMIT
    assert (result.path / "tokenizer.json").is_file()
    fast_loader.assert_called_once_with(
        str(snapshot),
        trust_remote_code=False,
        local_files_only=True,
        extra_special_tokens={},
    )


def test_proxy(tmp_path, monkeypatch):
    _, downloader = _mock_download(tmp_path, monkeypatch)
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

    assert cache.proxy_url == "http://proxy.internal:3128"
    assert "proxies" not in downloader.call_args.kwargs
    assert "proxies" not in loader.call_args.kwargs


def test_snapshot_local(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    downloader = Mock(side_effect=AssertionError("unexpected Hub download"))
    loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: str(snapshot / "tokenizer_config.json"),
    )
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(
        cache_directory=tmp_path, local_files_only=False, proxy_url=""
    )

    result = cache._resolve_sync("XiaomiMiMo/MiMo-V2.5", "main", True)

    assert result.revision == SNAPSHOT_COMMIT
    loader.assert_called_once_with(
        str(snapshot),
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    downloader.assert_not_called()


def test_snapshot_refresh(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    loader = Mock(side_effect=[OSError("missing merges"), FakeTokenizer()])
    downloader = Mock(return_value=str(snapshot))
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.try_to_load_from_cache",
        lambda *args, **kwargs: str(snapshot / "tokenizer_config.json"),
    )
    monkeypatch.setattr("llmperf_backend.tokenizers.snapshot_download", downloader)
    monkeypatch.setattr(
        "llmperf_backend.tokenizers.AutoTokenizer.from_pretrained", loader
    )
    cache = TokenizerCache(
        cache_directory=tmp_path, local_files_only=False, proxy_url=""
    )

    result = cache._resolve_sync("organization/tokenizer", "main", True)

    assert result.revision == SNAPSHOT_COMMIT
    assert loader.call_count == 2
    downloader.assert_called_once()


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

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from llmperf_backend.datasets import DatasetCache, DatasetResolutionError


DATASET_SPEC = {
    "source": "huggingface",
    "id": "organization/sharegpt",
    "filename": "data/sharegpt.json",
    "revision": "release",
    "format": "sharegpt",
}


def test_cache(tmp_path, monkeypatch):
    artifact = (
        tmp_path
        / "datasets--organization--sharegpt"
        / "snapshots"
        / "resolved-commit"
        / "data"
        / "sharegpt.json"
    )

    def download(**kwargs):
        artifact.parent.mkdir(parents=True)
        artifact.write_text("[]", encoding="utf-8")
        return str(artifact)

    loader = Mock(side_effect=download)
    monkeypatch.setattr("llmperf_backend.datasets.hf_hub_download", loader)
    monkeypatch.setattr(
        "llmperf_backend.datasets.try_to_load_from_cache", Mock(return_value=None)
    )
    cache = DatasetCache(cache_directory=tmp_path, proxy_url="")

    first = cache._resolve_sync(
        DATASET_SPEC["id"],
        DATASET_SPEC["filename"],
        DATASET_SPEC["revision"],
        DATASET_SPEC["format"],
    )
    resolved_spec = first.benchmark_spec()
    second = cache._resolve_sync(
        resolved_spec["id"],
        resolved_spec["filename"],
        resolved_spec["revision"],
        resolved_spec["format"],
    )

    assert first.revision == "resolved-commit"
    assert first.cached is False
    assert first.path == artifact
    assert second.path == first.path
    assert second.cached is True
    loader.assert_called_once_with(
        repo_id="organization/sharegpt",
        filename="data/sharegpt.json",
        repo_type="dataset",
        revision="release",
        cache_dir=tmp_path,
        proxies=None,
    )


def test_environment_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPERF_DATASET_CACHE", str(tmp_path))

    assert DatasetCache().cache_directory == tmp_path.resolve()


def test_offline_cache(tmp_path, monkeypatch):
    artifact = (
        tmp_path
        / "datasets--organization--sharegpt"
        / "snapshots"
        / "cached-commit"
        / "data"
        / "sharegpt.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("[]", encoding="utf-8")
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    monkeypatch.setattr(
        "llmperf_backend.datasets.try_to_load_from_cache",
        Mock(return_value=str(artifact)),
    )
    monkeypatch.setattr("llmperf_backend.datasets.hf_hub_download", downloader)
    cache = DatasetCache(
        cache_directory=tmp_path,
        local_files_only=True,
        proxy_url="",
    )

    result = cache._resolve_sync(
        DATASET_SPEC["id"],
        DATASET_SPEC["filename"],
        DATASET_SPEC["revision"],
        DATASET_SPEC["format"],
    )

    assert result.cached is True
    assert result.revision == "cached-commit"
    assert result.path == artifact
    downloader.assert_not_called()


def test_offline_miss(tmp_path, monkeypatch):
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    monkeypatch.setattr(
        "llmperf_backend.datasets.try_to_load_from_cache", Mock(return_value=None)
    )
    monkeypatch.setattr("llmperf_backend.datasets.hf_hub_download", downloader)
    cache = DatasetCache(
        cache_directory=tmp_path,
        local_files_only=True,
        proxy_url="",
    )

    with pytest.raises(DatasetResolutionError, match="not present in local cache"):
        cache._resolve_sync(
            DATASET_SPEC["id"],
            DATASET_SPEC["filename"],
            DATASET_SPEC["revision"],
            DATASET_SPEC["format"],
        )

    downloader.assert_not_called()


def test_offline_fallback(tmp_path, monkeypatch):
    artifact = (
        tmp_path
        / "datasets--organization--sharegpt"
        / "snapshots"
        / "cached-commit"
        / "data"
        / "sharegpt.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("[]", encoding="utf-8")
    downloader = Mock(side_effect=AssertionError("unexpected Hub lookup"))
    monkeypatch.setattr(
        "llmperf_backend.datasets.try_to_load_from_cache", Mock(return_value=None)
    )
    monkeypatch.setattr("llmperf_backend.datasets.hf_hub_download", downloader)
    cache = DatasetCache(
        cache_directory=tmp_path,
        local_files_only=True,
        proxy_url="",
    )

    result = cache._resolve_sync(
        DATASET_SPEC["id"],
        DATASET_SPEC["filename"],
        DATASET_SPEC["revision"],
        DATASET_SPEC["format"],
    )

    assert result.cached is True
    assert result.revision == "cached-commit"
    assert result.path == artifact
    downloader.assert_not_called()


def test_offline_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPERF_DATASET_OFFLINE", "true")
    assert DatasetCache(cache_directory=tmp_path).local_files_only is True

    monkeypatch.setenv("LLMPERF_DATASET_OFFLINE", "invalid")
    with pytest.raises(DatasetResolutionError, match="LLMPERF_DATASET_OFFLINE"):
        DatasetCache(cache_directory=tmp_path)


def test_shared_huggingface_proxy(tmp_path, monkeypatch):
    artifact = tmp_path / "snapshots" / "commit" / "sharegpt.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("[]", encoding="utf-8")
    loader = Mock(return_value=str(artifact))
    monkeypatch.setattr("llmperf_backend.datasets.hf_hub_download", loader)
    monkeypatch.setattr(
        "llmperf_backend.datasets.try_to_load_from_cache", Mock(return_value=None)
    )
    cache = DatasetCache(
        cache_directory=tmp_path,
        proxy_url="http://proxy.internal:3128",
    )

    cache._resolve_sync(
        DATASET_SPEC["id"],
        DATASET_SPEC["filename"],
        DATASET_SPEC["revision"],
        DATASET_SPEC["format"],
    )

    assert loader.call_args.kwargs["proxies"] == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }


def test_shared_proxy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPERF_HUGGINGFACE_PROXY", "http://proxy.environment:8080")

    cache = DatasetCache(cache_directory=tmp_path)

    assert cache.proxy_url == "http://proxy.environment:8080"


@pytest.mark.parametrize("filename", ["/etc/passwd", "../outside.json", "a\\b"])
def test_bad_filename(tmp_path, filename):
    cache = DatasetCache(cache_directory=tmp_path, proxy_url="")

    with pytest.raises(DatasetResolutionError, match="relative"):
        asyncio.run(cache.resolve({**DATASET_SPEC, "filename": filename}))


def test_bad_repository(tmp_path):
    cache = DatasetCache(cache_directory=tmp_path, proxy_url="")

    with pytest.raises(DatasetResolutionError, match="repository ID"):
        asyncio.run(cache.resolve({**DATASET_SPEC, "id": "/etc/dataset"}))

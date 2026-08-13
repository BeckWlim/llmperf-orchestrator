"""Server-owned Hugging Face dataset resolution and caching."""

import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path, PurePosixPath
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import HFValidationError, validate_repo_id

from llmperf_backend.huggingface import (
    HUGGINGFACE_PROXY,
    HuggingFaceProxyError,
    huggingface_proxy_label,
    resolve_huggingface_proxy,
)


DATASET_CACHE_DIRECTORY = "LLMPERF_DATASET_CACHE"
WORKER_DATASET_PATH = "LLMPERF_DATASET_PATH"
DEFAULT_DATASET_CACHE_DIRECTORY = Path("~/.cache/llmperf/datasets")
LOGGER = logging.getLogger(__name__)


class DatasetResolutionError(RuntimeError):
    """Raised when a submitted dataset cannot be resolved by the backend."""


@dataclass(frozen=True)
class DatasetResolution:
    source: str
    dataset_id: str
    filename: str
    revision: str
    format: str
    path: Path
    cached: bool

    def benchmark_spec(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "id": self.dataset_id,
            "filename": self.filename,
            "revision": self.revision,
            "format": self.format,
        }


class DatasetCache:
    """Resolve dataset specifications into backend-owned local files."""

    def __init__(
        self,
        cache_directory: Optional[Path] = None,
        proxy_url: Optional[str] = None,
    ):
        configured_directory = os.environ.get(DATASET_CACHE_DIRECTORY)
        selected_directory = (
            Path(configured_directory)
            if configured_directory
            else cache_directory or DEFAULT_DATASET_CACHE_DIRECTORY
        )
        self.cache_directory = selected_directory.expanduser().resolve()
        try:
            self.proxy_url = resolve_huggingface_proxy(proxy_url)
        except HuggingFaceProxyError as exc:
            raise DatasetResolutionError(str(exc)) from exc
        self._entries: Dict[Tuple[str, str, str, str], DatasetResolution] = {}
        self._lock = threading.RLock()
        LOGGER.info(
            "Dataset cache ready: directory=%s proxy=%s",
            self.cache_directory,
            huggingface_proxy_label(self.proxy_url),
        )

    async def resolve(self, spec: Mapping[str, Any]) -> DatasetResolution:
        source = str(spec.get("source", "huggingface"))
        if source != "huggingface":
            raise DatasetResolutionError("Only Hugging Face datasets are supported")
        dataset_id = str(spec.get("id", "")).strip()
        try:
            validate_repo_id(dataset_id)
        except HFValidationError as exc:
            raise DatasetResolutionError(
                f"Dataset id must be a Hugging Face repository ID: {exc}"
            ) from exc
        filename = self._validate_filename(str(spec.get("filename", "")))
        revision = str(spec.get("revision") or "main").strip()
        if not revision:
            raise DatasetResolutionError("Dataset revision must not be empty")
        dataset_format = str(spec.get("format", "sharegpt"))
        if dataset_format != "sharegpt":
            raise DatasetResolutionError(
                f"Unsupported dataset format: {dataset_format}"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._resolve_sync,
            dataset_id,
            filename,
            revision,
            dataset_format,
        )

    def _resolve_sync(
        self,
        dataset_id: str,
        filename: str,
        revision: str,
        dataset_format: str,
    ) -> DatasetResolution:
        key = (dataset_id, filename, revision, dataset_format)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and existing.path.is_file():
                return DatasetResolution(**{**existing.__dict__, "cached": True})

        cached_path = try_to_load_from_cache(
            dataset_id,
            filename,
            cache_dir=self.cache_directory,
            revision=revision,
            repo_type="dataset",
        )
        try:
            resolved_path = Path(
                hf_hub_download(
                    repo_id=dataset_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=self.cache_directory,
                    proxies=(
                        {"http": self.proxy_url, "https": self.proxy_url}
                        if self.proxy_url
                        else None
                    ),
                )
            ).resolve()
        except Exception as exc:
            raise DatasetResolutionError(
                f"Unable to resolve dataset {dataset_id!r}, file {filename!r}, "
                f"at revision {revision!r}: {exc}. Check Hugging Face network "
                f"settings, {HUGGINGFACE_PROXY}, or preload "
                f"{self.cache_directory}."
            ) from exc

        resolution = DatasetResolution(
            source="huggingface",
            dataset_id=dataset_id,
            filename=filename,
            revision=self._resolved_revision(resolved_path, revision),
            format=dataset_format,
            path=resolved_path,
            cached=isinstance(cached_path, str),
        )
        with self._lock:
            self._entries[key] = resolution
            self._entries[
                (dataset_id, filename, resolution.revision, dataset_format)
            ] = resolution
        return resolution

    @staticmethod
    def _validate_filename(filename: str) -> str:
        normalized = filename.strip()
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in normalized
        ):
            raise DatasetResolutionError(
                "Dataset filename must be a relative Hugging Face artifact path"
            )
        return normalized

    @staticmethod
    def _resolved_revision(path: Path, requested_revision: str) -> str:
        parts = path.parts
        try:
            snapshot_index = parts.index("snapshots")
            return parts[snapshot_index + 1]
        except (ValueError, IndexError):
            return requested_revision

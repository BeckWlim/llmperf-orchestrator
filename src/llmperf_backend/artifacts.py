"""Backend-owned artifact contracts, dataset materialization, and validation."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

from llmperf.prompt_datasets import (
    PromptDatasetSource,
    load_prompt_dataset,
    validate_external_dataset_adapter,
)
from llmperf.logging import route_library_logs
from llmperf_backend.outbound import (
    LLMPERF_PROXY,
    OutboundConfigurationError,
    OutboundPolicy,
    configure_outbound_transport,
    xet_transport_label,
)

# This Backend intentionally uses Transformers without a model framework. Suppress
# its import advisory and all Hub terminal bars before importing either library.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
    try_to_load_from_cache,
)
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, PreTrainedTokenizerFast

route_library_logs("huggingface_hub", "transformers")


DATASET_CACHE_DIRECTORY = "LLMPERF_DATASET_CACHE"
DATASET_OFFLINE = "LLMPERF_DATASET_OFFLINE"
DEFAULT_DATASET_CACHE_DIRECTORY = Path("~/.cache/llmperf/datasets")
IMMUTABLE_HUGGINGFACE_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
HUGGINGFACE_REPOSITORY_SEGMENT = re.compile(r"\A[\w](?:[\w.-]*[\w])?\Z")
READ_CHUNK_BYTES = 1024 * 1024
ArtifactKind = Literal["dataset", "tokenizer"]
LOGGER = logging.getLogger(__name__)


# Shared artifact contract and Hugging Face identity rules.


class ArtifactResolutionError(RuntimeError):
    """Raised when a Backend-owned artifact cannot be resolved."""


class ArtifactValidationError(RuntimeError):
    """Raised when a resolved cache artifact is absent, partial, or unstable."""


class HuggingFaceRepositoryError(ValueError):
    """Raised when an artifact repository ID is not a valid Hub identifier."""


class DatasetResolutionError(ArtifactResolutionError):
    """Raised when a submitted dataset cannot be resolved by the Backend."""


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Common identity and local materialization state for one artifact."""

    kind: ArtifactKind
    repository_id: str
    revision: str
    path: Path
    cached: bool
    filename: Optional[str] = None
    adapter: Optional[str] = None


@dataclass(frozen=True)
class ArtifactDownloadProgress:
    """One path-free byte-progress event emitted by a Hub transfer."""

    kind: ArtifactKind
    repository_id: str
    filename: Optional[str]
    phase: str
    completed_bytes: int
    total_bytes: Optional[int]

    def public_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "repository_id": self.repository_id,
            "filename": self.filename,
            "phase": self.phase,
            "completed_bytes": self.completed_bytes,
            "total_bytes": self.total_bytes,
        }


ArtifactProgressCallback = Callable[[ArtifactDownloadProgress], None]
_ARTIFACT_PROGRESS_CALLBACK: ContextVar[Optional[ArtifactProgressCallback]] = (
    ContextVar("llmperf_artifact_progress_callback", default=None)
)


@contextmanager
def artifact_progress_scope(
    callback: ArtifactProgressCallback,
) -> Iterator[None]:
    """Attach one request-scoped progress callback to resolver executor work."""

    callback_token = _ARTIFACT_PROGRESS_CALLBACK.set(callback)
    try:
        yield
    finally:
        _ARTIFACT_PROGRESS_CALLBACK.reset(callback_token)


def _hub_byte_reporter_class(
    callback: ArtifactProgressCallback,
    kind: ArtifactKind,
    repository_id: str,
    filename: Optional[str],
) -> type[tqdm_asyncio]:
    """Adapt the Hub transfer hook into non-rendering absolute byte events."""

    progress_lock = threading.Lock()
    progress_state: Dict[int, Tuple[str, int, Optional[int]]] = {}

    class HubByteReporter(tqdm_asyncio):
        def __init__(self, *args: Any, **kwargs: Any):
            progress_options = dict(kwargs)
            progress_options["disable"] = True
            description = str(progress_options.get("desc") or "").lower()
            self._phase = "reconstruct" if "reconstruct" in description else "download"
            self._reports_bytes = progress_options.get("unit") == "B"
            super().__init__(*args, **progress_options)
            self._progress_id = id(self)
            self._last_reported_bytes = -1
            self._emit_progress(force=True)

        def _emit_progress(self, *, force: bool = False) -> None:
            if not self._reports_bytes:
                return
            completed_bytes = max(0, int(self.n))
            total_bytes = int(self.total) if self.total is not None else None
            if not force and completed_bytes == self._last_reported_bytes:
                return
            with progress_lock:
                progress_state[self._progress_id] = (
                    self._phase,
                    completed_bytes,
                    total_bytes,
                )
                phase_progress = tuple(
                    state
                    for state in progress_state.values()
                    if state[0] == self._phase
                )
                aggregate_completed_bytes = sum(state[1] for state in phase_progress)
                known_total_bytes = tuple(
                    state[2] for state in phase_progress if state[2] is not None
                )
                aggregate_total_bytes = (
                    sum(known_total_bytes)
                    if len(known_total_bytes) == len(phase_progress)
                    else None
                )
            callback(
                ArtifactDownloadProgress(
                    kind=kind,
                    repository_id=repository_id,
                    filename=filename,
                    phase=self._phase,
                    completed_bytes=aggregate_completed_bytes,
                    total_bytes=aggregate_total_bytes,
                )
            )
            self._last_reported_bytes = completed_bytes

        def update(self, n: Optional[float] = 1) -> Optional[bool]:
            increment = 1 if n is None else n
            self.n += increment
            self._emit_progress()
            return None

        def close(self) -> None:
            self._emit_progress(force=True)
            super().close()

    return HubByteReporter


class ArtifactResolution(Protocol):
    """Capability required by common integrity validation."""

    def artifact_descriptor(self) -> ArtifactDescriptor: ...


class DatasetResolver(Protocol):
    """Capability required by request orchestration to resolve a dataset."""

    async def resolve(self, spec: Mapping[str, Any]) -> "DatasetResolution": ...


def environment_flag(
    environment: Mapping[str, str], name: str, default: bool = False
) -> bool:
    """Parse one shared boolean environment setting."""

    raw_value = environment.get(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise ArtifactResolutionError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0"
    )


def validate_huggingface_repository_id(repository_id: str) -> str:
    """Validate the stable public repository-ID grammar used by Hub APIs."""

    segments = repository_id.split("/")
    if len(segments) not in {1, 2} or any(
        not segment or len(segment) > 96 for segment in segments
    ):
        raise HuggingFaceRepositoryError(
            "repository ID must be 'name' or 'namespace/name' with segments up to "
            "96 characters"
        )
    if any(
        HUGGINGFACE_REPOSITORY_SEGMENT.fullmatch(segment) is None
        for segment in segments
    ):
        raise HuggingFaceRepositoryError(
            "repository ID segments must use letters, numbers, '_', '-' or '.', "
            "and cannot start or end with '-' or '.'"
        )
    if "--" in repository_id or ".." in repository_id:
        raise HuggingFaceRepositoryError("repository ID must not contain '--' or '..'")
    if repository_id.endswith(".git"):
        raise HuggingFaceRepositoryError("repository ID must not end with '.git'")
    return repository_id


def is_immutable_huggingface_revision(value: object) -> bool:
    """Return whether a Hub revision is content-addressed."""

    return isinstance(value, str) and bool(
        IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(value)
    )


def snapshot_revision(path: Path, requested_revision: str) -> str:
    """Extract the immutable revision from a standard Hub snapshot path."""

    path_parts = path.parts
    try:
        snapshot_index = path_parts.index("snapshots")
        return path_parts[snapshot_index + 1]
    except (ValueError, IndexError):
        return requested_revision


@dataclass(frozen=True)
class DatasetResolution:
    source: str
    dataset_id: str
    filename: str
    revision: str
    adapter: str
    path: Path
    cached: bool

    def benchmark_spec(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "id": self.dataset_id,
            "filename": self.filename,
            "revision": self.revision,
            "adapter": self.adapter,
        }

    def artifact_descriptor(self) -> ArtifactDescriptor:
        return ArtifactDescriptor(
            kind="dataset",
            repository_id=self.dataset_id,
            revision=self.revision,
            path=self.path,
            cached=self.cached,
            filename=self.filename,
            adapter=self.adapter,
        )


# Dataset materialization.


class DatasetCache:
    """Resolve dataset specifications into Backend-owned local files."""

    def __init__(
        self,
        cache_directory: Optional[Path] = None,
        local_files_only: Optional[bool] = None,
        proxy_url: Optional[str] = None,
        outbound_policy: Optional[OutboundPolicy] = None,
    ):
        configured_directory = os.environ.get(DATASET_CACHE_DIRECTORY)
        selected_directory = (
            Path(configured_directory)
            if configured_directory
            else cache_directory or DEFAULT_DATASET_CACHE_DIRECTORY
        )
        self.cache_directory = selected_directory.expanduser().resolve()
        try:
            configured_offline = environment_flag(os.environ, DATASET_OFFLINE)
        except ArtifactResolutionError as exc:
            raise DatasetResolutionError(str(exc)) from exc
        self.local_files_only = (
            configured_offline if local_files_only is None else local_files_only
        )
        if outbound_policy is not None and proxy_url is not None:
            raise DatasetResolutionError(
                "outbound_policy and proxy_url cannot both be provided"
            )
        try:
            selected_policy = outbound_policy or configure_outbound_transport(
                os.environ, proxy_url
            )
        except OutboundConfigurationError as exc:
            raise DatasetResolutionError(str(exc)) from exc
        self.outbound_policy = selected_policy
        self.proxy_url = selected_policy.proxy_url
        self._entries: Dict[Tuple[str, str, str, str], DatasetResolution] = {}
        self._lock = threading.RLock()
        LOGGER.info(
            "Dataset cache ready: directory=%s offline=%s proxy=%s xet=%s",
            self.cache_directory,
            self.local_files_only,
            selected_policy.proxy_label,
            xet_transport_label(selected_policy),
        )

    async def resolve(self, spec: Mapping[str, Any]) -> DatasetResolution:
        source = str(spec.get("source", "huggingface"))
        if source != "huggingface":
            raise DatasetResolutionError("Only Hugging Face datasets are supported")
        dataset_id = str(spec.get("id", "")).strip()
        try:
            validate_huggingface_repository_id(dataset_id)
        except HuggingFaceRepositoryError as exc:
            raise DatasetResolutionError(
                f"Dataset id must be a Hugging Face repository ID: {exc}"
            ) from exc
        filename = self._validate_filename(str(spec.get("filename", "")))
        revision = str(spec.get("revision") or "main").strip()
        if not revision:
            raise DatasetResolutionError("Dataset revision must not be empty")
        raw_adapter = str(spec.get("adapter", "")).strip()
        try:
            dataset_adapter = validate_external_dataset_adapter(raw_adapter)
        except ValueError as exc:
            raise DatasetResolutionError(str(exc)) from exc
        event_loop = asyncio.get_running_loop()
        progress_callback = _ARTIFACT_PROGRESS_CALLBACK.get()
        return await event_loop.run_in_executor(
            None,
            self._resolve_sync,
            dataset_id,
            filename,
            revision,
            dataset_adapter,
            progress_callback,
        )

    def _resolve_sync(
        self,
        dataset_id: str,
        filename: str,
        revision: str,
        dataset_adapter: str,
        progress_callback: Optional[ArtifactProgressCallback] = None,
    ) -> DatasetResolution:
        requested_key = (dataset_id, filename, revision, dataset_adapter)
        with self._lock:
            existing_resolution = self._entries.get(requested_key)
            if existing_resolution is not None and existing_resolution.path.is_file():
                return DatasetResolution(
                    source=existing_resolution.source,
                    dataset_id=existing_resolution.dataset_id,
                    filename=existing_resolution.filename,
                    revision=existing_resolution.revision,
                    adapter=existing_resolution.adapter,
                    path=existing_resolution.path,
                    cached=True,
                )

        raw_cached_path = try_to_load_from_cache(
            dataset_id,
            filename,
            cache_dir=self.cache_directory,
            revision=revision,
            repo_type="dataset",
        )
        cached_path = (
            Path(raw_cached_path)
            if isinstance(raw_cached_path, str) and Path(raw_cached_path).is_file()
            else None
        )
        if cached_path is None and self.local_files_only:
            cached_path = self._offline_artifact(dataset_id, filename, revision)
        if cached_path is not None:
            resolved_path = cached_path.resolve()
            cached_resolution = DatasetResolution(
                source="huggingface",
                dataset_id=dataset_id,
                filename=filename,
                revision=snapshot_revision(resolved_path, revision),
                adapter=dataset_adapter,
                path=resolved_path,
                cached=True,
            )
            self._remember(requested_key, cached_resolution)
            LOGGER.info(
                "Dataset artifact-cache hit: %s file=%s requested=%s resolved=%s",
                dataset_id,
                filename,
                revision,
                cached_resolution.revision,
            )
            return cached_resolution

        if self.local_files_only:
            raise DatasetResolutionError(
                f"Dataset {dataset_id!r}, file {filename!r}, at revision "
                f"{revision!r} is not present in local cache "
                f"{self.cache_directory}"
            )

        try:
            if progress_callback is None:
                raw_downloaded_path = hf_hub_download(
                    repo_id=dataset_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=self.cache_directory,
                )
            else:
                progress_class = _hub_byte_reporter_class(
                    progress_callback,
                    "dataset",
                    dataset_id,
                    filename,
                )
                raw_downloaded_path = hf_hub_download(
                    repo_id=dataset_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=self.cache_directory,
                    tqdm_class=progress_class,
                )
            downloaded_path = Path(raw_downloaded_path).resolve()
        except Exception as exc:
            raise DatasetResolutionError(
                f"Unable to resolve dataset {dataset_id!r}, file {filename!r}, "
                f"at revision {revision!r} from Hugging Face: {exc}. Check "
                f"Hugging Face network settings, {LLMPERF_PROXY}, or preload "
                f"{self.cache_directory} and enable {DATASET_OFFLINE}."
            ) from exc

        downloaded_resolution = DatasetResolution(
            source="huggingface",
            dataset_id=dataset_id,
            filename=filename,
            revision=snapshot_revision(downloaded_path, revision),
            adapter=dataset_adapter,
            path=downloaded_path,
            cached=False,
        )
        self._remember(requested_key, downloaded_resolution)
        return downloaded_resolution

    def _offline_artifact(
        self,
        dataset_id: str,
        filename: str,
        requested_revision: str,
    ) -> Optional[Path]:
        """Select one local dataset artifact without consulting Hub metadata."""

        repository_directory = self.cache_directory / (
            "datasets--" + dataset_id.replace("/", "--")
        )
        snapshots_directory = repository_directory / "snapshots"
        if not snapshots_directory.is_dir():
            return None
        candidates = []
        relative_path = PurePosixPath(filename)
        for snapshot_directory in sorted(snapshots_directory.iterdir()):
            if not snapshot_directory.is_dir():
                continue
            artifact_path = snapshot_directory.joinpath(*relative_path.parts)
            if artifact_path.is_file():
                candidates.append((artifact_path, snapshot_directory.name))
        exact_candidates = [
            candidate for candidate in candidates if candidate[1] == requested_revision
        ]
        if len(exact_candidates) == 1:
            return exact_candidates[0][0]
        if len(candidates) == 1:
            selected_candidate = candidates[0]
            LOGGER.info(
                "Dataset offline direct-cache fallback: %s file=%s requested=%s "
                "resolved=%s path=%s",
                dataset_id,
                filename,
                requested_revision,
                selected_candidate[1],
                selected_candidate[0],
            )
            return selected_candidate[0]
        if len(candidates) > 1:
            revisions = ", ".join(candidate[1] for candidate in candidates)
            raise DatasetResolutionError(
                f"Dataset {dataset_id!r}, file {filename!r}, revision "
                f"{requested_revision!r} has no exact local cache reference and "
                f"multiple usable local snapshots exist: {revisions}"
            )
        return None

    def _remember(
        self,
        requested_key: Tuple[str, str, str, str],
        resolution: DatasetResolution,
    ) -> None:
        with self._lock:
            self._entries[requested_key] = resolution
            resolved_key = (
                resolution.dataset_id,
                resolution.filename,
                resolution.revision,
                resolution.adapter,
            )
            self._entries[resolved_key] = resolution

    @staticmethod
    def _validate_filename(filename: str) -> str:
        normalized_filename = filename.strip()
        relative_path = PurePosixPath(normalized_filename)
        if (
            not normalized_filename
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in normalized_filename
        ):
            raise DatasetResolutionError(
                "Dataset filename must be a relative Hugging Face artifact path"
            )
        return normalized_filename


@dataclass(frozen=True)
class ArtifactEvidence:
    """Non-local evidence produced by one complete cache integrity pass."""

    kind: ArtifactKind
    repository_id: str
    filename: Optional[str]
    revision: str
    immutable_revision: bool
    cache_hit: bool
    file_count: int
    size_bytes: int
    sha256: str
    adapter: Optional[str] = None
    record_count: Optional[int] = None

    def public_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kind": self.kind,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "immutable_revision": self.immutable_revision,
            "cache_hit": self.cache_hit,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.filename is not None:
            result["filename"] = self.filename
        if self.adapter is not None:
            result["adapter"] = self.adapter
        if self.record_count is not None:
            result["record_count"] = self.record_count
        return result


# Active validation after materialization and before task persistence.


def _resolved_existing_path(path: Path, kind: ArtifactKind) -> Path:
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactValidationError(
            f"Resolved {kind} cache artifact does not exist"
        ) from exc
    if resolved_path.name.endswith(".incomplete"):
        raise ArtifactValidationError(
            f"Resolved {kind} cache artifact is still incomplete"
        )
    return resolved_path


def _hash_stable_file(path: Path) -> Tuple[int, str]:
    try:
        initial_stat = path.stat()
        if not path.is_file():
            raise ArtifactValidationError("Cache artifact entry is not a regular file")
        if path.name.endswith(".incomplete"):
            raise ArtifactValidationError("Cache artifact contains an incomplete file")
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
        final_stat = path.stat()
    except ArtifactValidationError:
        raise
    except OSError as exc:
        raise ArtifactValidationError("Cache artifact file is not readable") from exc
    if size_bytes != initial_stat.st_size or size_bytes != final_stat.st_size:
        raise ArtifactValidationError("Cache artifact size changed during validation")
    if initial_stat.st_mtime_ns != final_stat.st_mtime_ns:
        raise ArtifactValidationError(
            "Cache artifact modification time changed during validation"
        )
    return size_bytes, digest.hexdigest()


def _directory_files(directory: Path) -> Tuple[Path, ...]:
    return tuple(
        sorted(
            (entry for entry in directory.rglob("*") if entry.is_file()),
            key=lambda entry: entry.relative_to(directory).as_posix(),
        )
    )


def _hash_stable_directory(directory: Path) -> Tuple[int, int, str]:
    initial_files = _directory_files(directory)
    if not initial_files:
        raise ArtifactValidationError("Tokenizer cache directory contains no files")
    manifest_digest = hashlib.sha256()
    total_size_bytes = 0
    for artifact_file in initial_files:
        file_size_bytes, file_digest = _hash_stable_file(artifact_file)
        relative_name = artifact_file.relative_to(directory).as_posix()
        manifest_digest.update(relative_name.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(str(file_size_bytes).encode("ascii"))
        manifest_digest.update(b"\0")
        manifest_digest.update(file_digest.encode("ascii"))
        manifest_digest.update(b"\n")
        total_size_bytes += file_size_bytes
    final_files = _directory_files(directory)
    if initial_files != final_files:
        raise ArtifactValidationError(
            "Tokenizer cache directory changed during validation"
        )
    return len(initial_files), total_size_bytes, manifest_digest.hexdigest()


def validate_dataset_artifact(resolution: ArtifactResolution) -> ArtifactEvidence:
    """Read, hash, and parse one fully materialized dataset cache file."""

    descriptor = resolution.artifact_descriptor()
    if descriptor.kind != "dataset":
        raise ArtifactValidationError("Expected a dataset artifact descriptor")
    if descriptor.filename is None or descriptor.adapter is None:
        raise ArtifactValidationError(
            "Dataset artifact descriptor requires filename and adapter"
        )
    artifact_path = _resolved_existing_path(descriptor.path, "dataset")
    size_bytes, sha256 = _hash_stable_file(artifact_path)
    if size_bytes == 0:
        raise ArtifactValidationError("Dataset cache artifact is empty")
    try:
        prompt_dataset = load_prompt_dataset(
            PromptDatasetSource.external(
                descriptor.adapter,
                artifact_path,
                filename=descriptor.filename,
            )
        )
    except ValueError as exc:
        raise ArtifactValidationError(
            f"Dataset adapter validation failed: {exc}"
        ) from exc
    return ArtifactEvidence(
        kind="dataset",
        repository_id=descriptor.repository_id,
        filename=descriptor.filename,
        revision=descriptor.revision,
        immutable_revision=is_immutable_huggingface_revision(descriptor.revision),
        cache_hit=descriptor.cached,
        file_count=1,
        size_bytes=size_bytes,
        sha256=sha256,
        adapter=descriptor.adapter,
        record_count=len(prompt_dataset.records),
    )


def validate_tokenizer_artifact(resolution: ArtifactResolution) -> ArtifactEvidence:
    """Read and hash one atomically materialized tokenizer cache directory."""

    descriptor = resolution.artifact_descriptor()
    if descriptor.kind != "tokenizer":
        raise ArtifactValidationError("Expected a tokenizer artifact descriptor")
    artifact_directory = _resolved_existing_path(descriptor.path, "tokenizer")
    if not artifact_directory.is_dir():
        raise ArtifactValidationError(
            "Resolved tokenizer cache artifact is not a directory"
        )
    file_count, size_bytes, sha256 = _hash_stable_directory(artifact_directory)
    return ArtifactEvidence(
        kind="tokenizer",
        repository_id=descriptor.repository_id,
        filename=None,
        revision=descriptor.revision,
        immutable_revision=is_immutable_huggingface_revision(descriptor.revision),
        cache_hit=descriptor.cached,
        file_count=file_count,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def validate_resolved_artifacts(
    resolutions: Tuple[ArtifactResolution, ...],
) -> Tuple[ArtifactEvidence, ...]:
    """Validate an artifact set using only the common resolution contract."""

    evidence: list[ArtifactEvidence] = []
    for resolution in resolutions:
        descriptor = resolution.artifact_descriptor()
        if descriptor.kind == "dataset":
            artifact_evidence = validate_dataset_artifact(resolution)
        elif descriptor.kind == "tokenizer":
            artifact_evidence = validate_tokenizer_artifact(resolution)
        else:
            raise ArtifactValidationError(
                f"Unsupported artifact kind: {descriptor.kind}"
            )
        evidence.append(artifact_evidence)
    return tuple(evidence)


# Tokenizer materialization.


TOKENIZER_CACHE_DIRECTORY = "LLMPERF_TOKENIZER_CACHE"
TOKENIZER_OFFLINE = "LLMPERF_TOKENIZER_OFFLINE"
DEFAULT_TOKENIZER_CACHE_DIRECTORY = Path("~/.cache/llmperf/tokenizers")
TOKENIZERS_BACKEND_CLASS_ERROR = (
    "Tokenizer class TokenizersBackend does not exist or is not currently imported"
)
TOKENIZER_SNAPSHOT_ALLOW_PATTERNS = (
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template*",
    "chat_templates/*",
    "config.json",
    "*.model",
    "*.tiktoken",
    "*.bpe",
)


class TokenizerResolutionError(ArtifactResolutionError):
    """Raised when a submitted tokenizer cannot be resolved by the backend."""


class TokenizerResolver(Protocol):
    """Capability required by request orchestration to resolve a tokenizer."""

    async def resolve(self, spec: Mapping[str, Any]) -> "TokenizerResolution": ...


@dataclass(frozen=True)
class TokenizerResolution:
    source: str
    tokenizer_id: str
    revision: str
    use_fast: bool
    path: Path
    cached: bool

    def benchmark_spec(
        self,
        selection: Optional[str] = None,
        accuracy: Optional[str] = None,
        requested_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the immutable, non-local portion stored with a Runner."""

        result = {
            "source": self.source,
            "id": self.tokenizer_id,
            "revision": self.revision,
            "use_fast": self.use_fast,
            "immutable_revision": is_immutable_huggingface_revision(self.revision),
        }
        if selection is not None:
            result["selection"] = selection
        if accuracy is not None:
            result["accuracy"] = accuracy
        if requested_revision is not None:
            result["requested_revision"] = requested_revision
        return result

    def artifact_descriptor(self) -> ArtifactDescriptor:
        return ArtifactDescriptor(
            kind="tokenizer",
            repository_id=self.tokenizer_id,
            revision=self.revision,
            path=self.path,
            cached=self.cached,
        )


class TokenizerCache:
    """Resolve remote tokenizer IDs into server-owned local directories.

    Hugging Face downloads a bounded tokenizer-only snapshot. Transformers loads that
    local snapshot without network access, and the resulting tokenizer artifacts are
    copied into a small LLMPerf-managed cache so Workers never perform their own lookup.
    """

    def __init__(
        self,
        cache_directory: Optional[Path] = None,
        local_files_only: Optional[bool] = None,
        proxy_url: Optional[str] = None,
        outbound_policy: Optional[OutboundPolicy] = None,
    ):
        configured_directory = os.environ.get(TOKENIZER_CACHE_DIRECTORY)
        selected_directory = (
            Path(configured_directory)
            if configured_directory
            else cache_directory or DEFAULT_TOKENIZER_CACHE_DIRECTORY
        )
        self.cache_directory = selected_directory.expanduser().resolve()
        self.download_directory = self.cache_directory / "downloads"
        self.resolved_directory = self.cache_directory / "resolved"
        try:
            configured_offline = environment_flag(os.environ, TOKENIZER_OFFLINE)
        except ArtifactResolutionError as exc:
            raise TokenizerResolutionError(str(exc)) from exc
        self.local_files_only = (
            configured_offline if local_files_only is None else local_files_only
        )
        if outbound_policy is not None and proxy_url is not None:
            raise TokenizerResolutionError(
                "outbound_policy and proxy_url cannot both be provided"
            )
        try:
            selected_policy = outbound_policy or configure_outbound_transport(
                os.environ, proxy_url
            )
        except OutboundConfigurationError as exc:
            raise TokenizerResolutionError(str(exc)) from exc
        self.outbound_policy = selected_policy
        self.proxy_url = selected_policy.proxy_url
        self._entries: Dict[Tuple[str, str, bool], TokenizerResolution] = {}
        self._key_locks: Dict[Tuple[str, str, bool], threading.Lock] = {}
        self._lock = threading.RLock()
        LOGGER.info(
            "Tokenizer cache ready: directory=%s offline=%s proxy=%s xet=%s",
            self.cache_directory,
            self.local_files_only,
            selected_policy.proxy_label,
            xet_transport_label(selected_policy),
        )

    async def resolve(self, spec: Mapping[str, Any]) -> TokenizerResolution:
        if str(spec.get("source", "huggingface")) != "huggingface":
            raise TokenizerResolutionError("Only Hugging Face tokenizers are supported")
        tokenizer_id = str(spec.get("id", "")).strip()
        if not tokenizer_id:
            raise TokenizerResolutionError("Tokenizer id must not be empty")
        try:
            validate_huggingface_repository_id(tokenizer_id)
        except HuggingFaceRepositoryError as exc:
            raise TokenizerResolutionError(
                f"Tokenizer id must be a Hugging Face repository ID: {exc}"
            ) from exc
        raw_revision = spec.get("revision")
        revision = str(raw_revision).strip() if raw_revision is not None else "main"
        if not revision:
            raise TokenizerResolutionError("Tokenizer revision must not be empty")
        use_fast = bool(spec.get("use_fast", True))
        loop = asyncio.get_running_loop()
        progress_callback = _ARTIFACT_PROGRESS_CALLBACK.get()
        return await loop.run_in_executor(
            None,
            self._resolve_sync,
            tokenizer_id,
            revision,
            use_fast,
            progress_callback,
        )

    def _resolve_sync(
        self,
        tokenizer_id: str,
        revision: str,
        use_fast: bool,
        progress_callback: Optional[ArtifactProgressCallback] = None,
    ) -> TokenizerResolution:
        requested_key = (tokenizer_id, revision, use_fast)
        with self._lock:
            cached = self._entries.get(requested_key)
            if cached is not None and cached.path.is_dir():
                LOGGER.debug(
                    "Tokenizer memory-cache hit: %s@%s", tokenizer_id, revision
                )
                return TokenizerResolution(
                    source=cached.source,
                    tokenizer_id=cached.tokenizer_id,
                    revision=cached.revision,
                    use_fast=cached.use_fast,
                    path=cached.path,
                    cached=True,
                )
            key_lock = self._key_locks.setdefault(requested_key, threading.Lock())

        with key_lock:
            with self._lock:
                cached = self._entries.get(requested_key)
                if cached is not None and cached.path.is_dir():
                    LOGGER.debug(
                        "Tokenizer memory-cache hit: %s@%s", tokenizer_id, revision
                    )
                    return TokenizerResolution(
                        source=cached.source,
                        tokenizer_id=cached.tokenizer_id,
                        revision=cached.revision,
                        use_fast=cached.use_fast,
                        path=cached.path,
                        cached=True,
                    )
            cached_snapshot: Optional[Tuple[Path, str]] = None
            resolved_revision = self._resolved_revision(None, tokenizer_id, revision)
            existing_target = self.resolved_directory / self._artifact_key(
                tokenizer_id, resolved_revision, use_fast
            )
            if not existing_target.is_dir() and self.local_files_only:
                cached_snapshot = self._cached_snapshot(tokenizer_id, revision)
                if cached_snapshot is None:
                    cached_snapshot = self._offline_snapshot(
                        tokenizer_id, revision, use_fast
                    )
                if cached_snapshot is not None:
                    resolved_revision = cached_snapshot[1]
                    existing_target = self.resolved_directory / self._artifact_key(
                        tokenizer_id, resolved_revision, use_fast
                    )
            if existing_target.is_dir():
                LOGGER.info(
                    "Tokenizer artifact-cache hit: %s requested=%s resolved=%s "
                    "immutable=%s",
                    tokenizer_id,
                    revision,
                    resolved_revision,
                    is_immutable_huggingface_revision(resolved_revision),
                )
                resolution = TokenizerResolution(
                    source="huggingface",
                    tokenizer_id=tokenizer_id,
                    revision=resolved_revision,
                    use_fast=use_fast,
                    path=existing_target,
                    cached=True,
                )
                resolved_key = (tokenizer_id, resolved_revision, use_fast)
                with self._lock:
                    self._entries[requested_key] = resolution
                    # The API resolves mutable aliases such as ``main`` before
                    # persistence. The Scheduler then receives the immutable
                    # revision. Register both keys so that handoff is a local
                    # memory-cache hit instead of a second Hub lookup.
                    self._entries[resolved_key] = resolution
                return resolution
            try:
                self.download_directory.mkdir(parents=True, exist_ok=True)
                self.resolved_directory.mkdir(parents=True, exist_ok=True)
                LOGGER.info(
                    "Resolving tokenizer %s@%s (offline=%s, proxy=%s)",
                    tokenizer_id,
                    revision,
                    self.local_files_only,
                    self.outbound_policy.proxy_label,
                )
                if cached_snapshot is None:
                    cached_snapshot = self._cached_snapshot(tokenizer_id, revision)
                if cached_snapshot is None and self.local_files_only:
                    cached_snapshot = self._offline_snapshot(
                        tokenizer_id, revision, use_fast
                    )
                snapshot_path = cached_snapshot[0] if cached_snapshot else None
                resolved_snapshot_revision = (
                    cached_snapshot[1] if cached_snapshot else None
                )
                try:
                    if snapshot_path is None:
                        if self.local_files_only:
                            raise TokenizerResolutionError(
                                f"Tokenizer {tokenizer_id!r} at revision "
                                f"{revision!r} is not present in local cache "
                                f"{self.cache_directory}"
                            )
                        raise FileNotFoundError("no cached tokenizer snapshot")
                    tokenizer = self._load_local_tokenizer(
                        tokenizer_id, revision, snapshot_path, use_fast
                    )
                    LOGGER.info(
                        "Tokenizer snapshot-cache hit: %s requested=%s resolved=%s",
                        tokenizer_id,
                        revision,
                        resolved_snapshot_revision,
                    )
                except TokenizerResolutionError:
                    raise
                except Exception as cached_exc:
                    if self.local_files_only:
                        raise TokenizerResolutionError(
                            f"Cached tokenizer {tokenizer_id!r} at revision "
                            f"{revision!r} is unusable at {snapshot_path}: "
                            f"{cached_exc}"
                        ) from cached_exc
                    if snapshot_path is not None:
                        LOGGER.warning(
                            "Cached tokenizer snapshot is incomplete or unusable: "
                            "%s requested=%s path=%s; refreshing bounded artifacts",
                            tokenizer_id,
                            revision,
                            snapshot_path,
                        )
                    snapshot_path = self._download_snapshot(
                        tokenizer_id,
                        revision,
                        progress_callback,
                    )
                    resolved_snapshot_revision = snapshot_revision(
                        snapshot_path, revision
                    )
                    tokenizer = self._load_local_tokenizer(
                        tokenizer_id, revision, snapshot_path, use_fast
                    )
                resolved_revision = (
                    resolved_snapshot_revision
                    or self._resolved_revision(tokenizer, tokenizer_id, revision)
                )
                target = self.resolved_directory / self._artifact_key(
                    tokenizer_id, resolved_revision, use_fast
                )
                artifact_cached = target.is_dir()
                if not artifact_cached:
                    self._save_atomically(tokenizer, target)
                resolution = TokenizerResolution(
                    source="huggingface",
                    tokenizer_id=tokenizer_id,
                    revision=resolved_revision,
                    use_fast=use_fast,
                    path=target,
                    cached=artifact_cached,
                )
                LOGGER.info(
                    "Tokenizer resolved: %s requested=%s resolved=%s "
                    "immutable=%s cached=%s",
                    tokenizer_id,
                    revision,
                    resolved_revision,
                    is_immutable_huggingface_revision(resolved_revision),
                    artifact_cached,
                )
            except TokenizerResolutionError:
                raise
            except Exception as exc:
                mode = "local cache" if self.local_files_only else "Hugging Face"
                LOGGER.error(
                    "Unable to resolve tokenizer %s@%s from %s: %s",
                    tokenizer_id,
                    revision,
                    mode,
                    exc,
                )
                raise TokenizerResolutionError(
                    f"Unable to resolve tokenizer {tokenizer_id!r} at revision "
                    f"{revision!r} from {mode}: {exc}. Check "
                    f"{LLMPERF_PROXY}, proxy HTTPS CONNECT/TLS support, or "
                    f"preload {self.cache_directory} and enable "
                    f"{TOKENIZER_OFFLINE}."
                ) from exc

            resolved_key = (tokenizer_id, resolution.revision, use_fast)
            with self._lock:
                self._entries[requested_key] = resolution
                self._entries[resolved_key] = resolution
            return resolution

    def _download_snapshot(
        self,
        tokenizer_id: str,
        revision: str,
        progress_callback: Optional[ArtifactProgressCallback] = None,
    ) -> Path:
        if progress_callback is None:
            raw_snapshot_path = snapshot_download(
                repo_id=tokenizer_id,
                revision=revision,
                cache_dir=str(self.download_directory),
                local_files_only=self.local_files_only,
                allow_patterns=list(TOKENIZER_SNAPSHOT_ALLOW_PATTERNS),
            )
        else:
            progress_class = _hub_byte_reporter_class(
                progress_callback,
                "tokenizer",
                tokenizer_id,
                None,
            )
            raw_snapshot_path = snapshot_download(
                repo_id=tokenizer_id,
                revision=revision,
                cache_dir=str(self.download_directory),
                local_files_only=self.local_files_only,
                allow_patterns=list(TOKENIZER_SNAPSHOT_ALLOW_PATTERNS),
                tqdm_class=progress_class,
            )
        return Path(raw_snapshot_path)

    def _load_local_tokenizer(
        self,
        tokenizer_id: str,
        revision: str,
        snapshot_path: Path,
        use_fast: bool,
    ) -> Any:
        load_options: Dict[str, Any] = {
            "use_fast": use_fast,
            "trust_remote_code": False,
            "local_files_only": True,
        }
        try:
            return AutoTokenizer.from_pretrained(str(snapshot_path), **load_options)
        except ValueError as exc:
            if not use_fast or TOKENIZERS_BACKEND_CLASS_ERROR not in str(exc):
                raise
            # Transformers 5 renamed the generic tokenizer.json loader to
            # TokenizersBackend. Older releases cannot resolve that class name from
            # newer metadata, although they can load the same local tokenizer.json
            # through PreTrainedTokenizerFast.
            LOGGER.warning(
                "Tokenizer %s@%s requires the Transformers 5 TokenizersBackend "
                "class; using the compatible local fast tokenizer loader",
                tokenizer_id,
                revision,
            )
            fallback_options = dict(load_options)
            fallback_options.pop("use_fast", None)
            # Transformers 5 accepts a list here. Transformers 4 treats this field
            # as a mapping and calls .keys(), so override only that incompatible
            # metadata field. Special-token IDs remain in tokenizer.json.
            fallback_options["extra_special_tokens"] = {}
            return PreTrainedTokenizerFast.from_pretrained(
                str(snapshot_path), **fallback_options
            )

    def _resolved_revision(
        self,
        tokenizer: Any,
        tokenizer_id: str,
        requested_revision: str,
    ) -> str:
        resolved = getattr(tokenizer, "_commit_hash", None) if tokenizer else None
        if not resolved and tokenizer is not None:
            init_kwargs = getattr(tokenizer, "init_kwargs", {})
            if isinstance(init_kwargs, dict):
                resolved = init_kwargs.get("_commit_hash")
        if resolved and is_immutable_huggingface_revision(str(resolved)):
            return str(resolved)
        if is_immutable_huggingface_revision(requested_revision):
            return requested_revision
        cached_revision = self._cached_revision(tokenizer_id, requested_revision)
        return cached_revision or str(resolved or requested_revision)

    def _cached_revision(
        self, tokenizer_id: str, requested_revision: str
    ) -> Optional[str]:
        snapshot = self._cached_snapshot(tokenizer_id, requested_revision)
        return snapshot[1] if snapshot else None

    def _cached_snapshot(
        self, tokenizer_id: str, requested_revision: str
    ) -> Optional[Tuple[Path, str]]:
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        ):
            try:
                cached_path = try_to_load_from_cache(
                    tokenizer_id,
                    filename,
                    cache_dir=self.download_directory,
                    revision=requested_revision,
                )
            except Exception:
                LOGGER.debug(
                    "Unable to inspect cached tokenizer revision for %s@%s",
                    tokenizer_id,
                    requested_revision,
                    exc_info=True,
                )
                continue
            if not isinstance(cached_path, str) or not Path(cached_path).is_file():
                continue
            snapshot_path = Path(cached_path).parent
            revision = snapshot_revision(snapshot_path, requested_revision)
            if is_immutable_huggingface_revision(revision):
                return snapshot_path, revision
        return None

    def _offline_snapshot(
        self,
        tokenizer_id: str,
        requested_revision: str,
        use_fast: bool,
    ) -> Optional[Tuple[Path, str]]:
        """Select an unambiguous local snapshot without consulting Hub metadata."""

        repository = self.download_directory / (
            "models--" + tokenizer_id.replace("/", "--")
        )
        snapshots = repository / "snapshots"
        if not snapshots.is_dir():
            return None
        snapshots_by_revision = []
        loadable_snapshots = []
        for path in sorted(snapshots.iterdir()):
            revision = path.name
            if not path.is_dir() or not is_immutable_huggingface_revision(revision):
                continue
            candidate = (path, revision)
            snapshots_by_revision.append(candidate)
            if any(
                (path / filename).is_file()
                for filename in (
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "tokenizer.model",
                )
            ):
                loadable_snapshots.append(candidate)
        if not snapshots_by_revision:
            return None

        # A resolved artifact is self-contained. Prefer it even when the raw Hub
        # snapshot contains dangling relative symlinks after a cross-host cache
        # migration, because loading the source snapshot again is unnecessary.
        resolved = [
            item
            for item in snapshots_by_revision
            if (
                self.resolved_directory
                / self._artifact_key(tokenizer_id, item[1], use_fast)
            ).is_dir()
        ]
        exact_resolved = [item for item in resolved if item[1] == requested_revision]
        if len(exact_resolved) == 1:
            return exact_resolved[0]
        exact_loadable = [
            item for item in loadable_snapshots if item[1] == requested_revision
        ]
        if len(exact_loadable) == 1:
            return exact_loadable[0]
        selectable = resolved or loadable_snapshots
        if not selectable:
            return None
        if len(selectable) == 1:
            selected = selectable[0]
            LOGGER.info(
                "Tokenizer offline direct-cache fallback: %s requested=%s "
                "resolved=%s path=%s",
                tokenizer_id,
                requested_revision,
                selected[1],
                selected[0],
            )
            return selected
        revisions = ", ".join(item[1] for item in selectable)
        raise TokenizerResolutionError(
            f"Tokenizer {tokenizer_id!r} revision {requested_revision!r} has no "
            f"exact local cache reference and multiple usable local snapshots "
            f"exist: {revisions}"
        )

    @staticmethod
    def _artifact_key(tokenizer_id: str, revision: str, use_fast: bool) -> str:
        content = f"huggingface\0{tokenizer_id}\0{revision}\0{int(use_fast)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _save_atomically(self, tokenizer: Any, target: Path) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix=".resolving-", dir=str(self.resolved_directory))
        )
        try:
            tokenizer.save_pretrained(str(temporary))
            try:
                temporary.rename(target)
            except OSError:
                # Another backend process may have populated the shared cache first.
                if not target.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))


@dataclass(frozen=True)
class ArtifactCaches:
    """Dataset and tokenizer resolvers sharing one outbound transport policy."""

    dataset: DatasetCache
    tokenizer: TokenizerCache

    @classmethod
    def from_environment(
        cls, outbound_policy: Optional[OutboundPolicy] = None
    ) -> "ArtifactCaches":
        selected_policy = outbound_policy or configure_outbound_transport(os.environ)
        return cls(
            dataset=DatasetCache(outbound_policy=selected_policy),
            tokenizer=TokenizerCache(outbound_policy=selected_policy),
        )

"""Server-owned tokenizer resolution and local artifact caching."""

import asyncio
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

from llmperf.logging import route_library_logs
from llmperf_backend.huggingface import (
    HUGGINGFACE_PROXY,
    HuggingFaceProxyError,
    configure_huggingface_http,
    huggingface_proxy_label,
    resolve_huggingface_proxy,
)

# This backend intentionally uses Transformers without a model framework: only
# tokenizer configuration and artifacts are needed. Suppress that import advisory
# before Transformers initializes its private logger.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from huggingface_hub import snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import HFValidationError, validate_repo_id
from transformers import AutoTokenizer, PreTrainedTokenizerFast


route_library_logs("huggingface_hub", "transformers")


TOKENIZER_CACHE_DIRECTORY = "LLMPERF_TOKENIZER_CACHE"
TOKENIZER_OFFLINE = "LLMPERF_TOKENIZER_OFFLINE"
DEFAULT_TOKENIZER_CACHE_DIRECTORY = Path("~/.cache/llmperf/tokenizers")
LOGGER = logging.getLogger(__name__)
TOKENIZERS_BACKEND_CLASS_ERROR = (
    "Tokenizer class TokenizersBackend does not exist or is not currently imported"
)
IMMUTABLE_HUGGINGFACE_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
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


class TokenizerResolutionError(RuntimeError):
    """Raised when a submitted tokenizer cannot be resolved by the backend."""


def _environment_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TokenizerResolutionError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0"
    )


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
            "immutable_revision": bool(
                IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(self.revision)
            ),
        }
        if selection is not None:
            result["selection"] = selection
        if accuracy is not None:
            result["accuracy"] = accuracy
        if requested_revision is not None:
            result["requested_revision"] = requested_revision
        return result


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
        self.local_files_only = (
            _environment_flag(TOKENIZER_OFFLINE)
            if local_files_only is None
            else local_files_only
        )
        try:
            self.proxy_url = resolve_huggingface_proxy(proxy_url)
        except HuggingFaceProxyError as exc:
            raise TokenizerResolutionError(str(exc)) from exc
        configure_huggingface_http(self.proxy_url)
        self._entries: Dict[Tuple[str, str, bool], TokenizerResolution] = {}
        self._key_locks: Dict[Tuple[str, str, bool], threading.Lock] = {}
        self._lock = threading.RLock()
        LOGGER.info(
            "Tokenizer cache ready: directory=%s offline=%s proxy=%s",
            self.cache_directory,
            self.local_files_only,
            huggingface_proxy_label(self.proxy_url),
        )

    async def resolve(self, spec: Mapping[str, Any]) -> TokenizerResolution:
        if str(spec.get("source", "huggingface")) != "huggingface":
            raise TokenizerResolutionError("Only Hugging Face tokenizers are supported")
        tokenizer_id = str(spec.get("id", "")).strip()
        if not tokenizer_id:
            raise TokenizerResolutionError("Tokenizer id must not be empty")
        try:
            validate_repo_id(tokenizer_id)
        except HFValidationError as exc:
            raise TokenizerResolutionError(
                f"Tokenizer id must be a Hugging Face repository ID: {exc}"
            ) from exc
        raw_revision = spec.get("revision")
        revision = str(raw_revision).strip() if raw_revision is not None else "main"
        if not revision:
            raise TokenizerResolutionError("Tokenizer revision must not be empty")
        raw_requested_revision = spec.get("requested_revision")
        requested_revision = (
            str(raw_requested_revision).strip()
            if raw_requested_revision is not None
            else None
        )
        use_fast = bool(spec.get("use_fast", True))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._resolve_sync,
            tokenizer_id,
            revision,
            use_fast,
            requested_revision,
        )

    def _resolve_sync(
        self,
        tokenizer_id: str,
        revision: str,
        use_fast: bool,
        requested_revision: Optional[str] = None,
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
            legacy_revisions = [revision]
            if requested_revision and requested_revision not in legacy_revisions:
                legacy_revisions.append(requested_revision)
            if not existing_target.is_dir():
                for legacy_revision in legacy_revisions:
                    legacy_target = self.resolved_directory / self._artifact_key(
                        tokenizer_id, legacy_revision, use_fast
                    )
                    if legacy_target.is_dir():
                        existing_target = legacy_target
                        break
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
                    bool(IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(resolved_revision)),
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
                    huggingface_proxy_label(self.proxy_url),
                )
                if cached_snapshot is None:
                    cached_snapshot = self._cached_snapshot(tokenizer_id, revision)
                if cached_snapshot is None and self.local_files_only:
                    cached_snapshot = self._offline_snapshot(
                        tokenizer_id, revision, use_fast
                    )
                snapshot_path = cached_snapshot[0] if cached_snapshot else None
                snapshot_revision = cached_snapshot[1] if cached_snapshot else None
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
                        snapshot_revision,
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
                    snapshot_path = self._download_snapshot(tokenizer_id, revision)
                    snapshot_revision = self._snapshot_revision(
                        snapshot_path, revision
                    )
                    tokenizer = self._load_local_tokenizer(
                        tokenizer_id, revision, snapshot_path, use_fast
                    )
                resolved_revision = snapshot_revision or self._resolved_revision(
                    tokenizer, tokenizer_id, revision
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
                    bool(IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(resolved_revision)),
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
                    f"{HUGGINGFACE_PROXY}, proxy HTTPS CONNECT/TLS support, or "
                    f"preload {self.cache_directory} and enable "
                    f"{TOKENIZER_OFFLINE}."
                ) from exc

            resolved_key = (tokenizer_id, resolution.revision, use_fast)
            with self._lock:
                self._entries[requested_key] = resolution
                self._entries[resolved_key] = resolution
            return resolution

    def _download_snapshot(self, tokenizer_id: str, revision: str) -> Path:
        proxy = (
            {"http": self.proxy_url, "https": self.proxy_url}
            if self.proxy_url
            else None
        )
        return Path(
            snapshot_download(
                repo_id=tokenizer_id,
                revision=revision,
                cache_dir=str(self.download_directory),
                local_files_only=self.local_files_only,
                proxies=proxy,
                allow_patterns=list(TOKENIZER_SNAPSHOT_ALLOW_PATTERNS),
            )
        )

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
        if resolved and IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(str(resolved)):
            return str(resolved)
        if IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(requested_revision):
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
            revision = self._snapshot_revision(snapshot_path, requested_revision)
            if IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(revision):
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
        candidates = []
        for path in sorted(snapshots.iterdir()):
            revision = path.name
            if not path.is_dir() or not IMMUTABLE_HUGGINGFACE_REVISION.fullmatch(
                revision
            ):
                continue
            if not any(
                (path / filename).is_file()
                for filename in (
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "tokenizer.model",
                )
            ):
                continue
            candidates.append((path, revision))
        if not candidates:
            return None

        exact = [item for item in candidates if item[1] == requested_revision]
        if len(exact) == 1:
            return exact[0]
        resolved = [
            item
            for item in candidates
            if (
                self.resolved_directory
                / self._artifact_key(tokenizer_id, item[1], use_fast)
            ).is_dir()
        ]
        selectable = resolved or candidates
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
    def _snapshot_revision(path: Path, requested_revision: str) -> str:
        parts = path.parts
        try:
            snapshot_index = parts.index("snapshots")
            revision = parts[snapshot_index + 1]
        except (ValueError, IndexError):
            return requested_revision
        return revision

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

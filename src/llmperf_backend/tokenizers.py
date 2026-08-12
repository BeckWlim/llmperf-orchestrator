"""Server-owned tokenizer resolution and local artifact caching."""

import asyncio
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from llmperf.logging import route_library_logs

# This backend intentionally uses Transformers without a model framework: only
# tokenizer configuration and artifacts are needed. Suppress that import advisory
# before Transformers initializes its private logger.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from huggingface_hub.utils import HFValidationError, validate_repo_id
from transformers import AutoTokenizer


route_library_logs("huggingface_hub", "transformers")


TOKENIZER_CACHE_DIRECTORY_ENV = "LLMPERF_TOKENIZER_CACHE_DIR"
TOKENIZER_LOCAL_FILES_ONLY_ENV = "LLMPERF_TOKENIZER_LOCAL_FILES_ONLY"
TOKENIZER_PROXY_ENV = "LLMPERF_TOKENIZER_PROXY"
WORKER_TOKENIZER_PATH_ENV = "LLMPERF_TOKENIZER_PATH"
WORKER_TOKENIZER_USE_FAST_ENV = "LLMPERF_TOKENIZER_USE_FAST"
DEFAULT_TOKENIZER_CACHE_DIRECTORY = Path("~/.cache/llmperf/tokenizers")
LOGGER = logging.getLogger(__name__)


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


def _validate_proxy_url(proxy_url: str) -> str:
    normalized = proxy_url.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise TokenizerResolutionError(
            f"{TOKENIZER_PROXY_ENV} must be an HTTP(S) proxy URL"
        )
    if parsed.query or parsed.fragment:
        raise TokenizerResolutionError(
            f"{TOKENIZER_PROXY_ENV} must not contain query or fragment components"
        )
    return normalized


def _proxy_label(proxy_url: Optional[str]) -> str:
    """Describe a proxy for logs without exposing credentials."""

    if not proxy_url:
        return "environment/default"
    parsed = urlsplit(proxy_url)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


@dataclass(frozen=True)
class TokenizerResolution:
    source: str
    tokenizer_id: str
    revision: str
    use_fast: bool
    path: Path
    cached: bool

    def benchmark_spec(self) -> Dict[str, Any]:
        """Return the immutable, non-local portion stored with a Runner."""

        return {
            "source": self.source,
            "id": self.tokenizer_id,
            "revision": self.revision,
            "use_fast": self.use_fast,
        }


class TokenizerCache:
    """Resolve remote tokenizer IDs into server-owned local directories.

    Transformers performs the remote lookup. The resulting tokenizer artifacts are
    copied into a small LLMPerf-managed cache so Workers can load them with
    ``local_files_only=True`` and never perform their own network lookup.
    """

    def __init__(
        self,
        cache_directory: Optional[Path] = None,
        local_files_only: Optional[bool] = None,
        proxy_url: Optional[str] = None,
    ):
        configured_directory = os.environ.get(TOKENIZER_CACHE_DIRECTORY_ENV)
        selected_directory = (
            Path(configured_directory)
            if configured_directory
            else cache_directory or DEFAULT_TOKENIZER_CACHE_DIRECTORY
        )
        self.cache_directory = selected_directory.expanduser().resolve()
        self.download_directory = self.cache_directory / "downloads"
        self.resolved_directory = self.cache_directory / "resolved"
        self.local_files_only = (
            _environment_flag(TOKENIZER_LOCAL_FILES_ONLY_ENV)
            if local_files_only is None
            else local_files_only
        )
        configured_proxy = (
            proxy_url
            if proxy_url is not None
            else os.environ.get(TOKENIZER_PROXY_ENV, "")
        )
        self.proxy_url = (
            _validate_proxy_url(configured_proxy) if configured_proxy else None
        )
        self._entries: Dict[Tuple[str, str, bool], TokenizerResolution] = {}
        self._key_locks: Dict[Tuple[str, str, bool], threading.Lock] = {}
        self._lock = threading.RLock()
        LOGGER.info(
            "Tokenizer cache ready: directory=%s offline=%s proxy=%s",
            self.cache_directory,
            self.local_files_only,
            _proxy_label(self.proxy_url),
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
        use_fast = bool(spec.get("use_fast", True))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._resolve_sync, tokenizer_id, revision, use_fast
        )

    def _resolve_sync(
        self, tokenizer_id: str, revision: str, use_fast: bool
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
            existing_target = self.resolved_directory / self._artifact_key(
                tokenizer_id, revision, use_fast
            )
            if existing_target.is_dir():
                LOGGER.info(
                    "Tokenizer artifact-cache hit: %s@%s", tokenizer_id, revision
                )
                resolution = TokenizerResolution(
                    source="huggingface",
                    tokenizer_id=tokenizer_id,
                    revision=revision,
                    use_fast=use_fast,
                    path=existing_target,
                    cached=True,
                )
                with self._lock:
                    self._entries[requested_key] = resolution
                return resolution
            try:
                self.download_directory.mkdir(parents=True, exist_ok=True)
                self.resolved_directory.mkdir(parents=True, exist_ok=True)
                LOGGER.info(
                    "Resolving tokenizer %s@%s (offline=%s, proxy=%s)",
                    tokenizer_id,
                    revision,
                    self.local_files_only,
                    _proxy_label(self.proxy_url),
                )
                load_options: Dict[str, Any] = {
                    "revision": revision,
                    "use_fast": use_fast,
                    "trust_remote_code": False,
                    "cache_dir": str(self.download_directory),
                    "local_files_only": self.local_files_only,
                }
                if self.proxy_url:
                    load_options["proxies"] = {
                        "http": self.proxy_url,
                        "https": self.proxy_url,
                    }
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **load_options)
                resolved_revision = self._resolved_revision(tokenizer, revision)
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
                    "Tokenizer resolved: %s@%s (cached=%s)",
                    tokenizer_id,
                    resolved_revision,
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
                    f"{TOKENIZER_PROXY_ENV}, proxy HTTPS CONNECT/TLS support, or "
                    f"preload {self.cache_directory} and enable "
                    f"{TOKENIZER_LOCAL_FILES_ONLY_ENV}."
                ) from exc

            resolved_key = (tokenizer_id, resolution.revision, use_fast)
            with self._lock:
                self._entries[requested_key] = resolution
                self._entries[resolved_key] = resolution
            return resolution

    @staticmethod
    def _resolved_revision(tokenizer: Any, requested_revision: str) -> str:
        resolved = getattr(tokenizer, "_commit_hash", None)
        if not resolved:
            init_kwargs = getattr(tokenizer, "init_kwargs", {})
            if isinstance(init_kwargs, dict):
                resolved = init_kwargs.get("_commit_hash")
        return str(resolved or requested_revision)

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

"""Server-owned provider profiles, worker credential injection, and model discovery."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROVIDER_PREFIX = "LLMPERF_PROVIDER_"
LOGGER = logging.getLogger(__name__)
CONFIGURATION_ERROR = (
    "Provider configuration could not be loaded; correct it and reload Providers."
)
PROVIDER_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
PROFILE_FIELDS = (
    "PATH",
    "TTL",
    "DISCOVERY",
    "URLVAR",
    "KEYVAR",
    "ADAPTER",
    "MODELS",
    "URL",
    "KEY",
)
DEFAULT_BASE_VARIABLES = {
    "anthropic": "ANTHROPIC_API_BASE",
    "openai": "OPENAI_API_BASE",
}
DEFAULT_KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class ProviderConfigError(ValueError):
    pass


class ProviderDiscoveryError(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_provider_id(raw_name: str) -> str:
    provider_id = raw_name.strip().lower().replace("_", "-")
    if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
        raise ProviderConfigError(f"Invalid provider profile ID: {raw_name!r}")
    return provider_id


def _validate_api_base(provider_id: str, api_base: str) -> None:
    if not api_base:
        return
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderConfigError(
            f"Provider {provider_id!r} URL must be an HTTP(S) URL"
        )
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderConfigError(
            f"Provider {provider_id!r} URL must not contain userinfo, query, "
            "or fragment components"
        )


def _validate_environment_name(provider_id: str, field: str, name: str) -> str:
    if not ENVIRONMENT_NAME_PATTERN.fullmatch(name):
        raise ProviderConfigError(
            f"Provider {provider_id!r} {field} must be an environment variable name"
        )
    return name


@dataclass(frozen=True)
class ProviderProfile:
    provider_id: str
    adapter: str
    api_base: str
    api_key: str = field(repr=False)
    api_base_env: str
    api_key_env: str
    discovery: str
    models_path: str
    static_models: Tuple[str, ...]
    model_cache_ttl_seconds: int
    credential_source_env: Optional[str] = None

    def public_dict(self) -> Dict[str, Any]:
        discovery: Dict[str, Any] = {
            "mode": self.discovery,
            "cache_ttl_seconds": self.model_cache_ttl_seconds,
            "static_model_count": len(self.static_models),
        }
        if self.discovery == "openai":
            discovery["path"] = self.models_path
        return {
            "id": self.provider_id,
            "adapter": self.adapter,
            "base_url": self.api_base or None,
            "api_key_configured": bool(self.api_key),
            "typical_models": list(self.static_models[:3]),
            "model_discovery": discovery,
        }


class ProviderRegistry:
    """Atomically reloadable Provider definitions parsed from a narrow source."""

    def __init__(
        self,
        profiles: Sequence[ProviderProfile],
        reload_loader: Optional[Callable[[], Mapping[str, str]]] = None,
        profile_loader: Optional[Callable[[], Sequence[ProviderProfile]]] = None,
        *,
        allow_empty: bool = False,
    ):
        self._profiles = {profile.provider_id: profile for profile in profiles}
        if not self._profiles and not allow_empty:
            raise ProviderConfigError("At least one provider profile is required")
        self._reload_loader = reload_loader
        self._profile_loader = profile_loader
        self._lock = threading.RLock()
        self._generation = 0
        self._configuration_error: Optional[str] = None
        self._loaded_at = _utcnow()
        self._credential_environment_names = {
            name
            for profile in self._profiles.values()
            for name in (
                profile.api_base_env,
                profile.api_key_env,
                profile.credential_source_env,
            )
            if name is not None
        }

    @classmethod
    def from_environment(
        cls,
        environment: Optional[Mapping[str, str]] = None,
        *,
        reload_loader: Optional[Callable[[], Mapping[str, str]]] = None,
    ) -> "ProviderRegistry":
        return cls(
            cls.profiles_from_environment(environment), reload_loader=reload_loader
        )

    @classmethod
    def from_loader(
        cls, loader: Callable[[], Sequence[ProviderProfile]]
    ) -> "ProviderRegistry":
        try:
            return cls(loader(), profile_loader=loader)
        except Exception:
            # External Provider configuration must not take down the control plane.
            # Never log parser exceptions: their context can contain credentials.
            registry = cls([], profile_loader=loader, allow_empty=True)
            registry._configuration_error = CONFIGURATION_ERROR
            LOGGER.error(CONFIGURATION_ERROR)
            return registry

    @classmethod
    def profiles_from_environment(
        cls, environment: Optional[Mapping[str, str]] = None
    ) -> List[ProviderProfile]:
        values = dict(os.environ if environment is None else environment)
        grouped: Dict[str, Dict[str, str]] = {}
        for name, value in values.items():
            if not name.startswith(PROVIDER_PREFIX):
                continue
            remainder = name[len(PROVIDER_PREFIX) :]
            for field in PROFILE_FIELDS:
                suffix = f"_{field}"
                if remainder.endswith(suffix):
                    raw_provider_id = remainder[: -len(suffix)]
                    if raw_provider_id:
                        provider_id = _normalize_provider_id(raw_provider_id)
                        grouped.setdefault(provider_id, {})[field] = value
                    break

        return [
            cls._profile_from_values(provider_id, profile_values)
            for provider_id, profile_values in sorted(grouped.items())
        ]

    @staticmethod
    def _profile_from_values(
        provider_id: str, values: Mapping[str, str]
    ) -> ProviderProfile:
        # Legacy environment fields adapt into the same schema as providers.yaml.
        from pydantic import ValidationError

        from llmperf_backend.provider_files import ProviderFileProfile

        try:
            cache_ttl_seconds = int(values.get("TTL", "300"))
        except ValueError:
            raise ProviderConfigError(
                f"Provider {provider_id!r} TTL must be an integer"
            ) from None
        discovery_fields: Dict[str, object] = {
            "path": values.get("PATH", "/models").strip() or "/models",
            "cache_ttl_seconds": cache_ttl_seconds,
        }
        if "DISCOVERY" in values:
            discovery_fields["mode"] = values["DISCOVERY"].strip().lower()
        profile_fields: Dict[str, object] = {
            "adapter": values.get("ADAPTER", "openai").strip().lower(),
            "base_url": values.get("URL", ""),
            "api_key": values.get("KEY", ""),
            "models": [
                model.strip()
                for model in values.get("MODELS", "").split(",")
                if model.strip()
            ],
            "discovery": discovery_fields,
        }
        for legacy_field, document_field in (
            ("URLVAR", "worker_api_base_env"),
            ("KEYVAR", "worker_api_key_env"),
        ):
            if legacy_field in values:
                profile_fields[document_field] = values[legacy_field].strip()
        try:
            definition = ProviderFileProfile.model_validate(profile_fields)
        except ValidationError:
            raise ProviderConfigError(
                f"Invalid Provider fields for {provider_id!r}"
            ) from None
        return definition.resolve(provider_id, {})

    def get(self, provider_id: str) -> Optional[ProviderProfile]:
        with self._lock:
            return self._profiles.get(_normalize_provider_id(provider_id))

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def profile_snapshot(self, provider_id: str) -> Tuple[ProviderProfile, int]:
        normalized = _normalize_provider_id(provider_id)
        with self._lock:
            profile = self._profiles.get(normalized)
            if profile is None:
                raise ProviderConfigError(f"Unknown provider profile: {provider_id}")
            return profile, self._generation

    def require(self, provider_id: str) -> ProviderProfile:
        profile = self.get(provider_id)
        if profile is None:
            raise ProviderConfigError(f"Unknown provider profile: {provider_id}")
        return profile

    def list_public(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                self._profiles[provider_id].public_dict()
                for provider_id in sorted(self._profiles)
            ]

    def public_document(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "loaded_at": self._loaded_at.isoformat(),
                "configuration_error": self._configuration_error,
                "items": [
                    self._profiles[provider_id].public_dict()
                    for provider_id in sorted(self._profiles)
                ],
            }

    def reload(self) -> Dict[str, Any]:
        """Validate a Provider-only candidate before one atomic registry swap."""

        if self._reload_loader is None and self._profile_loader is None:
            raise ProviderConfigError(
                "This Provider Registry has no reloadable configuration source"
            )
        try:
            if self._profile_loader is not None:
                candidate = type(self)(self._profile_loader())
            else:
                assert self._reload_loader is not None
                environment = self._reload_loader()
                candidate = type(self).from_environment(environment)
        except ProviderConfigError:
            with self._lock:
                self._configuration_error = CONFIGURATION_ERROR
            raise
        except Exception:
            with self._lock:
                self._configuration_error = CONFIGURATION_ERROR
            raise ProviderConfigError(CONFIGURATION_ERROR) from None

        with self._lock:
            previous = self._profiles
            current = candidate._profiles
            previous_ids = set(previous)
            current_ids = set(current)
            changed = sorted(
                provider_id
                for provider_id in previous_ids & current_ids
                if previous[provider_id] != current[provider_id]
            )
            self._profiles = current
            self._configuration_error = None
            self._credential_environment_names.update(
                name
                for profile in current.values()
                for name in (
                    profile.api_base_env,
                    profile.api_key_env,
                    profile.credential_source_env,
                )
                if name is not None
            )
            self._generation += 1
            self._loaded_at = _utcnow()
            document = self.public_document()
            document.update(
                {
                    "reloaded": True,
                    "changes": {
                        "added": sorted(current_ids - previous_ids),
                        "updated": changed,
                        "removed": sorted(previous_ids - current_ids),
                    },
                }
            )
            return document

    def resolve_benchmark(self, benchmark: Mapping[str, Any]) -> Dict[str, Any]:
        resolved = dict(benchmark)
        profile = self.require(str(resolved["provider"]))
        resolved["provider"] = profile.provider_id
        resolved["adapter"] = profile.adapter
        model = str(resolved.get("model") or "")
        if not model:
            raise ProviderConfigError("A benchmark model must be selected")
        if profile.discovery == "static" and model not in profile.static_models:
            visible_models = ", ".join(profile.static_models[:10])
            if len(profile.static_models) > 10:
                visible_models += f", ... ({len(profile.static_models)} total)"
            raise ProviderConfigError(
                f"Model {model!r} is not configured for static Provider "
                f"{profile.provider_id!r}; configured models: {visible_models}"
            )
        return resolved

    def worker_environment(
        self, provider_id: str, base_environment: Mapping[str, str]
    ) -> Dict[str, str]:
        with self._lock:
            profile = self._profiles.get(_normalize_provider_id(provider_id))
            if profile is None:
                raise ProviderConfigError(f"Unknown provider profile: {provider_id}")
            credential_environment_names = tuple(self._credential_environment_names)
        environment = {
            name: value
            for name, value in base_environment.items()
            if not name.startswith(PROVIDER_PREFIX)
            and name not in {"LLMPERF_REQUEST_API_BASE", "LLMPERF_REQUEST_API_KEY"}
        }
        for environment_name in credential_environment_names:
            environment.pop(environment_name, None)
        if profile.api_base:
            environment[profile.api_base_env] = profile.api_base
            environment["LLMPERF_REQUEST_API_BASE"] = profile.api_base
        if profile.api_key:
            environment[profile.api_key_env] = profile.api_key
            environment["LLMPERF_REQUEST_API_KEY"] = profile.api_key
        return environment


@dataclass(frozen=True)
class _ModelCacheEntry:
    models: Tuple[str, ...]
    fetched_at: datetime
    expires_at: datetime
    source: str
    registry_generation: int


class ProviderModelDiscovery:
    """Discover provider models with a bounded in-memory TTL cache."""

    def __init__(self, registry: ProviderRegistry, timeout_seconds: float = 10.0):
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, _ModelCacheEntry] = {}
        self._lock = threading.RLock()

    async def models(self, provider_id: str, refresh: bool = False) -> Dict[str, Any]:
        profile, registry_generation = self.registry.profile_snapshot(provider_id)
        now = _utcnow()
        with self._lock:
            cached = self._cache.get(profile.provider_id)
            if (
                cached is not None
                and cached.registry_generation == registry_generation
                and not refresh
                and now < cached.expires_at
            ):
                return self._response(profile, cached, cached=True)

        if profile.discovery == "disabled":
            raise ProviderDiscoveryError(
                f"Model discovery is disabled for provider {profile.provider_id!r}"
            )
        if profile.discovery == "static":
            models = profile.static_models
            source = "static"
        else:
            loop = asyncio.get_running_loop()
            models = await loop.run_in_executor(
                None, self._fetch_openai_models, profile
            )
            source = "remote"

        fetched_at = _utcnow()
        entry = _ModelCacheEntry(
            models=tuple(sorted(set(models))),
            fetched_at=fetched_at,
            expires_at=fetched_at + timedelta(seconds=profile.model_cache_ttl_seconds),
            source=source,
            registry_generation=registry_generation,
        )
        with self._lock:
            self._cache[profile.provider_id] = entry
        return self._response(profile, entry, cached=False)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _fetch_openai_models(self, profile: ProviderProfile) -> Tuple[str, ...]:
        if not profile.api_base:
            raise ProviderDiscoveryError(
                f"Provider {profile.provider_id!r} has no API base URL"
            )
        url = f"{profile.api_base}{profile.models_path}"
        headers = {"Accept": "application/json"}
        if profile.api_key:
            headers["Authorization"] = f"Bearer {profile.api_key}"
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                document = json.loads(response.read())
        except HTTPError as exc:
            raise ProviderDiscoveryError(
                f"Provider {profile.provider_id!r} returned HTTP {exc.code}"
            ) from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise ProviderDiscoveryError(
                f"Unable to discover models for provider "
                f"{profile.provider_id!r}: {reason}"
            ) from exc
        data = document.get("data") if isinstance(document, dict) else None
        if not isinstance(data, list):
            raise ProviderDiscoveryError(
                f"Provider {profile.provider_id!r} returned an unsupported "
                "model document"
            )
        models = tuple(
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        )
        if not models:
            raise ProviderDiscoveryError(
                f"Provider {profile.provider_id!r} returned no model IDs"
            )
        return models

    @staticmethod
    def _response(
        profile: ProviderProfile, entry: _ModelCacheEntry, cached: bool
    ) -> Dict[str, Any]:
        return {
            "provider": profile.provider_id,
            "models": list(entry.models),
            "source": entry.source,
            "cached": cached,
            "fetched_at": entry.fetched_at.isoformat(),
            "expires_at": entry.expires_at.isoformat(),
        }

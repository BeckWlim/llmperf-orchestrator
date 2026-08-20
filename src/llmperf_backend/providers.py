"""Server-owned provider profiles, worker credential injection, and model discovery."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import re
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROVIDER_PREFIX = "LLMPERF_PROVIDER_"
SUPPORTED_ADAPTERS = {"openai", "anthropic", "litellm", "sagemaker", "vertexai"}
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
    api_key: str
    api_base_env: str
    api_key_env: str
    discovery: str
    models_path: str
    static_models: Tuple[str, ...]
    model_cache_ttl_seconds: int

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
    ):
        self._profiles = {profile.provider_id: profile for profile in profiles}
        if not self._profiles:
            raise ProviderConfigError("At least one provider profile is required")
        self._reload_loader = reload_loader
        self._lock = threading.RLock()
        self._generation = 0
        self._loaded_at = _utcnow()
        self._credential_environment_names = {
            name
            for profile in self._profiles.values()
            for name in (profile.api_base_env, profile.api_key_env)
        }

    @classmethod
    def from_environment(
        cls,
        environment: Optional[Mapping[str, str]] = None,
        *,
        reload_loader: Optional[Callable[[], Mapping[str, str]]] = None,
    ) -> "ProviderRegistry":
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

        return cls(
            [
                cls._profile_from_values(provider_id, profile_values)
                for provider_id, profile_values in sorted(grouped.items())
            ],
            reload_loader=reload_loader,
        )

    @staticmethod
    def _profile_from_values(
        provider_id: str, values: Mapping[str, str]
    ) -> ProviderProfile:
        adapter = values.get("ADAPTER", "openai").strip().lower()
        if adapter not in SUPPORTED_ADAPTERS:
            raise ProviderConfigError(
                f"Provider {provider_id!r} has unsupported ADAPTER {adapter!r}"
            )
        api_base = values.get("URL", "").strip().rstrip("/")
        _validate_api_base(provider_id, api_base)
        static_models = tuple(
            model.strip()
            for model in values.get("MODELS", "").split(",")
            if model.strip()
        )
        if not api_base and not static_models:
            raise ProviderConfigError(
                f"Provider {provider_id!r} requires URL or MODELS"
            )
        inferred_discovery = (
            "static" if static_models else "openai" if api_base else "disabled"
        )
        discovery = values.get("DISCOVERY", inferred_discovery)
        discovery = discovery.strip().lower()
        if discovery not in {"openai", "static", "disabled"}:
            raise ProviderConfigError(
                f"Provider {provider_id!r} DISCOVERY must be openai, static, "
                "or disabled"
            )
        try:
            cache_ttl = int(values.get("TTL", "300"))
        except ValueError as exc:
            raise ProviderConfigError(
                f"Provider {provider_id!r} TTL must be an integer"
            ) from exc
        if cache_ttl < 0 or cache_ttl > 86400:
            raise ProviderConfigError(
                f"Provider {provider_id!r} model cache TTL must be between 0 and 86400"
            )
        if discovery == "static" and not static_models:
            raise ProviderConfigError(
                f"Provider {provider_id!r} static discovery requires MODELS"
            )
        models_path = values.get("PATH", "/models").strip() or "/models"
        if not models_path.startswith("/"):
            models_path = f"/{models_path}"
        api_base_env = _validate_environment_name(
            provider_id,
            "URLVAR",
            values.get(
                "URLVAR", DEFAULT_BASE_VARIABLES.get(adapter, "OPENAI_API_BASE")
            ).strip(),
        )
        api_key_env = _validate_environment_name(
            provider_id,
            "KEYVAR",
            values.get(
                "KEYVAR", DEFAULT_KEY_VARIABLES.get(adapter, "OPENAI_API_KEY")
            ).strip(),
        )
        return ProviderProfile(
            provider_id=provider_id,
            adapter=adapter,
            api_base=api_base,
            api_key=values.get("KEY", ""),
            api_base_env=api_base_env,
            api_key_env=api_key_env,
            discovery=discovery,
            models_path=models_path,
            static_models=static_models,
            model_cache_ttl_seconds=cache_ttl,
        )

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
                "items": [
                    self._profiles[provider_id].public_dict()
                    for provider_id in sorted(self._profiles)
                ],
            }

    def reload(self) -> Dict[str, Any]:
        """Validate a Provider-only candidate before one atomic registry swap."""

        if self._reload_loader is None:
            raise ProviderConfigError(
                "This Provider Registry has no reloadable configuration source"
            )
        try:
            environment = self._reload_loader()
            candidate = type(self).from_environment(environment)
        except ProviderConfigError:
            raise
        except Exception as exc:
            raise ProviderConfigError(
                f"Unable to reload Provider Profiles: {exc}"
            ) from exc

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
            self._credential_environment_names.update(
                name
                for profile in current.values()
                for name in (profile.api_base_env, profile.api_key_env)
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
        }
        for environment_name in credential_environment_names:
            environment.pop(environment_name, None)
        if profile.api_base:
            environment[profile.api_base_env] = profile.api_base
        if profile.api_key:
            environment[profile.api_key_env] = profile.api_key
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

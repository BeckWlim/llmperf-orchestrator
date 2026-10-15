"""Strict YAML/JSON Provider documents with optional inline or environment credentials."""

import json
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
import yaml

from llmperf.version import ProtocolVersion
from llmperf_backend.providers import (
    DEFAULT_BASE_VARIABLES,
    DEFAULT_KEY_VARIABLES,
    ProviderConfigError,
    ProviderProfile,
    _normalize_provider_id,
    _validate_api_base,
    _validate_environment_name,
)


class _StrictDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProviderDiscoveryConfig(_StrictDocument):
    mode: Optional[Literal["openai", "static", "disabled"]] = None
    path: str = "/models"
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)


class ProviderFileProfile(_StrictDocument):
    adapter: Literal["openai", "anthropic", "litellm", "sagemaker", "vertexai"] = (
        "openai"
    )
    base_url: str = ""
    api_key: Optional[SecretStr] = None
    api_key_env: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    discovery: ProviderDiscoveryConfig = Field(default_factory=ProviderDiscoveryConfig)
    worker_api_base_env: Optional[str] = None
    worker_api_key_env: Optional[str] = None

    @model_validator(mode="after")
    def validate_credentials(self) -> "ProviderFileProfile":
        if self.api_key is not None and self.api_key_env is not None:
            raise ValueError("Specify api_key or api_key_env, not both")
        return self

    def resolve(
        self, provider_id: str, environment: Mapping[str, str]
    ) -> ProviderProfile:
        api_base = self.base_url.strip().rstrip("/")
        _validate_api_base(provider_id, api_base)
        model_ids = tuple(model.strip() for model in self.models)
        if any(not model for model in model_ids):
            raise ProviderConfigError(
                f"Provider {provider_id!r} models must not contain empty IDs"
            )
        if not api_base and not model_ids:
            raise ProviderConfigError(
                f"Provider {provider_id!r} requires base_url or models"
            )
        discovery_mode = self.discovery.mode or (
            "static" if model_ids else "openai" if api_base else "disabled"
        )
        if discovery_mode == "static" and not model_ids:
            raise ProviderConfigError(
                f"Provider {provider_id!r} static discovery requires models"
            )
        source_env = self.api_key_env
        if source_env is not None:
            _validate_environment_name(provider_id, "api_key_env", source_env)
        if self.api_key is not None:
            raw_key = self.api_key.get_secret_value()
        elif source_env is not None:
            raw_key = environment.get(source_env, "")
        else:
            raw_key = ""
        base_env = _validate_environment_name(
            provider_id,
            "worker_api_base_env",
            (
                self.worker_api_base_env
                if self.worker_api_base_env is not None
                else DEFAULT_BASE_VARIABLES.get(self.adapter, "OPENAI_API_BASE")
            ),
        )
        key_env = _validate_environment_name(
            provider_id,
            "worker_api_key_env",
            (
                self.worker_api_key_env
                if self.worker_api_key_env is not None
                else DEFAULT_KEY_VARIABLES.get(self.adapter, "OPENAI_API_KEY")
            ),
        )
        discovery_path = self.discovery.path.strip() or "/models"
        models_path = "/" + discovery_path.lstrip("/")
        return ProviderProfile(
            provider_id=provider_id,
            adapter=self.adapter,
            api_base=api_base,
            api_key=raw_key.strip(),
            api_base_env=base_env,
            api_key_env=key_env,
            discovery=discovery_mode,
            models_path=models_path,
            static_models=model_ids,
            model_cache_ttl_seconds=self.discovery.cache_ttl_seconds,
            credential_source_env=source_env,
        )


class ProviderFileDocument(_StrictDocument):
    version: ProtocolVersion
    providers: Dict[str, ProviderFileProfile] = Field(min_length=1)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML fields rather than silently replacing a profile."""


def _yaml_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ProviderConfigError(
                "Provider document contains a non-string or duplicate key"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def _json_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProviderConfigError("Provider document contains a duplicate key")
        result[key] = value
    return result


def load_provider_file(
    path: Path, environment: Mapping[str, str]
) -> List[ProviderProfile]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProviderConfigError(f"Unable to read Provider file {path}") from exc
    try:
        if path.suffix.lower() == ".json":
            raw_document = json.loads(content, object_pairs_hook=_json_mapping)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            raw_document = yaml.load(content, Loader=_UniqueKeyLoader)
        else:
            raise ProviderConfigError("Provider file must use .yaml, .yml, or .json")
        document = ProviderFileDocument.model_validate(raw_document)
    except (yaml.YAMLError, json.JSONDecodeError):
        # Parser context can contain credentials accidentally pasted into a file.
        raise ProviderConfigError(
            f"Invalid Provider document syntax in {path}"
        ) from None
    except ValidationError as exc:
        locations = [
            ".".join(str(part) for part in error["loc"]) for error in exc.errors()
        ]
        raise ProviderConfigError(
            f"Invalid Provider fields in {path}: {', '.join(locations)}"
        ) from None
    profiles: Dict[str, ProviderProfile] = {}
    for raw_id, definition in document.providers.items():
        provider_id = _normalize_provider_id(raw_id)
        if provider_id in profiles:
            raise ProviderConfigError(
                f"Duplicate Provider ID after normalization: {provider_id!r}"
            )
        profiles[provider_id] = definition.resolve(provider_id, environment)
    return [profiles[provider_id] for provider_id in sorted(profiles)]

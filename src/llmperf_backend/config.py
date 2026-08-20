"""Backend environment loading, YAML validation, and atomic configuration state."""

import copy
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import load_dotenv
import yaml
from pydantic import ValidationError

from llmperf.user_config import backend_environment_path, read_environment_file
from llmperf_backend.models import AppConfig, dump_model, validate_app_config

CONFIG_PATH = "LLMPERF_BACKEND_CONFIG"
ENV_FILE = "LLMPERF_ENV_FILE"
PROVIDER_PREFIX = "LLMPERF_PROVIDER_"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)

# Capture the real process environment before dotenv values are added. Provider
# reloads can then preserve exported-value precedence without retaining stale file
# values from an older version of the environment file.
_PROCESS_ENVIRONMENT = dict(os.environ)


class ConfigError(ValueError):
    """Raised when a YAML configuration cannot be loaded or validated."""


def resolve_environment_path(path: Optional[Path] = None) -> Path:
    """Resolve an explicit path, override, or canonical user config path."""

    if path is not None:
        environment_path = Path(path)
        return environment_path.expanduser().resolve()
    configured_path = os.environ.get(ENV_FILE)
    if configured_path:
        environment_path = Path(configured_path)
        return environment_path.expanduser().resolve()
    return backend_environment_path()


def load_environment(
    path: Optional[Path] = None,
    *,
    override: bool = False,
) -> Optional[Path]:
    """Load the optional Backend dotenv file with explicit-path validation."""

    environment_path = resolve_environment_path(path)
    explicitly_selected = path is not None or bool(os.environ.get(ENV_FILE))
    if not environment_path.is_file():
        if explicitly_selected:
            raise RuntimeError(f"Environment file does not exist: {environment_path}")
        return None
    load_dotenv(dotenv_path=environment_path, override=override)
    return environment_path


def load_provider_environment(
    path: Optional[Path] = None,
    *,
    process_environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Read reloadable Provider-only settings with process-value precedence."""

    environment_path = resolve_environment_path(path)
    explicitly_selected = path is not None or bool(os.environ.get(ENV_FILE))
    if not environment_path.is_file():
        if explicitly_selected:
            raise RuntimeError(f"Environment file does not exist: {environment_path}")
        file_values: Mapping[str, str] = {}
    else:
        file_values = read_environment_file(environment_path)

    effective_environment = {
        name: value
        for name, value in file_values.items()
        if name.startswith(PROVIDER_PREFIX)
    }
    process_values = (
        _PROCESS_ENVIRONMENT if process_environment is None else process_environment
    )
    effective_environment.update(
        {
            name: value
            for name, value in process_values.items()
            if name.startswith(PROVIDER_PREFIX)
        }
    )
    return effective_environment


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name = match.group("name")
        if name in os.environ:
            return os.environ[name]
        default = match.group("default")
        if default is not None:
            return default
        raise ConfigError(f"Environment variable {name!r} is not set")

    return _ENV_PATTERN.sub(replace, value)


def _parse_yaml(content: str, source: str) -> Dict[str, Any]:
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {source}: {exc}") from exc

    if parsed is None:
        raise ConfigError(f"Configuration {source} is empty")
    if not isinstance(parsed, dict):
        raise ConfigError(
            f"Configuration {source} must contain a YAML mapping at its root"
        )
    return _expand_environment(parsed)


def load_config_text(content: str, source: str = "<request>") -> AppConfig:
    """Safely parse and validate YAML text without changing active state."""

    parsed = _parse_yaml(content, source)
    try:
        return validate_app_config(parsed)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {source}: {exc}") from exc


def resolve_config_path(path: Optional[Path] = None) -> Path:
    # Load before consulting LLMPERF_BACKEND_CONFIG. Existing process variables
    # retain precedence because dotenv loading uses override=False.
    load_environment()
    if path is not None:
        config_path = Path(path)
        return config_path.expanduser().resolve()
    configured_path = os.environ.get(CONFIG_PATH)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load an application configuration from a YAML file."""

    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration {config_path}: {exc}") from exc
    return load_config_text(content, str(config_path))


@dataclass(frozen=True)
class ConfigSnapshot:
    source: str
    loaded_at: str
    generation: int
    config: Dict[str, Any]


class ConfigStore:
    """Thread-safe active configuration with validate-before-swap reloads."""

    def __init__(self, path: Optional[Path] = None):
        self._path = resolve_config_path(path)
        self._lock = threading.RLock()
        self._generation = 0
        self._config = load_config(self._path)
        self._loaded_at = self._now()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @property
    def path(self) -> Path:
        return self._path

    def current(self) -> AppConfig:
        """Return an isolated copy of the validated active configuration."""

        with self._lock:
            model_copy = getattr(self._config, "model_copy", None)
            if model_copy is not None:
                return model_copy(deep=True)
            return self._config.copy(deep=True)

    def snapshot(self) -> ConfigSnapshot:
        with self._lock:
            return ConfigSnapshot(
                source=str(self._path),
                loaded_at=self._loaded_at,
                generation=self._generation,
                config=copy.deepcopy(dump_model(self._config)),
            )

    def reload(self) -> ConfigSnapshot:
        candidate = load_config(self._path)
        with self._lock:
            self._config = candidate
            self._loaded_at = self._now()
            self._generation += 1
            return self.snapshot()

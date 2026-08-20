"""Load backend environment variables from an optional dotenv file."""

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from dotenv import load_dotenv

from llmperf.user_config import backend_environment_path, read_environment_file

ENV_FILE = "LLMPERF_ENV_FILE"
PROVIDER_PREFIX = "LLMPERF_PROVIDER_"

# Capture the real process environment before ``load_environment`` adds dotenv
# values to ``os.environ``. Reloads can then preserve the documented precedence
# (exported process variables over the mutable dotenv file) without retaining
# stale values that were loaded from an older version of that file.
_PROCESS_ENVIRONMENT = dict(os.environ)


def resolve_environment_path(path: Optional[Path] = None) -> Path:
    """Resolve an explicit path, override, or the canonical user config path."""

    if path is not None:
        return Path(path).expanduser().resolve()
    configured_path = os.environ.get(ENV_FILE)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return backend_environment_path()


def load_environment(
    path: Optional[Path] = None,
    *,
    override: bool = False,
) -> Optional[Path]:
    """Load a dotenv file, preserving process environment values by default.

    A missing default user file is normal. An explicitly selected file must exist so
    a misspelled production configuration does not fail silently.
    """

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
    """Read the reloadable Provider-only environment with safe precedence.

    This intentionally excludes every non-Provider setting. A Provider reload
    therefore cannot change database, Scheduler, Ray, authentication, listener,
    or other Backend runtime configuration.
    """

    environment_path = resolve_environment_path(path)
    explicitly_selected = path is not None or bool(os.environ.get(ENV_FILE))
    if not environment_path.is_file():
        if explicitly_selected:
            raise RuntimeError(f"Environment file does not exist: {environment_path}")
        file_values: Mapping[str, str] = {}
    else:
        file_values = read_environment_file(environment_path)

    effective = {
        name: value
        for name, value in file_values.items()
        if name.startswith(PROVIDER_PREFIX)
    }
    process_values = (
        _PROCESS_ENVIRONMENT if process_environment is None else process_environment
    )
    effective.update(
        {
            name: value
            for name, value in process_values.items()
            if name.startswith(PROVIDER_PREFIX)
        }
    )
    return effective

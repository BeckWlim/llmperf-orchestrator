"""Load backend environment variables from an optional dotenv file."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


ENV_FILE_ENV = "LLMPERF_ENV_FILE"
DEFAULT_ENV_FILE = ".env"


def resolve_environment_path(path: Optional[Path] = None) -> Path:
    """Resolve an explicit path, LLMPERF_ENV_FILE, or the working-directory .env."""

    if path is not None:
        return Path(path).expanduser().resolve()
    configured_path = os.environ.get(ENV_FILE_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (Path.cwd() / DEFAULT_ENV_FILE).resolve()


def load_environment(
    path: Optional[Path] = None,
    *,
    override: bool = False,
) -> Optional[Path]:
    """Load a dotenv file, preserving process environment values by default.

    A missing default ``.env`` is normal. An explicitly selected file must exist so
    a misspelled production configuration does not fail silently.
    """

    environment_path = resolve_environment_path(path)
    explicitly_selected = path is not None or bool(os.environ.get(ENV_FILE_ENV))
    if not environment_path.is_file():
        if explicitly_selected:
            raise RuntimeError(f"Environment file does not exist: {environment_path}")
        return None
    load_dotenv(dotenv_path=environment_path, override=override)
    return environment_path

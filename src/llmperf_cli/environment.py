"""Load llmperfctl-only settings from an optional user dotenv file."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from llmperf.user_config import cli_environment_path


CLI_ENV_FILE = "LLMPERF_CLI_ENV_FILE"


def resolve_cli_environment_path(path: Optional[Path] = None) -> Path:
    """Resolve an explicit CLI dotenv path or the per-user default."""

    if path is not None:
        return Path(path).expanduser().resolve()
    configured_path = os.environ.get(CLI_ENV_FILE)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return cli_environment_path()


def load_cli_environment(
    path: Optional[Path] = None,
    *,
    override: bool = False,
) -> Optional[Path]:
    """Load CLI settings while preserving exported process variables by default.

    A missing default ``~/.config/llmperf/cli.env`` is normal. An explicitly
    selected file must exist so a misspelled path cannot silently use CLI defaults.
    """

    environment_path = resolve_cli_environment_path(path)
    explicitly_selected = path is not None or bool(os.environ.get(CLI_ENV_FILE))
    if not environment_path.is_file():
        if explicitly_selected:
            raise RuntimeError(
                f"CLI environment file does not exist: {environment_path}"
            )
        return None
    load_dotenv(dotenv_path=environment_path, override=override)
    return environment_path

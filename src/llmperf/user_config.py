"""Secure user-level configuration files shared by LLMPerf commands."""

import os
from pathlib import Path
import re
import tempfile
from typing import Dict, Optional

from dotenv import dotenv_values


CONFIG_DIRECTORY_NAME = "llmperf"
BACKEND_ENV_FILENAME = "backend.env"
CLI_ENV_FILENAME = "cli.env"
_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SENSITIVE_FRAGMENTS = ("KEY", "PASSWORD", "SECRET", "TOKEN")


class UserConfigError(ValueError):
    """Raised when a persistent user configuration operation is invalid."""


def user_config_directory() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    root = (
        Path(configured).expanduser() if configured else Path("~/.config").expanduser()
    )
    return (root / CONFIG_DIRECTORY_NAME).resolve()


def backend_environment_path() -> Path:
    return user_config_directory() / BACKEND_ENV_FILENAME


def cli_environment_path() -> Path:
    return user_config_directory() / CLI_ENV_FILENAME


def read_environment_file(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    try:
        parsed = dotenv_values(path, interpolate=False)
    except OSError as exc:
        raise UserConfigError(f"Unable to read configuration {path}: {exc}") from exc
    return {str(name): str(value or "") for name, value in parsed.items()}


def set_environment_value(path: Path, name: str, value: str) -> None:
    _validate_name(name)
    _validate_value(value)
    values = read_environment_file(path)
    values[name] = value
    _write_environment_file(path, values)


def unset_environment_value(path: Path, name: str) -> bool:
    _validate_name(name)
    values = read_environment_file(path)
    removed = name in values
    if removed:
        values.pop(name)
        _write_environment_file(path, values)
    return removed


def display_environment_value(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    upper_name = name.upper()
    if upper_name == "DATABASE_URL" or any(
        fragment in upper_name for fragment in _SENSITIVE_FRAGMENTS
    ):
        return "<redacted>"
    return value


def _validate_name(name: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise UserConfigError(
            "Configuration name must contain only uppercase letters, digits, and "
            "underscores, and cannot start with a digit"
        )


def _validate_value(value: str) -> None:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise UserConfigError("Configuration values must be single-line text")


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _write_environment_file(path: Path, values: Dict[str, str]) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent), text=True
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for name in sorted(values):
                    stream.write(f"{name}={_quote(values[name])}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()
    except OSError as exc:
        raise UserConfigError(f"Unable to write configuration {path}: {exc}") from exc

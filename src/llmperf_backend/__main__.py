"""Command-line entry point for the LLMPerf FastAPI backend."""

import argparse
import json
import os
import sys
from typing import Any, Dict

from llmperf.logging import configure_logging
from llmperf.user_config import (
    UserConfigError,
    backend_environment_path,
    display_environment_value,
    read_environment_file,
    set_environment_value,
    unset_environment_value,
)
from llmperf_backend.config import load_config
from llmperf_backend.outbound import (
    normalize_outbound_environment,
    configure_ray_direct,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmperf-backend",
        description="Run the LLMPerf backend or manage its persistent configuration.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    config = commands.add_parser(
        "config",
        help="Manage persistent backend environment settings",
    )
    config_commands = config.add_subparsers(
        dest="config_command", required=True, metavar="COMMAND"
    )

    config_set = config_commands.add_parser(
        "set", help="Persist one backend environment setting"
    )
    config_set.add_argument("name", metavar="NAME")
    config_set.add_argument("value", metavar="VALUE", nargs="?")
    config_set.add_argument(
        "--stdin",
        action="store_true",
        help="read VALUE from standard input (recommended for secrets)",
    )

    config_get = config_commands.add_parser(
        "get", help="Show one persisted setting, redacting secrets"
    )
    config_get.add_argument("name", metavar="NAME")

    config_unset = config_commands.add_parser(
        "unset", help="Remove one persisted setting"
    )
    config_unset.add_argument("name", metavar="NAME")

    config_commands.add_parser(
        "list", help="List persisted settings with secrets redacted"
    )
    config_commands.add_parser("path", help="Print the persistent config file path")
    return parser


def execute_config(arguments: argparse.Namespace) -> Dict[str, Any]:
    path = backend_environment_path()
    if arguments.config_command == "path":
        return {"path": str(path), "exists": path.is_file()}
    if arguments.config_command == "set":
        use_stdin = bool(getattr(arguments, "stdin", False))
        if use_stdin == (arguments.value is not None):
            raise UserConfigError("Provide exactly one of VALUE or --stdin")
        value = sys.stdin.read().rstrip("\r\n") if use_stdin else arguments.value
        set_environment_value(path, arguments.name, value)
        provider_setting = arguments.name.startswith("LLMPERF_PROVIDER_")
        return {
            "name": arguments.name,
            "value": display_environment_value(arguments.name, value),
            "path": str(path),
            "restart_required": not provider_setting,
            "provider_reload_required": provider_setting,
        }
    if arguments.config_command == "unset":
        removed = unset_environment_value(path, arguments.name)
        provider_setting = arguments.name.startswith("LLMPERF_PROVIDER_")
        return {
            "name": arguments.name,
            "removed": removed,
            "path": str(path),
            "restart_required": removed and not provider_setting,
            "provider_reload_required": removed and provider_setting,
        }
    values = read_environment_file(path)
    if arguments.config_command == "get":
        value = values.get(arguments.name)
        if value is None:
            raise UserConfigError(
                f"Configuration setting is not defined: {arguments.name}"
            )
        return {
            "name": arguments.name,
            "value": display_environment_value(arguments.name, value),
            "path": str(path),
        }
    return {
        "path": str(path),
        "items": {
            name: display_environment_value(name, value)
            for name, value in sorted(values.items())
        },
    }


def serve() -> None:
    """Load persistent settings and run the API service."""

    config = load_config()
    normalize_outbound_environment(os.environ)
    configure_ray_direct(os.environ)
    import uvicorn

    configure_logging(
        config.server.log_level,
        color=os.environ.get("LLMPERF_LOG_COLOR", "auto").lower(),
    )
    uvicorn.run(
        "llmperf_backend.app:create_app",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        workers=config.server.workers,
        reload=config.server.reload,
        log_config=None,
    )


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "config":
        try:
            print(json.dumps(execute_config(arguments), indent=2))
        except UserConfigError as exc:
            parser.exit(1, f"error: {exc}\n")
        return
    serve()


if __name__ == "__main__":
    main()

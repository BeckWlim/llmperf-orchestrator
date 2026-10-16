"""Stable command entry point for python -m llmperf_cli."""

from llmperf_cli.commands.main import main

if __name__ == "__main__":
    raise SystemExit(main())

import io
import stat
import sys

import pytest

import llmperf_cli.__main__ as cli_main
from llmperf.user_config import cli_environment_path, read_environment_file
from llmperf_cli.environment import (
    load_cli_environment,
    resolve_cli_environment_path,
)


def test_default_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("LLMPERF_CLI_ENV_FILE", raising=False)
    monkeypatch.delenv("LLMPERF_URL", raising=False)
    path = cli_environment_path()
    path.parent.mkdir(parents=True)
    path.write_text("LLMPERF_URL=http://cli-env.example:12666\n", encoding="utf-8")

    loaded = load_cli_environment()

    assert resolve_cli_environment_path() == path
    assert loaded == path
    assert cli_main.build_parser().parse_args(["health"]).url == (
        "http://cli-env.example:12666"
    )
    monkeypatch.delenv("LLMPERF_URL", raising=False)


def test_process_precedence(tmp_path, monkeypatch):
    path = tmp_path / "cli.env"
    path.write_text("LLMPERF_URL=http://from-file.example\n", encoding="utf-8")
    monkeypatch.setenv("LLMPERF_URL", "http://from-process.example")

    assert load_cli_environment(path) == path
    assert cli_main.build_parser().parse_args(["health"]).url == (
        "http://from-process.example"
    )
    assert (
        cli_main.build_parser()
        .parse_args(["--url", "http://from-argument.example", "health"])
        .url
        == "http://from-argument.example"
    )


def test_explicit_missing(tmp_path, monkeypatch):
    missing = tmp_path / "missing.env"
    monkeypatch.setenv("LLMPERF_CLI_ENV_FILE", str(missing))

    with pytest.raises(RuntimeError, match="CLI environment file does not exist"):
        load_cli_environment()


def test_main_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("LLMPERF_CLI_ENV_FILE", raising=False)
    monkeypatch.delenv("LLMPERF_URL", raising=False)
    monkeypatch.setenv("LLMPERF_SSH_DIR", str(tmp_path / "missing-ssh"))
    path = cli_environment_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        "LLMPERF_URL=http://main-env.example:12666\n",
        encoding="utf-8",
    )
    observed = {}

    def fake_execute(client, arguments):
        observed["url"] = client.base_url
        observed["command"] = arguments.command
        return {"status": "ok"}

    monkeypatch.setattr(cli_main, "execute", fake_execute)
    monkeypatch.setattr(sys, "argv", ["llmperfctl", "health"])

    cli_main.main()

    assert observed == {
        "url": "http://main-env.example:12666",
        "command": "health",
    }
    assert '"status": "ok"' in capsys.readouterr().out
    monkeypatch.delenv("LLMPERF_URL", raising=False)


def test_config_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = cli_environment_path()

    configured = cli_main.execute_cli_config(
        cli_main.build_parser().parse_args(
            ["config", "set", "LLMPERF_URL", "http://backend.example:12666"]
        )
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("secret-token\n"))
    cli_main.execute_cli_config(
        cli_main.build_parser().parse_args(
            ["config", "set", "LLMPERF_TOKEN", "--stdin"]
        )
    )

    assert configured["effective_next_run"] is True
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_environment_file(path) == {
        "LLMPERF_TOKEN": "secret-token",
        "LLMPERF_URL": "http://backend.example:12666",
    }
    listed = cli_main.execute_cli_config(
        cli_main.build_parser().parse_args(["config", "list"])
    )
    assert listed["items"] == {
        "LLMPERF_TOKEN": "<redacted>",
        "LLMPERF_URL": "http://backend.example:12666",
    }
    fetched = cli_main.execute_cli_config(
        cli_main.build_parser().parse_args(["config", "get", "LLMPERF_TOKEN"])
    )
    assert fetched["value"] == "<redacted>"
    removed = cli_main.execute_cli_config(
        cli_main.build_parser().parse_args(["config", "unset", "LLMPERF_TOKEN"])
    )
    assert removed["removed"] is True
    assert read_environment_file(path) == {
        "LLMPERF_URL": "http://backend.example:12666"
    }


def test_config_creates(tmp_path, monkeypatch, capsys):
    path = tmp_path / "selected" / "cli.env"
    monkeypatch.setenv("LLMPERF_CLI_ENV_FILE", str(path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["llmperfctl", "config", "set", "LLMPERF_URL", "http://selected.example"],
    )

    cli_main.main()

    assert read_environment_file(path) == {"LLMPERF_URL": "http://selected.example"}
    assert '"effective_next_run": true' in capsys.readouterr().out

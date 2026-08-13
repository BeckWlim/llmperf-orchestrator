from argparse import Namespace
import io
import os
import stat

import pytest

from llmperf.user_config import (
    UserConfigError,
    backend_environment_path,
    read_environment_file,
)
from llmperf_backend.__main__ import build_parser, execute_config
from llmperf_backend.environment import load_environment, resolve_environment_path


def _arguments(*values):
    return build_parser().parse_args(["config", *values])


def test_config_set_get_list_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = backend_environment_path()

    configured = execute_config(_arguments("set", "LLMPERF_SERVER_HOST", "0.0.0.0"))
    execute_config(
        _arguments(
            "set",
            "LLMPERF_PROVIDER_ALIYUN_KEY",
            "secret with 'quote' and ${literal}",
        )
    )

    assert configured["path"] == str(path)
    assert configured["restart_required"] is True
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_environment_file(path) == {
        "LLMPERF_PROVIDER_ALIYUN_KEY": "secret with 'quote' and ${literal}",
        "LLMPERF_SERVER_HOST": "0.0.0.0",
    }
    assert (
        execute_config(_arguments("get", "LLMPERF_PROVIDER_ALIYUN_KEY"))["value"]
        == "<redacted>"
    )
    assert execute_config(_arguments("list"))["items"] == {
        "LLMPERF_PROVIDER_ALIYUN_KEY": "<redacted>",
        "LLMPERF_SERVER_HOST": "0.0.0.0",
    }
    assert execute_config(_arguments("unset", "LLMPERF_SERVER_HOST"))["removed"] is True
    assert "LLMPERF_SERVER_HOST" not in read_environment_file(path)


def test_user_config_is_default_and_process_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = backend_environment_path()
    execute_config(_arguments("set", "DEPLOYMENT_TEST_VALUE", "from-file"))
    monkeypatch.setenv("DEPLOYMENT_TEST_VALUE", "from-process")

    loaded = load_environment()

    assert resolve_environment_path() == path
    assert loaded == path
    assert os.environ["DEPLOYMENT_TEST_VALUE"] == "from-process"


def test_config_set_reads_secret_from_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("stdin-secret\n"))

    result = execute_config(_arguments("set", "LLMPERF_PROVIDER_ALIYUN_KEY", "--stdin"))

    assert result["value"] == "<redacted>"
    assert read_environment_file(backend_environment_path()) == {
        "LLMPERF_PROVIDER_ALIYUN_KEY": "stdin-secret"
    }


@pytest.mark.parametrize("values", [(), ("value", "--stdin")])
def test_config_set_requires_exactly_one_value_source(tmp_path, monkeypatch, values):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.raises(UserConfigError, match="exactly one"):
        execute_config(_arguments("set", "VALID_NAME", *values))


def test_legacy_local_env_is_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.delenv("LLMPERF_ENV_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / ".env"
    legacy.write_text("LEGACY_TEST_VALUE=loaded\n", encoding="utf-8")

    assert resolve_environment_path() == legacy


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("bad-name", "value", "uppercase letters"),
        ("VALID_NAME", "two\nlines", "single-line"),
    ],
)
def test_invalid_config_value(tmp_path, monkeypatch, name, value, message):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.raises(UserConfigError, match=message):
        execute_config(Namespace(config_command="set", name=name, value=value))

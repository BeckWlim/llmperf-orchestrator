from argparse import Namespace
import io
import os
import stat

import pytest

from llmperf.configuration.environment import (
    UserConfigError,
    backend_environment_path,
    read_environment_file,
)
from llmperf_backend.__main__ import build_parser, execute_config
from llmperf_backend.configuration.loader import (
    load_environment,
    resolve_environment_path,
)


def _arguments(*values):
    return build_parser().parse_args(["config", *values])


def test_config_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = backend_environment_path()

    configured = execute_config(_arguments("set", "LLMPERF_SERVER_HOST", "0.0.0.0"))
    path.with_name("providers.yaml").write_text(
        'version: "1.0.0"\nproviders: {aliyun: {models: [model], api_key_env: ALIYUN_API_KEY}}'
    )
    provider_configured = execute_config(
        _arguments(
            "set",
            "ALIYUN_API_KEY",
            "secret with 'quote' and ${literal}",
        )
    )

    assert configured["path"] == str(path)
    assert configured["restart_required"] is True
    assert provider_configured["restart_required"] is False
    assert provider_configured["provider_reload_required"] is True
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_environment_file(path) == {
        "ALIYUN_API_KEY": "secret with 'quote' and ${literal}",
        "LLMPERF_SERVER_HOST": "0.0.0.0",
    }
    assert execute_config(_arguments("get", "ALIYUN_API_KEY"))["value"] == "<redacted>"
    assert execute_config(_arguments("list"))["items"] == {
        "ALIYUN_API_KEY": "<redacted>",
        "LLMPERF_SERVER_HOST": "0.0.0.0",
    }
    assert execute_config(_arguments("unset", "LLMPERF_SERVER_HOST"))["removed"] is True
    assert "LLMPERF_SERVER_HOST" not in read_environment_file(path)


def test_provider_file_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    configured = execute_config(
        _arguments("set", "LLMPERF_PROVIDERS_FILE", "providers.yaml")
    )
    assert configured["provider_reload_required"] is True
    assert configured["restart_required"] is False
    removed = execute_config(_arguments("unset", "LLMPERF_PROVIDERS_FILE"))
    assert removed["provider_reload_required"] is True


def test_unreferenced_provider_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    configured = execute_config(
        _arguments("set", "LLMPERF_PROVIDER_OLD_URL", "http://localhost:8001/v1")
    )
    assert configured["provider_reload_required"] is False


def test_proxy_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    proxy_url = "http://proxy-user:proxy-password@proxy.internal:3128"

    configured = execute_config(_arguments("set", "LLMPERF_PROXY", proxy_url))
    fetched = execute_config(_arguments("get", "LLMPERF_PROXY"))
    listed = execute_config(_arguments("list"))

    assert configured["value"] == "<redacted>"
    assert fetched["value"] == "<redacted>"
    assert listed["items"] == {"LLMPERF_PROXY": "<redacted>"}
    assert read_environment_file(backend_environment_path()) == {
        "LLMPERF_PROXY": proxy_url
    }


def test_env_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = backend_environment_path()
    execute_config(_arguments("set", "DEPLOYMENT_TEST_VALUE", "from-file"))
    monkeypatch.setenv("DEPLOYMENT_TEST_VALUE", "from-process")

    loaded = load_environment()

    assert resolve_environment_path() == path
    assert loaded == path
    assert os.environ["DEPLOYMENT_TEST_VALUE"] == "from-process"


def test_secret_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("stdin-secret\n"))

    result = execute_config(_arguments("set", "ALIYUN_API_KEY", "--stdin"))

    assert result["value"] == "<redacted>"
    assert read_environment_file(backend_environment_path()) == {
        "ALIYUN_API_KEY": "stdin-secret"
    }


@pytest.mark.parametrize("values", [(), ("value", "--stdin")])
def test_value_source_guard(tmp_path, monkeypatch, values):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.raises(UserConfigError, match="exactly one"):
        execute_config(_arguments("set", "VALID_NAME", *values))


def test_no_cwd_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.delenv("LLMPERF_ENV_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("IGNORED_VALUE=ignored\n", encoding="utf-8")

    assert resolve_environment_path() == backend_environment_path()


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

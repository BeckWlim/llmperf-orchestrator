import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from llmperf_backend.outbound import (
    LLMPERF_NO_PROXY,
    LLMPERF_PROXY,
    OutboundConfigurationError,
    RAY_GRPC_ENABLE_HTTP_PROXY,
    STANDARD_NO_PROXY_NAMES,
    STANDARD_PROXY_NAMES,
    configure_outbound_transport,
    configure_ray_direct,
    normalize_outbound_environment,
)


def test_http_proxy(monkeypatch):
    configure = Mock()
    client = Mock()
    client_class = Mock(return_value=client)
    monkeypatch.setattr("llmperf_backend.outbound.set_client_factory", configure)
    monkeypatch.setattr("llmperf_backend.outbound.httpx.Client", client_class)

    policy = configure_outbound_transport({}, "http://proxy.internal:3128")

    factory = configure.call_args.args[0]
    assert factory() is client
    client_class.assert_called_once_with(
        follow_redirects=True,
        timeout=None,
        proxy="http://proxy.internal:3128",
    )
    assert policy.source == "explicit"


def test_http_default(monkeypatch):
    configure = Mock()
    client = Mock()
    client_class = Mock(return_value=client)
    monkeypatch.setattr("llmperf_backend.outbound.set_client_factory", configure)
    monkeypatch.setattr("llmperf_backend.outbound.httpx.Client", client_class)

    configure_outbound_transport({})

    factory = configure.call_args.args[0]
    assert factory() is client
    client_class.assert_called_once_with(follow_redirects=True, timeout=None)


def test_empty_proxy(monkeypatch):
    configure = Mock()
    monkeypatch.setattr("llmperf_backend.outbound.set_client_factory", configure)

    policy = configure_outbound_transport(
        {"HTTPS_PROXY": "http://proxy.internal:3128"}, ""
    )

    assert policy.source == "explicit"
    assert policy.proxy_url is None


def test_proxy_general_mapping():
    proxy_url = "http://proxy.internal:3128"
    environment = {LLMPERF_PROXY: proxy_url}

    policy = normalize_outbound_environment(environment)

    assert policy.source == "llmperf"
    assert policy.proxy_url == proxy_url
    assert set(policy.standard_proxy_names) == set(STANDARD_PROXY_NAMES)
    assert all(environment[name] == proxy_url for name in STANDARD_PROXY_NAMES)


def test_proxy_standard_precedence():
    general_proxy_url = "http://general-proxy.internal:3128"
    https_proxy_url = "http://https-proxy.internal:3128"
    environment = {
        LLMPERF_PROXY: general_proxy_url,
        "HTTPS_PROXY": https_proxy_url,
    }

    normalize_outbound_environment(environment)

    assert environment["HTTPS_PROXY"] == https_proxy_url
    assert environment["https_proxy"] == https_proxy_url
    assert environment["HTTP_PROXY"] == general_proxy_url
    assert environment["ALL_PROXY"] == general_proxy_url


def test_removed_proxy_ignored():
    environment = {"LLMPERF_HUGGINGFACE_PROXY": "http://legacy-proxy.internal:3128"}

    policy = normalize_outbound_environment(environment)

    assert policy.source == "none"
    assert not any(name in environment for name in STANDARD_PROXY_NAMES)


def test_proxy_bypass_mapping():
    bypass_value = "127.0.0.1,localhost,.internal"
    environment = {LLMPERF_NO_PROXY: bypass_value}

    policy = normalize_outbound_environment(environment)

    assert set(policy.no_proxy_names) == set(STANDARD_NO_PROXY_NAMES)
    assert all(environment[name] == bypass_value for name in STANDARD_NO_PROXY_NAMES)


def test_proxy_bad_url():
    with pytest.raises(OutboundConfigurationError, match=r"HTTP\(S\) proxy URL"):
        normalize_outbound_environment({LLMPERF_PROXY: "proxy.internal:3128"})


def test_ray_proxy_disabled():
    environment = {RAY_GRPC_ENABLE_HTTP_PROXY: "true"}

    configure_ray_direct(environment)

    assert environment[RAY_GRPC_ENABLE_HTTP_PROXY] == "0"


def test_proxy_bootstrap_import():
    excluded_names = {
        LLMPERF_PROXY,
        LLMPERF_NO_PROXY,
        "LLMPERF_HUGGINGFACE_PROXY",
        "HF_HUB_DISABLE_XET",
        *STANDARD_PROXY_NAMES,
        *STANDARD_NO_PROXY_NAMES,
    }
    subprocess_environment = {
        name: value for name, value in os.environ.items() if name not in excluded_names
    }
    subprocess_environment[LLMPERF_PROXY] = "http://proxy.internal:3128"
    command = (
        "import os; import llmperf_backend.artifacts; "
        "from huggingface_hub.constants import HF_HUB_DISABLE_XET; "
        "names = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', "
        "'http_proxy', 'https_proxy', 'all_proxy'); "
        "raise SystemExit(0 if not HF_HUB_DISABLE_XET and "
        "all(os.environ.get(name) for name in names) else 1)"
    )

    completed_process = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        env=subprocess_environment,
    )

    assert completed_process.returncode == 0

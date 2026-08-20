"""Process-wide outbound transport policy for Backend clients and Ray control."""

from dataclasses import dataclass, replace
import os
from typing import Literal, MutableMapping, Optional, Tuple
from urllib.parse import urlsplit


LLMPERF_PROXY = "LLMPERF_PROXY"
LLMPERF_NO_PROXY = "LLMPERF_NO_PROXY"
RAY_GRPC_ENABLE_HTTP_PROXY = "RAY_grpc_enable_http_proxy"
STANDARD_PROXY_GROUPS = (
    ("HTTP_PROXY", "http_proxy"),
    ("HTTPS_PROXY", "https_proxy"),
    ("ALL_PROXY", "all_proxy"),
)
STANDARD_PROXY_NAMES = tuple(name for group in STANDARD_PROXY_GROUPS for name in group)
STANDARD_NO_PROXY_NAMES = ("NO_PROXY", "no_proxy")
OutboundSource = Literal["explicit", "llmperf", "standard", "none"]


class OutboundConfigurationError(ValueError):
    """Raised when one process-wide outbound setting is invalid."""


@dataclass(frozen=True)
class OutboundPolicy:
    """Sanitized, process-wide outbound transport configuration."""

    source: OutboundSource
    proxy_url: Optional[str]
    standard_proxy_names: Tuple[str, ...]
    no_proxy_names: Tuple[str, ...]

    @property
    def proxy_label(self) -> str:
        """Return a credential-free proxy description for logs."""

        if not self.proxy_url:
            return "environment/default"
        parsed_url = urlsplit(self.proxy_url)
        port = f":{parsed_url.port}" if parsed_url.port else ""
        return f"{parsed_url.scheme}://{parsed_url.hostname}{port}"


def configure_ray_direct(environment: MutableMapping[str, str]) -> None:
    """Keep Backend-to-Ray gRPC control traffic off HTTP proxies."""

    environment[RAY_GRPC_ENABLE_HTTP_PROXY] = "0"


def _environment_value(
    environment: MutableMapping[str, str], name: str
) -> Optional[str]:
    raw_value = environment.get(name)
    if raw_value is None:
        return None
    normalized_value = raw_value.strip()
    return normalized_value or None


def _validated_proxy(name: str, proxy_url: str) -> str:
    normalized_url = proxy_url.strip()
    parsed_url = urlsplit(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise OutboundConfigurationError(f"{name} must be an HTTP(S) proxy URL")
    if parsed_url.query or parsed_url.fragment:
        raise OutboundConfigurationError(
            f"{name} must not contain query or fragment components"
        )
    return normalized_url


def _fill_aliases(
    environment: MutableMapping[str, str],
    names: Tuple[str, str],
    fallback_value: Optional[str],
) -> None:
    selected_value = fallback_value
    for name in names:
        configured_value = _environment_value(environment, name)
        if configured_value is not None:
            selected_value = configured_value
            break
    if selected_value is None:
        return
    for name in names:
        if _environment_value(environment, name) is None:
            environment[name] = selected_value


def standard_https_proxy(
    environment: MutableMapping[str, str],
) -> Optional[str]:
    """Return the standard proxy applicable to an HTTPS destination."""

    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        configured_value = _environment_value(environment, name)
        if configured_value is not None:
            return _validated_proxy(name, configured_value)
    return None


def normalize_outbound_environment(
    environment: MutableMapping[str, str],
) -> OutboundPolicy:
    """Project LLMPerf proxy settings onto standard process variables.

    Explicit standard variables retain precedence. ``LLMPERF_PROXY`` fills every
    missing uppercase/lowercase HTTP, HTTPS, and ALL proxy alias so Python clients,
    command-line tools, and native ``hf-xet`` share one outbound policy.
    """

    original_standard_proxy = standard_https_proxy(environment)
    configured_proxy = _environment_value(environment, LLMPERF_PROXY)
    llmperf_proxy = (
        _validated_proxy(LLMPERF_PROXY, configured_proxy)
        if configured_proxy is not None
        else None
    )
    for proxy_group in STANDARD_PROXY_GROUPS:
        _fill_aliases(environment, proxy_group, llmperf_proxy)

    configured_no_proxy = _environment_value(environment, LLMPERF_NO_PROXY)
    _fill_aliases(environment, STANDARD_NO_PROXY_NAMES, configured_no_proxy)
    active_proxy_names = tuple(
        name
        for name in STANDARD_PROXY_NAMES
        if _environment_value(environment, name) is not None
    )
    active_no_proxy_names = tuple(
        name
        for name in STANDARD_NO_PROXY_NAMES
        if _environment_value(environment, name) is not None
    )
    active_proxy = standard_https_proxy(environment)
    if original_standard_proxy is not None:
        selected_source: OutboundSource = "standard"
    elif llmperf_proxy is not None:
        selected_source = "llmperf"
    else:
        selected_source = "none"
    return OutboundPolicy(
        source=selected_source,
        proxy_url=active_proxy,
        standard_proxy_names=active_proxy_names,
        no_proxy_names=active_no_proxy_names,
    )


# Native hf-xet reads standard proxy variables during import/session creation.
# Normalize the environment before importing Hugging Face client modules.
normalize_outbound_environment(os.environ)

from huggingface_hub import constants as huggingface_constants, set_client_factory
import httpx


def configure_outbound_transport(
    environment: MutableMapping[str, str],
    proxy_url: Optional[str] = None,
) -> OutboundPolicy:
    """Install one Hugging Face HTTP client from the active outbound policy."""

    environment_policy = normalize_outbound_environment(environment)
    if proxy_url is not None:
        normalized_proxy = proxy_url.strip()
        active_proxy = (
            _validated_proxy("proxy_url", normalized_proxy)
            if normalized_proxy
            else None
        )
        active_policy = replace(
            environment_policy,
            source="explicit",
            proxy_url=active_proxy,
        )
    else:
        active_policy = environment_policy

    def client_factory() -> httpx.Client:
        if active_policy.proxy_url:
            return httpx.Client(
                follow_redirects=True,
                timeout=None,
                proxy=active_policy.proxy_url,
            )
        return httpx.Client(follow_redirects=True, timeout=None)

    set_client_factory(client_factory)
    return active_policy


def xet_transport_label(policy: OutboundPolicy) -> str:
    """Describe the selected Xet path without exposing proxy values."""

    if huggingface_constants.HF_HUB_DISABLE_XET:
        return "disabled-explicit"
    if policy.standard_proxy_names:
        return "enabled-standard-proxy"
    return "enabled-direct"

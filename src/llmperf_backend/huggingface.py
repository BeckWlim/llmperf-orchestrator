"""Shared Hugging Face network configuration for backend-owned artifacts."""

import os
from typing import Optional
from urllib.parse import urlsplit

from huggingface_hub import configure_http_backend
import requests


HUGGINGFACE_PROXY = "LLMPERF_HUGGINGFACE_PROXY"


class HuggingFaceProxyError(ValueError):
    """Raised when the shared Hugging Face proxy setting is invalid."""


def resolve_huggingface_proxy(proxy_url: Optional[str] = None) -> Optional[str]:
    """Resolve and validate an explicit or environment-provided proxy URL."""

    configured = (
        proxy_url if proxy_url is not None else os.environ.get(HUGGINGFACE_PROXY, "")
    )
    if not configured:
        return None
    normalized = configured.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HuggingFaceProxyError(f"{HUGGINGFACE_PROXY} must be an HTTP(S) proxy URL")
    if parsed.query or parsed.fragment:
        raise HuggingFaceProxyError(
            f"{HUGGINGFACE_PROXY} must not contain query or fragment components"
        )
    return normalized


def huggingface_proxy_label(proxy_url: Optional[str]) -> str:
    """Describe a proxy for logs without exposing credentials."""

    if not proxy_url:
        return "environment/default"
    parsed = urlsplit(proxy_url)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def configure_huggingface_http(proxy_url: Optional[str]) -> None:
    """Apply the shared proxy to every Hugging Face Hub HTTP request.

    ``snapshot_download`` only forwards its ``proxies`` argument to individual
    file downloads in huggingface_hub 0.x. Its initial ``HfApi.repo_info`` call
    uses the process-wide Hub session instead, so configure that session as well.
    """

    if not proxy_url:
        configure_http_backend()
        return

    proxies = {"http": proxy_url, "https": proxy_url}

    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies.update(proxies)
        return session

    configure_http_backend(backend_factory=backend_factory)

"""Shared Hugging Face network configuration for backend-owned artifacts."""

import os
from typing import Optional
from urllib.parse import urlsplit


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

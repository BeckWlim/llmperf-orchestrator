"""Shared Hugging Face network configuration for backend-owned artifacts."""

import os
import re
from typing import Optional
from urllib.parse import urlsplit

from huggingface_hub import set_client_factory
import httpx

HUGGINGFACE_PROXY = "LLMPERF_HUGGINGFACE_PROXY"
HUGGINGFACE_REPOSITORY_SEGMENT = re.compile(r"\A[\w](?:[\w.-]*[\w])?\Z")


class HuggingFaceProxyError(ValueError):
    """Raised when the shared Hugging Face proxy setting is invalid."""


class HuggingFaceRepositoryError(ValueError):
    """Raised when an artifact repository ID is not a valid Hub identifier."""


def validate_huggingface_repository_id(repository_id: str) -> str:
    """Validate the stable public repository-ID grammar used by Hub APIs."""

    segments = repository_id.split("/")
    if len(segments) not in {1, 2} or any(
        not segment or len(segment) > 96 for segment in segments
    ):
        raise HuggingFaceRepositoryError(
            "repository ID must be 'name' or 'namespace/name' with segments up to "
            "96 characters"
        )
    if any(
        HUGGINGFACE_REPOSITORY_SEGMENT.fullmatch(segment) is None
        for segment in segments
    ):
        raise HuggingFaceRepositoryError(
            "repository ID segments must use letters, numbers, '_', '-' or '.', "
            "and cannot start or end with '-' or '.'"
        )
    if "--" in repository_id or ".." in repository_id:
        raise HuggingFaceRepositoryError("repository ID must not contain '--' or '..'")
    if repository_id.endswith(".git"):
        raise HuggingFaceRepositoryError("repository ID must not end with '.git'")
    return repository_id


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
    """Apply the shared proxy to every Hugging Face Hub HTTP request."""

    def client_factory() -> httpx.Client:
        options = {"follow_redirects": True, "timeout": None}
        if proxy_url:
            options["proxy"] = proxy_url
        return httpx.Client(**options)

    set_client_factory(client_factory)

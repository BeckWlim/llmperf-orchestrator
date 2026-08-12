from pathlib import Path

import pytest

jwt = pytest.importorskip("jwt")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from llmperf_backend.auth import TokenVerifier
from llmperf_backend.models import AuthConfig
from llmperf_cli.auth import PrivateKeyTokenProvider


class UnusedRepository:
    async def get_trusted_client_by_key_id(self, key_id):
        return None


def write_keys(tmp_path: Path):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def test_valid_token(tmp_path: Path):
    private_path, public_path = write_keys(tmp_path)
    provider = PrivateKeyTokenProvider(
        private_path,
        issuer="trusted-ctl",
        audience="llmperf-api",
        subject="test-ctl",
        ttl_seconds=30,
    )
    verifier = TokenVerifier(
        AuthConfig(
            enabled=True,
            public_key_path=str(public_path),
            issuer="trusted-ctl",
            audience="llmperf-api",
            bootstrap_subject="test-ctl",
        ),
        UnusedRepository(),
    )

    claims = verifier.verify_token(provider())

    assert claims["sub"] == "test-ctl"
    assert claims["aud"] == "llmperf-api"


def test_wrong_audience(tmp_path: Path):
    private_path, public_path = write_keys(tmp_path)
    provider = PrivateKeyTokenProvider(
        private_path,
        issuer="trusted-ctl",
        audience="wrong-api",
        subject="test-ctl",
        ttl_seconds=30,
    )
    verifier = TokenVerifier(
        AuthConfig(
            enabled=True,
            public_key_path=str(public_path),
            issuer="trusted-ctl",
            audience="llmperf-api",
            bootstrap_subject="test-ctl",
        ),
        UnusedRepository(),
    )

    with pytest.raises(jwt.InvalidAudienceError):
        verifier.verify_token(provider())

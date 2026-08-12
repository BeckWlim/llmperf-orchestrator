"""Private-key token provider for trusted CLI requests."""

from pathlib import Path
from hashlib import sha256
import stat
import time
from typing import List, Optional
from uuid import uuid4

import jwt
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from llmperf_cli.client import ClientError


DEFAULT_SSH_DIRECTORY = Path("~/.ssh")
DEFAULT_DISCOVERY_LIMIT = 32
_IGNORED_SSH_FILENAMES = {
    "authorized_keys",
    "config",
    "environment",
    "known_hosts",
    "known_hosts.old",
    "rc",
}


class PrivateKeyTokenProvider:
    """Create and cache short-lived RS256 JWTs, refreshing before expiry."""

    def __init__(
        self,
        private_key_path: Path,
        issuer: str,
        audience: str,
        subject: str,
        ttl_seconds: int = 60,
    ):
        if ttl_seconds < 10 or ttl_seconds > 300:
            raise ClientError("Token TTL must be between 10 and 300 seconds")
        key_path = private_key_path.expanduser().resolve()
        try:
            key_mode = stat.S_IMODE(key_path.stat().st_mode)
            private_key_bytes = key_path.read_bytes()
        except OSError as exc:
            raise ClientError(f"Unable to read private key {key_path}: {exc}") from exc
        if key_mode & 0o077:
            raise ClientError(
                f"Private key {key_path} is accessible by group/others; run chmod 600"
            )
        private_key = None
        for loader in (
            serialization.load_pem_private_key,
            serialization.load_ssh_private_key,
        ):
            try:
                private_key = loader(private_key_bytes, password=None)
                break
            except (ValueError, TypeError, UnsupportedAlgorithm):
                continue
        if private_key is None:
            raise ClientError(
                f"Unable to parse unencrypted PEM or OpenSSH private key {key_path}"
            )
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ClientError("Trusted client private key must be RSA")
        if private_key.key_size < 2048:
            raise ClientError("Trusted client RSA key must be at least 2048 bits")
        self.private_key = private_key
        public_der = private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.key_id = sha256(public_der).hexdigest()[:16]
        self.issuer = issuer
        self.audience = audience
        self.subject = subject
        self.ttl_seconds = ttl_seconds
        self._token: Optional[str] = None
        self._expires_at = 0

    def __call__(self) -> str:
        now = int(time.time())
        refresh_before = min(10, max(2, self.ttl_seconds // 4))
        if self._token is not None and now < self._expires_at - refresh_before:
            return self._token
        self._expires_at = now + self.ttl_seconds
        self._token = jwt.encode(
            {
                "iss": self.issuer,
                "aud": self.audience,
                "sub": self.subject,
                "iat": now,
                "exp": self._expires_at,
                "jti": str(uuid4()),
            },
            self.private_key,
            algorithm="RS256",
            headers={"kid": self.key_id},
        )
        return self._token


def discover_private_key_providers(
    ssh_directory: Path,
    issuer: str,
    audience: str,
    subject: str,
    ttl_seconds: int = 60,
    limit: int = DEFAULT_DISCOVERY_LIMIT,
) -> List[PrivateKeyTokenProvider]:
    """Discover usable RSA private keys without failing on unrelated SSH files."""

    directory = ssh_directory.expanduser().resolve()
    if not directory.is_dir() or limit <= 0:
        return []

    def priority(path: Path):
        preferred = {"llmperfctl": 0, "id_rsa": 1}
        return preferred.get(path.name, 2), path.name

    candidates = []
    try:
        entries = sorted(directory.iterdir(), key=priority)
    except OSError:
        return []
    for path in entries:
        if len(candidates) >= limit:
            break
        if (
            path.name in _IGNORED_SSH_FILENAMES
            or path.name.endswith(".pub")
            or path.name.endswith("-cert.pub")
            or path.is_symlink()
        ):
            continue
        try:
            if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
                continue
            provider = PrivateKeyTokenProvider(
                path,
                issuer,
                audience,
                subject,
                ttl_seconds,
            )
        except (ClientError, OSError):
            continue
        candidates.append(provider)
    return candidates

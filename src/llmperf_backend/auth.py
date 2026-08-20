"""Hot-reloadable public-key authentication for trusted llmperfctl clients."""

from dataclasses import dataclass
from hashlib import sha256
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Protocol, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import UnsupportedAlgorithm
import jwt
from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llmperf_backend.models import AuthConfig

BEARER = HTTPBearer(auto_error=False)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyMaterial:
    pem: bytes
    key_id: str
    file_signature: Tuple[int, int, int]


class TrustedClientRepository(Protocol):
    """Persistence capability needed by authentication."""

    async def get_trusted_client_by_key_id(
        self, key_id: str
    ) -> Optional[Dict[str, Any]]: ...


def _key_id(public_key: rsa.RSAPublicKey) -> str:
    der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return sha256(der).hexdigest()[:16]


def _deserialize_public_key(data: bytes) -> rsa.RSAPublicKey:
    errors = []
    for loader in (
        serialization.load_pem_public_key,
        serialization.load_ssh_public_key,
    ):
        try:
            public_key = loader(data)
            break
        except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
            errors.append(str(exc))
    else:
        raise ValueError("Unable to parse PEM or OpenSSH public key")
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise ValueError("Trusted client public key must be RSA")
    if public_key.key_size < 2048:
        raise ValueError("Trusted client RSA public key must be at least 2048 bits")
    return public_key


def normalize_public_key(public_key_text: str) -> Tuple[str, str]:
    """Accept PEM/OpenSSH RSA input and return (kid, normalized PEM)."""

    try:
        public_key = _deserialize_public_key(public_key_text.encode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"Unable to parse trusted public key: {exc}") from exc
    normalized = public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    return _key_id(public_key), normalized


class TokenVerifier:
    """Verify JWTs and atomically reload a rotating PEM/OpenSSH public key."""

    def __init__(self, config: AuthConfig, repository: TrustedClientRepository):
        self.config = config
        self.repository = repository
        self._lock = threading.RLock()
        self._key_path: Optional[Path] = None
        self._active: Optional[KeyMaterial] = None
        self._previous: Optional[KeyMaterial] = None
        self._previous_expires_at = 0.0
        self._next_reload_check = 0.0
        self._generation = 0
        self._last_reload_error: Optional[str] = None
        if config.enabled:
            if not config.public_key_path:
                raise RuntimeError(
                    "auth.public_key_path is required when auth is enabled"
                )
            self._key_path = Path(config.public_key_path).expanduser().resolve()
            self._active = self._load_key(self._key_path)
            self._generation = 1

    @staticmethod
    def _load_key(path: Path) -> KeyMaterial:
        try:
            with path.open("rb") as stream:
                source = stream.read()
                file_stat = os.fstat(stream.fileno())
            public_key = _deserialize_public_key(source)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load auth public key {path}: {exc}") from exc
        pem = public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return KeyMaterial(
            pem=pem,
            key_id=_key_id(public_key),
            file_signature=(
                int(file_stat.st_ino),
                int(file_stat.st_mtime_ns),
                int(file_stat.st_size),
            ),
        )

    def refresh(self, force: bool = False) -> bool:
        """Reload a changed PEM without replacing a known-good key on failure."""

        if not self.config.enabled or self._key_path is None:
            return False
        now = time.monotonic()
        with self._lock:
            if not force and now < self._next_reload_check:
                return False
            self._next_reload_check = now + self.config.reload_interval_seconds
            try:
                file_stat = self._key_path.stat()
                signature = (
                    int(file_stat.st_ino),
                    int(file_stat.st_mtime_ns),
                    int(file_stat.st_size),
                )
                if (
                    self._active is not None
                    and signature == self._active.file_signature
                ):
                    self._expire_previous(now)
                    return False
                candidate = self._load_key(self._key_path)
                if self._active is not None and candidate.key_id == self._active.key_id:
                    self._active = candidate
                    self._last_reload_error = None
                    self._expire_previous(now)
                    return False
                old_active = self._active
                self._active = candidate
                self._previous = old_active
                self._previous_expires_at = now + self.config.previous_key_grace_seconds
                self._generation += 1
                self._last_reload_error = None
                LOGGER.info(
                    "Reloaded trusted client public key: generation=%s kid=%s",
                    self._generation,
                    candidate.key_id,
                )
                return True
            except (OSError, RuntimeError) as exc:
                self._last_reload_error = str(exc)
                LOGGER.error(
                    "Public-key hot reload failed; retaining active key: %s", exc
                )
                return False

    def _expire_previous(self, now: float) -> None:
        if self._previous is not None and now >= self._previous_expires_at:
            self._previous = None
            self._previous_expires_at = 0.0

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._expire_previous(time.monotonic())
            return {
                "enabled": self.config.enabled,
                "generation": self._generation,
                "active_key_id": self._active.key_id if self._active else None,
                "previous_key_active": self._previous is not None,
                "reload_error": self._last_reload_error is not None,
            }

    def _decode(self, token: str, key: KeyMaterial) -> Dict[str, Any]:
        return self._decode_pem(token, key.pem)

    def _decode_pem(self, token: str, pem: Any) -> Dict[str, Any]:
        return jwt.decode(
            token,
            pem,
            algorithms=[self.config.algorithm],
            issuer=self.config.issuer,
            audience=self.config.audience,
            leeway=self.config.leeway_seconds,
            options={"require": ["exp", "iat", "iss", "aud", "sub", "jti"]},
        )

    def verify_token(self, token: str) -> Dict[str, Any]:
        if not self.config.enabled:
            return {
                "sub": "anonymous",
                "role": "superuser",
                "auth_source": "disabled",
            }
        self.refresh()
        with self._lock:
            self._expire_previous(time.monotonic())
            if self._active is None:
                raise RuntimeError("Authentication public key is not loaded")
            try:
                token_key_id = jwt.get_unverified_header(token).get("kid")
            except jwt.PyJWTError:
                token_key_id = None
            if token_key_id == self._active.key_id:
                claims = self._decode(token, self._active)
                return self._bootstrap_principal(claims)
            if self._previous is not None and token_key_id == self._previous.key_id:
                claims = self._decode(token, self._previous)
                return self._bootstrap_principal(claims)
            if token_key_id is not None:
                raise jwt.InvalidSignatureError("Token key ID is not trusted")
            try:
                claims = self._decode(token, self._active)
            except jwt.InvalidSignatureError:
                if self._previous is None:
                    raise
                claims = self._decode(token, self._previous)
            return self._bootstrap_principal(claims)

    def _bootstrap_principal(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        if claims.get("sub") != self.config.bootstrap_subject:
            raise jwt.InvalidTokenError(
                "Bootstrap token subject does not match key owner"
            )
        principal = dict(claims)
        principal["role"] = "superuser"
        principal["auth_source"] = "bootstrap"
        return principal

    async def _verify_any_trusted_key(self, token: str) -> Dict[str, Any]:
        self.refresh()
        try:
            token_key_id = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError:
            token_key_id = None
        with self._lock:
            bootstrap_ids = {
                key.key_id for key in (self._active, self._previous) if key is not None
            }
        if token_key_id is None or token_key_id in bootstrap_ids:
            return self.verify_token(token)
        trusted_client = await self.repository.get_trusted_client_by_key_id(
            token_key_id
        )
        if trusted_client is None:
            raise jwt.InvalidSignatureError("Token key ID is not trusted")
        claims = self._decode_pem(token, trusted_client["public_key_pem"])
        if claims.get("sub") != trusted_client["username"]:
            raise jwt.InvalidTokenError("Token subject does not match key owner")
        principal = dict(claims)
        principal["role"] = trusted_client["role"]
        principal["auth_source"] = "database"
        return principal

    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Security(BEARER),
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            principal = {
                "sub": "anonymous",
                "role": "superuser",
                "auth_source": "disabled",
            }
            request.state.principal = principal
            return principal
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A trusted client token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            principal = await self._verify_any_trusted_key(credentials.credentials)
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired trusted client token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        request.state.principal = principal
        return principal

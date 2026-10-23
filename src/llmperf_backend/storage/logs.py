"""Immutable Runner log files; PostgreSQL stores only the content key."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Optional


class RunnerLogStore:
    def __init__(self, directory: str, byte_limit: int = 1_000_000):
        self.directory = Path(directory).expanduser()
        self.byte_limit = byte_limit

    def write(self, stdout: str, stderr: str) -> Optional[str]:
        if not stdout and not stderr:
            return None
        document = {
            name: value.encode("utf-8")[-self.byte_limit :].decode(
                "utf-8", errors="replace"
            )
            for name, value in (("stdout", stdout), ("stderr", stderr))
        }
        content = json.dumps(document, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        key = f"{digest[:2]}/{digest}.json"
        destination = self.directory / key
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
        return key

    def read(self, key: Optional[str]) -> dict[str, Optional[str]]:
        if key is None:
            return {"stdout": None, "stderr": None}
        if re.fullmatch(r"[0-9a-f]{2}/[0-9a-f]{64}\.json", key) is None:
            raise ValueError("Invalid Runner log key")
        try:
            content = (self.directory / key).read_bytes()
        except FileNotFoundError:
            return {"stdout": None, "stderr": None}
        if hashlib.sha256(content).hexdigest() != Path(key).stem:
            raise ValueError("Runner log digest mismatch")
        document = json.loads(content)
        if not isinstance(document, dict) or any(
            not isinstance(document.get(name), str) for name in ("stdout", "stderr")
        ):
            raise ValueError("Invalid Runner log document")
        return {"stdout": document["stdout"], "stderr": document["stderr"]}

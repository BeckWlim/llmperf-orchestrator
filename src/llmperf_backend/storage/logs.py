"""Immutable Runner and Worker log artifacts; PostgreSQL stores their keys."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from pydantic import JsonValue

from llmperf_backend.runtime.messages import WorkerCompletion


@dataclass(frozen=True)
class StoredWorkerLog:
    key: str
    payload: dict[str, JsonValue]


class RunnerLogStore:
    def __init__(self, directory: str, byte_limit: int = 1_000_000):
        self.directory = Path(directory).expanduser()
        self.byte_limit = byte_limit

    def _write(self, prefix: str, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        key = f"{prefix}/{digest}.json"
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
                directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary_path.unlink(missing_ok=True)
        return key

    def _read(self, key: str, prefix: str) -> bytes:
        if re.fullmatch(re.escape(prefix) + r"/[0-9a-f]{64}\.json", key) is None:
            raise ValueError("Invalid Runner log key")
        content = (self.directory / key).read_bytes()
        if hashlib.sha256(content).hexdigest() != Path(key).stem:
            raise ValueError("Runner log digest mismatch")
        return content

    def write(self, runner_id: str, stdout: str, stderr: str) -> Optional[str]:
        if not stdout and not stderr:
            return None
        prefix = f"{UUID(runner_id)}/runner"
        document = {
            name: value.encode("utf-8")[-self.byte_limit :].decode(
                "utf-8", errors="replace"
            )
            for name, value in (("stdout", stdout), ("stderr", stderr))
        }
        return self._write(
            prefix, json.dumps(document, ensure_ascii=False).encode("utf-8")
        )

    def read(self, key: Optional[str]) -> dict[str, Optional[str]]:
        if key is None:
            return {"stdout": None, "stderr": None}
        runner_id = str(UUID(key.split("/", 1)[0]))
        try:
            content = self._read(key, f"{runner_id}/runner")
        except FileNotFoundError:
            return {"stdout": None, "stderr": None}
        document = json.loads(content)
        if not isinstance(document, dict) or any(
            not isinstance(document.get(name), str) for name in ("stdout", "stderr")
        ):
            raise ValueError("Invalid Runner log document")
        return {"stdout": document["stdout"], "stderr": document["stderr"]}

    def write_worker(
        self, runner_id: str, worker_id: str, completion: WorkerCompletion
    ) -> StoredWorkerLog:
        """Keep full text; the Runner journal byte limit does not apply to completion data."""
        prefix = f"{UUID(runner_id)}/workers/{UUID(worker_id)}"
        stored_completion = completion.for_storage()
        payload = stored_completion.model_dump(mode="json")
        key = self._write(
            prefix, json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        return StoredWorkerLog(key=key, payload=payload)

    def read_worker(self, runner_id: str, worker_id: str, key: str) -> WorkerCompletion:
        prefix = f"{UUID(runner_id)}/workers/{UUID(worker_id)}"
        return WorkerCompletion.model_validate_json(self._read(key, prefix))

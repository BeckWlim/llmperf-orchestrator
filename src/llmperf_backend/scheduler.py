"""Async, database-backed scheduling for LLMPerf benchmark Workers."""

import asyncio
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from llmperf.utils import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.models import DatabaseConfig, SchedulerConfig
from llmperf_backend.persistence import (
    CANCELLED,
    FAILED,
    SUCCEEDED,
    TERMINAL_STATUSES,
    RunnerRepository,
)
from llmperf_backend.providers import ProviderRegistry
from llmperf_backend.tokenizers import TokenizerCache
from llmperf_backend.datasets import DatasetCache, WORKER_DATASET_PATH


WORKER_DATABASE_URL = "LLMPERF_WORKER_DB"
LOGGER = logging.getLogger(__name__)


class Scheduler:
    """Claim durable Runners and supervise calculation-only Worker subprocesses."""

    def __init__(
        self,
        repository: RunnerRepository,
        config: SchedulerConfig,
        database_config: DatabaseConfig,
        provider_registry: ProviderRegistry,
        tokenizer_cache: Optional[TokenizerCache] = None,
        dataset_cache: Optional[DatasetCache] = None,
    ):
        self.repository = repository
        self.config = config
        self.database_config = database_config
        self.provider_registry = provider_registry
        self.tokenizer_cache = tokenizer_cache or TokenizerCache()
        self.dataset_cache = dataset_cache or DatasetCache()
        self.scheduler_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        self._slots: List[asyncio.Task] = []
        self._busy_slots = set()
        self._stop: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if not self.config.enabled:
            LOGGER.info("Scheduler %s is disabled", self.scheduler_id)
            return
        self._stop = asyncio.Event()
        await self.repository.requeue_stale(self.config.stale_after_seconds)
        self._slots = [
            asyncio.create_task(self._worker(index), name=f"llmperf-scheduler-{index}")
            for index in range(self.config.max_concurrent_runners)
        ]
        LOGGER.info(
            "Scheduler %s started with %d slot(s)",
            self.scheduler_id,
            len(self._slots),
        )

    async def stop(self) -> None:
        if self._stop is None:
            return
        self._stop.set()
        for task in self._slots:
            task.cancel()
        if self._slots:
            await asyncio.gather(*self._slots, return_exceptions=True)
        self._slots.clear()
        self._busy_slots.clear()
        self._stop = None
        LOGGER.info("Scheduler %s stopped", self.scheduler_id)

    def status(self) -> Dict[str, Any]:
        state = (
            "disabled"
            if not self.config.enabled
            else ("running" if self._stop is not None else "stopped")
        )
        busy_slots = len(self._busy_slots)
        return {
            "scheduler_id": self.scheduler_id,
            "status": state,
            "max_concurrent_runners": self.config.max_concurrent_runners,
            "live_slots": sum(not slot.done() for slot in self._slots),
            "busy_slots": busy_slots,
            "worker_module": self.config.worker_module,
        }

    async def _worker(self, index: int) -> None:
        if self._stop is None:
            raise RuntimeError("Scheduler must be started before running workers")
        scheduler_slot_id = f"{self.scheduler_id}:{index}"
        while not self._stop.is_set():
            try:
                runner = await self.repository.claim_next(scheduler_slot_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Unable to claim a benchmark Runner")
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), self.config.poll_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            if runner is None:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), self.config.poll_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            LOGGER.info(
                "Scheduler slot %s claimed Runner %s",
                scheduler_slot_id,
                runner["runner_id"],
            )
            self._busy_slots.add(index)
            try:
                await self._execute(runner)
            finally:
                self._busy_slots.discard(index)

    def _working_directory(self) -> Path:
        return Path(self.config.working_directory).expanduser().resolve()

    def build_command(self, runner_id: str) -> List[str]:
        return [
            sys.executable,
            "-m",
            self.config.worker_module,
            "--runner-id",
            runner_id,
        ]

    def _bounded_log(self, content: bytes) -> str:
        return content[-self.config.log_bytes_limit :].decode("utf-8", errors="replace")

    async def worker_environment(
        self, runner: Dict[str, Any], base_environment: Dict[str, str]
    ) -> Dict[str, str]:
        provider_id = str(runner["benchmark"]["provider"])
        environment = self.provider_registry.worker_environment(
            provider_id, base_environment
        )
        tokenizer = runner["benchmark"].get("tokenizer")
        if tokenizer is not None:
            resolution = await self.tokenizer_cache.resolve(tokenizer)
            environment[TOKENIZER_PATH] = str(resolution.path)
            environment[TOKENIZER_FAST] = "true" if resolution.use_fast else "false"
        dataset = runner["benchmark"].get("dataset")
        if dataset is not None:
            resolution = await self.dataset_cache.resolve(dataset)
            environment[WORKER_DATASET_PATH] = str(resolution.path)
        return environment

    async def _execute(self, runner: Dict[str, Any]) -> None:
        runner_id = runner["runner_id"]
        process: Optional[asyncio.subprocess.Process] = None
        communicate_task: Optional[asyncio.Task] = None
        try:
            environment = await self.worker_environment(runner, dict(os.environ))
            environment[WORKER_DATABASE_URL] = self.database_config.url
            process = await asyncio.create_subprocess_exec(
                *self.build_command(runner_id),
                cwd=str(self._working_directory()),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await self.repository.set_process(runner_id, process.pid)
            LOGGER.info("Started Worker pid=%s for Runner %s", process.pid, runner_id)
            communicate_task = asyncio.create_task(process.communicate())
            cancelled = False
            while not communicate_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=self.config.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    cancelled = await self.repository.heartbeat(runner_id)
                    if cancelled:
                        await self._terminate(process)
                        break

            stdout_bytes, stderr_bytes = await communicate_task
            stdout = self._bounded_log(stdout_bytes)
            stderr = self._bounded_log(stderr_bytes)
            if not cancelled:
                current = await self.repository.get_runner(runner_id)
                cancelled = bool(current and current["cancel_requested"])
            if cancelled:
                await self.repository.finish_runner(
                    runner_id,
                    CANCELLED,
                    "Benchmark cancelled by user",
                    process.returncode,
                    stdout,
                    stderr,
                )
                LOGGER.info("Runner %s cancelled", runner_id)
                return
            if process.returncode != 0:
                await self.repository.finish_runner(
                    runner_id,
                    FAILED,
                    f"Benchmark worker exited with code {process.returncode}",
                    process.returncode,
                    stdout,
                    stderr,
                )
                LOGGER.error(
                    "Worker for Runner %s exited with code %s",
                    runner_id,
                    process.returncode,
                )
                return
            current = await self.repository.get_runner(runner_id)
            if current is None or current["status"] not in {SUCCEEDED, FAILED}:
                await self.repository.finish_runner(
                    runner_id,
                    FAILED,
                    "Worker exited successfully without committing benchmark results",
                    process.returncode,
                    stdout,
                    stderr,
                )
                return
            await self.repository.set_logs(
                runner_id, process.returncode, stdout, stderr
            )
            LOGGER.info("Worker completed Runner %s", runner_id)
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                await self._terminate(process)
            if communicate_task is not None:
                await asyncio.gather(communicate_task, return_exceptions=True)
            current = await self.repository.get_runner(runner_id)
            if current is not None and current["status"] not in TERMINAL_STATUSES:
                await self.repository.requeue_runner(
                    runner_id, "Scheduler stopped; Runner requeued"
                )
            raise
        except Exception as exc:
            LOGGER.exception("Scheduler failed while executing Runner %s", runner_id)
            stdout = ""
            stderr = ""
            exit_code = process.returncode if process is not None else None
            if communicate_task is not None and communicate_task.done():
                try:
                    stdout_bytes, stderr_bytes = communicate_task.result()
                    stdout = self._bounded_log(stdout_bytes)
                    stderr = self._bounded_log(stderr_bytes)
                except Exception:
                    pass
            await self.repository.finish_runner(
                runner_id, FAILED, str(exc), exit_code, stdout, stderr
            )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), self.config.cancel_grace_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

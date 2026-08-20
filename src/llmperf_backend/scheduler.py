"""Async, database-backed scheduling for LLMPerf benchmark Workers."""

import asyncio
import logging
import os
import socket
from typing import Any, Dict, List, Optional, Protocol, Sequence
from uuid import uuid4

from llmperf.utils import TOKENIZER_FAST, TOKENIZER_PATH
from llmperf_backend.models import (
    DatabaseConfig,
    PerformanceGuardConfig,
    SchedulerConfig,
)
from llmperf_backend.outbound import configure_ray_direct
from llmperf_backend.persistence import (
    CANCELLED,
    FAILED,
    SUCCEEDED,
    TERMINAL_STATUSES,
)
from llmperf_backend.providers import ProviderRegistry
from llmperf_backend.artifacts import (
    ArtifactCaches,
    DatasetCache,
    DatasetResolver,
    TokenizerCache,
    TokenizerResolver,
)
from llmperf_backend.safety import RuntimePerformanceGuard
from llmperf_backend.worker import (
    Worker,
    WORKER_DATASET_PATH,
    benchmark_actor_count,
    summarize_outcome,
)
from llmperf.common import RAY_ACTOR_CPUS_ENV

WORKER_DATABASE_URL = "LLMPERF_WORKER_DB"
WORKER_RAY_ACTOR_CPUS = RAY_ACTOR_CPUS_ENV
LOGGER = logging.getLogger(__name__)


class SchedulerRepository(Protocol):
    """Persistence operations required by Scheduler."""

    async def requeue_stale(self, stale_after_seconds: int) -> int: ...

    async def claim_next(self, scheduler_id: str) -> Optional[Dict[str, Any]]: ...

    async def heartbeat(self, runner_id: str) -> bool: ...

    async def complete_runner(
        self,
        runner_id: str,
        summary: Dict[str, Any],
        request_metrics: Sequence[Dict[str, Any]],
        exit_code: int,
        stdout: str,
        stderr: str,
        terminal_status: str = SUCCEEDED,
        error_message: Optional[str] = None,
    ) -> bool: ...

    async def finish_runner(
        self,
        runner_id: str,
        status: str,
        message: str,
        exit_code: Optional[int],
        stdout: str,
        stderr: str,
    ) -> bool: ...

    async def get_runner(self, runner_id: str) -> Optional[Dict[str, Any]]: ...

    async def requeue_runner(self, runner_id: str, message: str) -> None: ...


class Scheduler:
    """Claim durable Runners and supervise Ray-backed Worker handles."""

    def __init__(
        self,
        repository: SchedulerRepository,
        config: SchedulerConfig,
        database_config: DatabaseConfig,
        provider_registry: ProviderRegistry,
        tokenizer_cache: Optional[TokenizerResolver] = None,
        dataset_cache: Optional[DatasetResolver] = None,
        performance_guard_config: Optional[PerformanceGuardConfig] = None,
    ):
        self.repository = repository
        self.config = config
        self.database_config = database_config
        self.provider_registry = provider_registry
        if tokenizer_cache is None and dataset_cache is None:
            artifact_caches = ArtifactCaches.from_environment()
            self.tokenizer_cache: TokenizerResolver = artifact_caches.tokenizer
            self.dataset_cache: DatasetResolver = artifact_caches.dataset
        else:
            self.tokenizer_cache = tokenizer_cache or TokenizerCache()
            self.dataset_cache = dataset_cache or DatasetCache()
        self.performance_guard = RuntimePerformanceGuard(
            performance_guard_config or PerformanceGuardConfig()
        )
        self.scheduler_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        self._slots: List[asyncio.Task] = []
        self._busy_slots = set()
        self._stop: Optional[asyncio.Event] = None
        self._ray_module: Any = None
        self._ray_context: Any = None
        self._ray_address: Optional[str] = None
        self._ray_mode = "external" if config.ray_address else "embedded"
        self._ray_healthy = False
        self._ray_object_store_tripped = False
        self._ray_status: Dict[str, Any] = {"status": "stopped"}
        self._ray_monitor_task: Optional[asyncio.Task] = None
        self._worker_remote: Any = None
        self._active_workers: Dict[str, Worker] = {}

    async def start(self) -> None:
        if not self.config.enabled:
            LOGGER.info("Scheduler %s is disabled", self.scheduler_id)
            return
        self._stop = asyncio.Event()
        try:
            await self._start_ray_runtime()
            await self.repository.requeue_stale(self.config.stale_after_seconds)
            self._slots = [
                asyncio.create_task(
                    self._worker(index), name=f"llmperf-scheduler-{index}"
                )
                for index in range(self.config.max_concurrent_runners)
            ]
        except Exception:
            await self._stop_ray_runtime()
            self._stop = None
            raise
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
        if self._ray_monitor_task is not None:
            self._ray_monitor_task.cancel()
            await asyncio.gather(self._ray_monitor_task, return_exceptions=True)
            self._ray_monitor_task = None
        await self._stop_ray_runtime()
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
            "worker_kind": "ray_task",
            "active_workers": len(self._active_workers),
            "ray_mode": self._ray_mode,
            "ray_address": self._ray_address or self.config.ray_address,
            "ray_actor_num_cpus": self.config.ray_actor_num_cpus,
            "ray_runtime": dict(self._ray_status),
            "performance_guard": self.performance_guard.status(),
        }

    async def _start_ray_runtime(self) -> None:
        configure_ray_direct(os.environ)
        import ray

        options: Dict[str, Any] = {
            "namespace": "llmperf-control",
            "ignore_reinit_error": False,
        }
        if self.config.ray_address:
            options["address"] = self.config.ray_address
        else:
            options.update(
                {
                    "include_dashboard": False,
                    "num_cpus": self.config.ray_num_cpus,
                    "object_store_memory": (self.config.ray_object_store_memory_bytes),
                }
            )
        self._ray_module = ray
        self._ray_context = await asyncio.to_thread(ray.init, **options)
        address_info = getattr(self._ray_context, "address_info", {}) or {}
        self._ray_address = self.config.ray_address or address_info.get("address")
        if not self._ray_address:
            raise RuntimeError("Ray runtime did not expose a connectable address")
        await self._sample_ray_runtime()
        if not self._ray_healthy:
            raise RuntimeError("Ray runtime failed its initial health check")
        self._worker_remote = Worker.remote(ray)
        self._ray_monitor_task = asyncio.create_task(
            self._monitor_ray_runtime(), name="llmperf-ray-monitor"
        )
        LOGGER.info(
            "Ray runtime ready: mode=%s address=%s resources=%s",
            self._ray_mode,
            self._ray_address,
            self._ray_status.get("cluster_resources"),
        )

    async def _stop_ray_runtime(self) -> None:
        ray = self._ray_module
        self._ray_healthy = False
        self._ray_object_store_tripped = False
        self._ray_status = {"status": "stopped"}
        self._ray_context = None
        self._ray_address = None
        self._worker_remote = None
        self._ray_module = None
        if ray is not None:
            await asyncio.to_thread(ray.shutdown)

    async def _sample_ray_runtime(self) -> None:
        ray = self._ray_module
        if ray is None:
            self._ray_healthy = False
            self._ray_status = {"status": "stopped"}
            return

        def snapshot() -> Dict[str, Any]:
            cluster_resources = ray.cluster_resources()
            result = {
                "status": "healthy",
                "cluster_resources": cluster_resources,
                "available_resources": ray.available_resources(),
            }
            object_store_total = float(
                cluster_resources.get("object_store_memory", 0) or 0
            )
            object_store_available = float(
                result["available_resources"].get("object_store_memory", 0) or 0
            )
            result["object_store_available_ratio"] = (
                object_store_available / object_store_total
                if object_store_total > 0
                else None
            )
            # ``ray.nodes`` is useful with a native driver but is not guaranteed by
            # every Ray Client transport. Resource RPCs are the health invariant;
            # node detail is best-effort to keep external runtimes compatible.
            try:
                nodes = ray.nodes()
                result["alive_nodes"] = sum(bool(node.get("Alive")) for node in nodes)
            except Exception:
                result["alive_nodes"] = None
            return result

        try:
            self._ray_status = await asyncio.wait_for(
                asyncio.to_thread(snapshot),
                timeout=self.config.ray_health_timeout_seconds,
            )
            self._ray_healthy = (
                float(self._ray_status["cluster_resources"].get("CPU", 0)) > 0
            )
            object_store_ratio = self._ray_status.get("object_store_available_ratio")
            if isinstance(object_store_ratio, (int, float)):
                guard_config = self.performance_guard.config
                if self._ray_object_store_tripped:
                    self._ray_object_store_tripped = (
                        object_store_ratio
                        < guard_config.resume_ray_object_store_available_ratio
                    )
                else:
                    self._ray_object_store_tripped = (
                        object_store_ratio
                        <= guard_config.min_ray_object_store_available_ratio
                    )
            self._ray_status["claim_blocked"] = self._ray_object_store_tripped
            self._ray_status["claim_block_reason"] = (
                "ray_object_store_low" if self._ray_object_store_tripped else None
            )
            if not self._ray_healthy:
                self._ray_status["status"] = "unhealthy"
                self._ray_status["error"] = "Ray runtime exposes no CPU resources"
        except Exception as exc:
            if self._ray_healthy:
                LOGGER.error("Ray runtime health check failed: %s", exc)
            self._ray_healthy = False
            self._ray_status = {"status": "unhealthy", "error": str(exc)}

    async def _monitor_ray_runtime(self) -> None:
        if self._stop is None:
            return
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), self.config.ray_health_interval_seconds
                )
            except asyncio.TimeoutError:
                await self._sample_ray_runtime()

    async def _worker(self, index: int) -> None:
        if self._stop is None:
            raise RuntimeError("Scheduler must be started before running workers")
        scheduler_slot_id = f"{self.scheduler_id}:{index}"
        while not self._stop.is_set():
            if (
                not self.performance_guard.allow_claim()
                or not self._ray_healthy
                or self._ray_object_store_tripped
            ):
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), self.config.poll_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass
                continue
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
        environment[WORKER_RAY_ACTOR_CPUS] = str(self.config.ray_actor_num_cpus)
        return environment

    async def _prepare_worker_environment(
        self, runner: Dict[str, Any], base_environment: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """Resolve local artifacts while retaining cancellation and timeout control."""

        runner_id = runner["runner_id"]

        async def finish_cancelled() -> None:
            await self.repository.finish_runner(
                runner_id,
                CANCELLED,
                "Benchmark cancelled before Worker start",
                None,
                "",
                "",
            )
            LOGGER.info("Runner %s cancelled before Worker start", runner_id)

        # A previous Scheduler may have requeued a running Runner while preserving
        # its durable cancellation request. Check before touching artifacts so a
        # restart cannot turn that Runner into a Provider call.
        if await self.repository.heartbeat(runner_id):
            await finish_cancelled()
            return None
        task = asyncio.create_task(self.worker_environment(runner, base_environment))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.artifact_resolution_timeout_seconds
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise RuntimeError(
                        "Worker artifact resolution exceeded "
                        f"{self.config.artifact_resolution_timeout_seconds:g} seconds"
                    )
                done, _ = await asyncio.wait(
                    {task},
                    timeout=min(self.config.poll_interval_seconds, remaining),
                )
                if done:
                    environment = task.result()
                    if await self.repository.heartbeat(runner_id):
                        await finish_cancelled()
                        return None
                    return environment
                if await self.repository.heartbeat(runner_id):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await finish_cancelled()
                    return None
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _wait_worker(self, worker: Worker, runner_id: str) -> bool:
        if self._stop is None:
            raise RuntimeError("Scheduler must be started before supervising Workers")
        while not worker.ready():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), self.config.poll_interval_seconds
                )
            except asyncio.TimeoutError:
                if await self.repository.heartbeat(runner_id):
                    return True
                continue
            raise asyncio.CancelledError
        return False

    async def _cancel_worker(self, worker: Worker) -> None:
        worker.cancel(force=False)
        deadline = asyncio.get_running_loop().time() + self.config.cancel_grace_seconds
        while not worker.ready() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(min(0.1, self.config.cancel_grace_seconds))
        if not worker.ready():
            worker.cancel(force=True)
        worker.close()

    async def _execute(self, runner: Dict[str, Any]) -> None:
        runner_id = runner["runner_id"]
        worker: Optional[Worker] = None
        try:
            environment = await self._prepare_worker_environment(
                runner, dict(os.environ)
            )
            if environment is None:
                return
            if self._ray_module is None or self._worker_remote is None:
                raise RuntimeError("Ray Worker runtime is not initialized")
            actor_count = benchmark_actor_count(runner["benchmark"])
            worker = Worker(
                self._ray_module,
                self._worker_remote,
                runner_id,
                actor_count,
                self.config.ray_actor_num_cpus,
            )
            self._active_workers[runner_id] = worker
            execution_runtime = {
                "backend": "ray",
                "worker_kind": "ray_task",
                "ray_mode": self._ray_mode,
                "ray_namespace": "llmperf-control",
                "ray_actor_num_cpus": self.config.ray_actor_num_cpus,
                "ray_actor_count": actor_count,
                "resource_scheduling": "independent_actors",
                "runner_id": runner_id,
                "campaign_id": runner.get("campaign_id"),
            }
            worker.start(
                runner["benchmark"],
                environment,
                execution_runtime,
                self.config.log_bytes_limit,
            )
            LOGGER.info(
                "Started Ray Worker task=%s for Runner %s with %d actor(s)",
                worker.task_id(),
                runner_id,
                actor_count,
            )
            cancelled = await self._wait_worker(worker, runner_id)
            if cancelled:
                await self._cancel_worker(worker)
                await self.repository.finish_runner(
                    runner_id,
                    CANCELLED,
                    "Benchmark cancelled by user",
                    None,
                    "",
                    "",
                )
                LOGGER.info("Runner %s cancelled", runner_id)
                return

            payload = worker.result()
            stdout = str(payload.get("stdout") or "")
            stderr = str(payload.get("stderr") or "")
            if not payload.get("ok"):
                message = str(payload.get("error") or "Ray Worker failed")
                await self.repository.finish_runner(
                    runner_id,
                    FAILED,
                    message,
                    1,
                    stdout,
                    stderr,
                )
                LOGGER.error("Ray Worker for Runner %s failed: %s", runner_id, message)
                return

            summary = payload["summary"]
            requests = payload["requests"]
            summary["runner_metadata"] = runner["metadata"]
            summary["execution_runtime"]["worker_id"] = worker.task_id()
            terminal_status, message = summarize_outcome(summary, requests)
            committed = await self.repository.complete_runner(
                runner_id,
                summary,
                requests,
                0,
                stdout,
                stderr,
                terminal_status=terminal_status,
                error_message=message if terminal_status == FAILED else None,
            )
            if not committed:
                current = await self.repository.get_runner(runner_id)
                if current is not None and current["cancel_requested"]:
                    await self.repository.finish_runner(
                        runner_id,
                        CANCELLED,
                        "Benchmark cancelled before result commit",
                        None,
                        stdout,
                        stderr,
                    )
            LOGGER.info(
                "Ray Worker completed Runner %s: %s", runner_id, terminal_status
            )
        except asyncio.CancelledError:
            if worker is not None:
                worker.cancel(force=True)
                worker.close()
            current = await self.repository.get_runner(runner_id)
            if current is not None and current["status"] not in TERMINAL_STATUSES:
                await self.repository.requeue_runner(
                    runner_id, "Scheduler stopped; Runner requeued"
                )
            raise
        except Exception as exc:
            LOGGER.exception("Scheduler failed while executing Runner %s", runner_id)
            await self.repository.finish_runner(
                runner_id, FAILED, str(exc), 1, "", str(exc)
            )
        finally:
            if worker is not None:
                worker.close()
            self._active_workers.pop(runner_id, None)

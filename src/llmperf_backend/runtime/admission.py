"""Releasable Runner admission shared by new and cooling Runners."""

import asyncio


class RunnerAdmission:
    def __init__(self, capacity: int):
        self.capacity = asyncio.Semaphore(capacity)
        self.active: set[str] = set()

    async def acquire(self, runner_id: str) -> None:
        if runner_id in self.active:
            return
        await self.capacity.acquire()
        self.active.add(runner_id)

    def release(self, runner_id: str) -> None:
        if runner_id in self.active:
            self.active.remove(runner_id)
            self.capacity.release()

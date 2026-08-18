# Testing Boundaries

Tests are organized by responsibility so the same behavior is not asserted repeatedly at
multiple layers.

| Module | Verifies | Does not verify |
|---|---|---|
| `test_backend_worker.py` | Ray task/ObjectRef wrapper, Actor resources, environment isolation, result outcome | PostgreSQL transactions or Scheduler polling |
| `test_backend_scheduler.py` | single Ray runtime ownership, Scheduler-to-Runner-to-Worker assembly, heartbeat/cancellation handoff | benchmark metric algorithms or real database concurrency |
| `test_backend_safety.py` | workload expansion estimates, Actor budget, host memory and Object Store watermarks | actual Ray scheduling or Provider behavior |
| `test_benchmark.py` | request execution, metric normalization, cache-probe dependencies and statistics | Backend lifecycle or persistence |
| `test_sql.py` | PostgreSQL state machines, transactional concurrency, cross-Campaign claim fairness | fallback databases or Provider networking |
| `test_cli.py` | HTTP command contracts, nullable Worker PID, logging and rendering | internal Scheduler/Ray implementation |

`Worker` is a stable domain term and no longer means an OS subprocess. A Runner's
`worker.process_id` remains nullable; the Ray task identity is stored in
`summary.execution_runtime.worker_id`.

Default tests must not connect to PostgreSQL or a Provider:

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q
```

Run PostgreSQL semantics only with an explicitly configured disposable test database:

```bash
export LLMPERF_TEST_DB='postgresql+asyncpg:///llmperf_test'
.venv/bin/pytest -q -m postgresql
```

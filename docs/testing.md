# Testing Boundaries

Tests are organized by responsibility so the same behavior is not asserted repeatedly at
multiple layers.

| Module | Verifies | Does not verify |
|---|---|---|
| `test_examples.py` | Core workload YAML, bounded trials, and compiler expansion | Live Provider behavior |
| `test_task_compiler.py`, `test_batch_tasks.py` | Definition rules, seeds, node/Runner mapping, and batch boundaries | Production performance |
| `test_durations.py` | Duration units, fractional timing, removed field rejection, and submission budgets | Wall-clock execution precision |
| `test_runner_runtime.py` | Async mailbox, rolling optional ceiling, slot release between turns, and atomic dispatcher output | Live PostgreSQL/Ray integration or throughput equivalence |
| `test_backend_worker.py` | Ray task/ObjectRef wrapper, Actor resources, environment isolation, result outcome | PostgreSQL transactions or Scheduler polling |
| `test_backend_scheduler.py` | single Ray runtime ownership, Scheduler-to-Runner-to-Worker assembly, heartbeat/cancellation handoff | benchmark metric algorithms or real database concurrency |
| `test_ray_concurrency.py` | isolated Ray/HTTP verification of 64 simultaneous streams on eight actors, including shared credentials and 64 independent atomic handles | live Provider performance or paid inference |
| `test_backend_safety.py` | workload expansion estimates, Actor budget, host memory and Object Store watermarks | actual Ray scheduling or Provider behavior |
| `test_benchmark.py` | request execution, provider/local token-count policy, missing-usage coverage, tokenization outside timed execution, cache-probe dependencies and statistics | Backend lifecycle or persistence |
| `test_sql.py`, `test_runtime_storage.py` | PostgreSQL metadata, history/version commits, schema rejection, fencing, restart without replay, and runtime Campaign fairness | fallback databases or Provider networking |
| `test_runtime.py` | DAG readiness, delayed joins, bounded bus, shared capacity and cancellation, ordered conversations | durable storage and actual network transport |
| `test_cli.py` | HTTP command contracts, nullable Worker PID, logging and rendering | internal Scheduler/Ray implementation |
| `test_providers.py` | strict YAML/JSON profiles, required file selection, credential isolation, atomic reload and rejection of environment-only profiles | live Provider availability |

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

Verify real Ray concurrency against a local mock HTTP server:

```bash
LLMPERF_TEST_RAY=1 .venv/bin/pytest -q tests/test_ray_concurrency.py
```

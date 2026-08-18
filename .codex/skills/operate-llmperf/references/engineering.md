# LLMPerf Engineering Rules

## Architecture boundaries

- `models.py`: strict API/YAML models and configuration constraints.
- `task_compiler.py`: bounded matrix/repeat/parallel expansion into atomic invoke DAGs.
- `persistence.py`: PostgreSQL transactions, queues, task state, aggregation, and export.
- `planner.py`: materialize due RunnerPlan or Dispatch work as ordinary Runners.
- `scheduler.py`: claim queued Runners and supervise Worker handles.
- `worker.py`: execute one retry-free Ray task and return results.
- `safety.py`: estimate expanded Runners, requests, tokens, and effective concurrency.
- `llmperf_cli`: command parsing, HTTP boundaries, polling, logs, and rendering.
- `cache_probe.py`, `usage.py`, `cache_analysis.py`: within-Runner KV-cache evidence.

Keep the compiler, Planner, and Scheduler distinct. The compiler creates a finite graph;
Planner consumes due generic Dispatches; Scheduler consumes queued Runners. Waiting must
not occupy a Worker slot. Dependency UUIDs are scheduling keys; prompt hashes only verify
payload identity.

Worker is a Scheduler-local Ray execution handle for one Runner, not a subprocess.
Scheduler is the only Ray driver. Each Worker submits one task, and that task creates
Runner-owned serial client Actors (`max_concurrency=1`). Tasks must not initialize Ray or
connect to PostgreSQL. Campaigns share the runtime but never Actors or mutable request state.

Cross-Campaign execution is supported. PostgreSQL claims prefer Campaigns with fewer
running Runners and use transactional locks to avoid simultaneous bias. Ray queues Actors
independently; do not use an all-or-nothing placement group.

Runtime guards only stop new claims. Memory or Object Store pressure must not fail queued
or kill running Runners. Drop persisted ObjectRefs after completion. Ray infrastructure
restarts/retries remain disabled for benchmark correctness.

## Persistence

PostgreSQL is the only supported runtime database. Do not add SQLite or in-memory authority.
When an explicit compatibility break is authorized, update together:

1. SQLAlchemy records;
2. `sql/postgresql/init.sql`;
3. Repository transactions;
4. API, CLI, and export contracts;
5. PostgreSQL integration tests and architecture documentation.

Hot queue queries require appropriate status/time/claim indexes and query-plan review.

## State and errors

- Runner terminal states are `succeeded`, `failed`, and `cancelled`.
- Campaign `status` is lifecycle; `outcome` is aggregate execution result.
- Worker/Ray failures propagate to Runner outcome.
- Zero completed requests is failure and preserves the first request error and counts.
- Unknown Provider cache counters remain null, never zero.
- A failed occurrence does not rewind a RunnerPlan cursor.
- Do not retry an ambiguously sent cache-family node with the same payload.

## Tests

Test function names contain at most three underscores. PostgreSQL tests request the
`postgresql_url` fixture, read only `LLMPERF_TEST_DB`, require a
`postgresql+asyncpg` URL whose database name contains `test`, and skip when absent.

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q tests/test_cli.py tests/test_planner.py
.venv/bin/pytest -q
```

Avoid duplicated assertions across pure logic, API contract, and real PostgreSQL tests.
CLI tests verify the centralized projection/rendering boundary described in `io.md`.

## Examples and documentation

- `examples/` contains runnable operational examples named `example-<purpose>.yaml` with no
  more than three hyphens.
- Defaults must finish quickly when the Provider is available. Use bounded second-scale
  timings; document production-scale timings separately.
- Secrets never appear in examples.
- Update README, architecture, skills, and tests with YAML/API/export changes.
- Project prose, labels, comments, and user-facing messages are English-only.

## Delivery

Use `rg` for discovery and `apply_patch` for edits. Preserve unrelated dirty-worktree
changes, avoid destructive Git commands, run focused then complete tests, run
`git diff --check`, and report skipped PostgreSQL coverage. If a process still holds old
code or schema, state the restart/cutover requirements explicitly.

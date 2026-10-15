# LLMPerf Engineering Rules

## Architecture boundaries

- `models.py`: strict general API/YAML models and configuration constraints.
- `task_compiler.py`: task-specific Pydantic AST, `BaseNode` hierarchy, bounded workflow
  expansion, UUID-free compilation table, and runtime assembly.
- `persistence.py`: durable metadata, fenced attempts, checkpoints, messages, results and export.
- `planner.py`: materialize due RunnerPlan occurrences.
- `runtime.py`: bounded event bus, cached DAG readiness, timers and fair ready queue.
- `scheduler.py`: supervise Worker handles and the shared request pool.
- `worker.py`: local atomic handles or retry-free batch Ray tasks.
- `request_pool.py`: fair shared capacity and isolated per-request Ray adapters.
- `preparation.py`: initialization-only corpus calibration and bounded prompt chunks.
- `rounds.py`: deterministic Runner expansion and runtime round eligibility.
- `safety.py`: estimate expanded Runners, requests, tokens, and effective concurrency.
- `llmperf_cli`: command parsing, HTTP boundaries, polling, logs, and rendering.
- `cache_probe.py`, `usage.py`, `cache_analysis.py`: within-Runner KV-cache evidence.

Keep compiler, Planner, RuntimeCoordinator and Scheduler distinct. Planner materializes
scheduled occurrences. The coordinator owns dependencies, timers and a fair ready queue;
Scheduler owns the fixed shared Ray pool and bounded Worker handles. Waiting never occupies
a request slot. See `docs/RUNTIME.md` for the concrete runtime and storage contract.

Atomic invokes use lightweight Backend handles and cached tokenizers; ordinary batch Runners
use retry-free Ray tasks. Both share request actors across Runners. Never mutate process
credentials, initialize Ray in a Worker, or connect an actor to PostgreSQL. Default capacity
is eight actors times eight requests; the active Runner bound is independent.

Runtime guards stop new admission. Commit results and conversation messages before releasing
successors. Recover interrupted execution as indeterminate without automatic Provider replay.

Optional preparation actors reserve one CPU each from `ray_num_cpus`; request actors use
the remaining CPU budget. Calibration is once per ordinary Runner, before its timed loop.
Bundled-sonnet and external concatenate workloads use the estimated chunk path through
the existing dataset loader. External calibration samples at most 128 records with a
4,096-character cap each; actors access the full memory-mapped corpus. Exact replay and
sample mode retain existing preparation. Round barriers derive from durable Runner metadata
and status, and consume no request slots while waiting.

## Persistence

PostgreSQL is the durable metadata platform. Runtime queues are reconstructible; do not add SQLite or in-memory durable authority.
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
.venv/bin/python -m basedpyright
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q tests/test_cli.py tests/test_planner.py
.venv/bin/pytest -q
```

Treat the BasedPyright result as required acceptance evidence for Python changes. Run it
proactively before delivery and report its outcome alongside compile and test results; this
is an Agent delivery check, not a requirement to add or maintain a CI workflow.

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

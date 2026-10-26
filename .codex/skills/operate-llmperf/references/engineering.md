# LLMPerf Engineering Rules

## Architecture boundaries

Read `docs/limitations.md` for family seed and runtime capacity boundaries. Trials
generate and submit; they do not shuffle admission. Runner controls Worker readiness,
while shared Worker scheduling owns fairness. Defaults are 1024 active Runners,
eight aggregation threads, and 64 Worker slots. Preserve bounded async submission;
prepare input only inside an admitted Ray request actor invocation.

- `configuration/models.py`: strict general API/YAML models and configuration constraints.
- `compiler/tasks.py`: task-specific Pydantic AST, `BaseNode` hierarchy, bounded workflow
  expansion, UUID-free compilation table, and runtime assembly.
- `compiler/requests.py`: Worker template repetition, step dependencies, ordered message
  source selection, and planned Worker/token accounting.
- `storage/repository.py`: durable metadata, fenced attempts, checkpoints, messages, results and export.
- `runtime/planner.py`: materialize due RunnerPlan occurrences.
- `runtime/coordinator.py`: reconciliation hints, cached DAG readiness, timers and fair ready queue.
- `runtime/scheduler.py`: supervise Worker handles and the shared request pool.
- `runtime/runner.py`: async progression, dependency checks, descendant cancellation,
  message assembly, and optional outstanding-Worker ceiling.
- `runtime/preparation.py`: lightweight Runner progression, legacy calculation helpers, and aggregation.
- `runtime/worker_input.py`: Worker recipes and actor-local input preparation/cache.
- `runtime/dispatch.py`: bounded one-request Workers and durable output publication.
- `runtime/messages.py`: durable Runner mailbox, completion and cancellation envelopes.
- `runtime/triggers.py`: readiness conditions and asynchronous timer event sources.
- `runtime/worker.py`: benchmark calculation helpers and standalone execution handles.
- `runtime/request_pool.py`: fair shared capacity and isolated per-request Ray adapters.
- `llmperf/datasets/preparation.py`: initialization-only corpus calibration and bounded prompt chunks.
- `compiler/rounds.py`: deterministic Runner expansion and runtime round eligibility.
- Matrix phases are generic completion barriers, not cache-tier policies. Keep roles
  in YAML, preserve family identity by group name/coordinates, and persist date-based
  seeds before execution. Do not reread the clock or move fairness into Runner logic.
- `compiler/safety.py`: estimate expanded Runners, requests, tokens, and effective concurrency.
- `llmperf_cli`: command parsing, HTTP boundaries, polling, logs, and rendering.
- `llmperf/datasets/probes.py`, `llmperf/metrics/usage.py`, `llmperf/metrics/cache.py`: within-Runner KV-cache evidence.

Keep compiler, Planner, RuntimeCoordinator and Scheduler distinct. Planner materializes
scheduled occurrences. The coordinator owns dependencies, timers and a fair ready queue;
Scheduler owns the fixed shared Ray pool and bounded Worker handles. Waiting never occupies
a request slot. See `docs/runtime.md` for the concrete runtime and storage contract.

Ray request actors prepare inputs with cached tokenizers; both atomic invokes and compiled `run` batches
submit one-request Workers through the same dispatcher. All share request actors across Runners. Never mutate process
credentials, initialize Ray in a Worker, or connect an actor to PostgreSQL. Default capacity
is eight actors times eight requests; the active Runner bound is independent.

Runtime guards stop new admission. Commit results and generic Worker messages before releasing
successors. Recover interrupted execution as indeterminate without automatic Provider replay.

Runner owns Provider/model/tokenizer, generation settings, and named `inputs` with per-input
new-token budgets. `runner.worker` defines execution logic and message composition, with
one Worker by default. Steps reference named inputs; `repeat` and `steps` declare finite
work. `depends_on` waits for success; `reuse_messages: true` on a
dependency also copies its structured messages. Multiple sources concatenate in declared
order without deduplication. Runner owns cancellation when a predecessor fails; the bus
stores and delivers events, and Workers never wait for other Workers. Runner `timeout`
and optional `concurrent_workers` stay outside `request`. There is no submission-budget
authoring field or automatic retry count.

Keep helper functions in the module that owns their behavior. Named-input preparation
belongs in `runtime/preparation.py`, timer sources with `runtime/triggers.py`, private
sampling in benchmark throughput, and task evidence projection in metrics results.
Retain separate shared models/protocols when they prevent dependency cycles or define a
cross-layer contract; small file size alone is not a reason to merge those boundaries.

The preparation-worker setting retains CPU headroom when sizing the request pool;
Backend no longer starts separate preparation actors. Each request actor serializes
CPU preparation and overlaps Provider I/O. One physical slot covers one complete
Worker. Ordinary concatenate recipes retain calibrated indexed generation; other
generated inputs use versioned deterministic request seeds. Round barriers consume
no request slots while waiting. Dedicated Backend maintenance renews the runtime lease
every five seconds; ordinary fenced writes share its row lock through commit. Read-only
dependency scans hold no lease lock. Expired or superseded ownership is never revived.

The dispatcher delivers committed events directly to bounded Backend mailboxes. One shared
reconciler recovers missed deliveries from PostgreSQL in bounded batches; do not restore
per-Runner polling on the managed path. Acknowledge processed batches only after progression.
Recovered backlogs request another pass after acknowledgment. Heartbeat writes and durable
cancellation checks are batched by Backend maintenance. Preserve single-send reservations,
committed-before-delivered ordering, and indeterminate recovery for interrupted Runners.

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

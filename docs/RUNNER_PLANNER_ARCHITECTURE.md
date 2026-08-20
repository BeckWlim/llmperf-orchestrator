# LLMPerf Runner Planner Architecture

## Purpose

Planner turns durable, time-qualified work into ordinary queued Runners. It supports both
recurring RunnerPlans and dependency-based task Dispatches without owning execution.

```text
RunnerPlan cursor ──┐
                    ├─> Planner transaction -> queued Runner -> Scheduler -> Worker
Task Dispatch DAG ──┘
```

Scheduler remains the execution owner. Waiting schedules and dependency delays consume no
Scheduler slot, Worker, Ray Actor, or Provider connection.

## Responsibility boundaries

Control plane:

- validates strict YAML/API input;
- resolves Provider, tokenizer, and dataset references;
- applies the performance guard;
- atomically persists Campaign workload.

Workload Compiler:

- expands finite matrix/trial combinations;
- lowers typed workflow nodes into a UUID-free logical compilation table;
- derives deterministic payload seeds;
- assembles runtime UUID dependencies only after expansion is complete;
- persists generic dependencies and delays.

Planner:

- finds due RunnerPlan occurrences and pending task Dispatches;
- claims rows transactionally;
- materializes immutable Runner templates;
- advances cursors and records events;
- never interprets role or experiment names.

Scheduler:

- fairly claims queued Runners;
- creates and supervises one Worker handle per Runner;
- maintains heartbeats and cancellation;
- persists terminal results.

## RunnerPlan model

A RunnerPlan stores a frozen Runner template plus recurrence state:

- timezone and recurrence definition;
- start/end bounds and optional maximum occurrences;
- overlap policy and misfire grace;
- next/last fire time and occurrence cursor;
- emitted/skipped counts and lifecycle status.

Supported recurrence shapes include bounded interval and calendar schedules. Preview uses
the same calculation as persistence and exposes daylight-saving adjustments. The database
cursor is authoritative; the CLI does not compute occurrences.

Occurrence identity is unique per `(runner_plan_id, plan_occurrence)`. Concurrent Planner
processes use PostgreSQL row locks so one occurrence is materialized once. A failed Runner
does not rewind the cursor.

## Task Dispatch model

A compiled task node is stored in `benchmark_runner_dispatches` with:

- `task_instance_id` and `node_id`;
- state: `blocked`, `pending`, `emitted`, or `cancelled`;
- `due_at`;
- full dependency ID list in lineage;
- planned `after_seconds`, role, and payload ID;
- frozen Runner template;
- emitted Runner ID and actual request timestamps.

Root nodes are immediately pending. Other nodes are blocked. After a successful node,
Repository releases a child only when all dependencies succeeded:

```text
child.due_at = max(dependency actual completion) + child.after_seconds
```

Each node uses the complete dependency list rather than a single parent field. Parallel
branches share an incoming frontier, and the next workflow node depends on every outgoing
branch frontier. Compiler topology uses stable logical node paths; Dispatch UUIDs are
introduced only during final assembly.

In 1.0 this is deliberately an all-success completion policy. Optional predecessors,
quorum joins, continue-on-error edges, and observation-conditioned activation are not
encoded in Task Dispatches. They require a future generic dependency-policy extension;
they should not be implemented as experiment-specific Planner branches.

## Persistence transactions

Campaign creation persists Campaign, immediate Runners, RunnerPlans, task definitions,
instances, and Dispatches atomically. No Worker or Provider call occurs in this transaction.

Planner claim/materialization transaction:

1. lock a due plan or Dispatch with `FOR UPDATE SKIP LOCKED`;
2. validate that the row still qualifies;
3. create one queued Runner from the frozen template;
4. bind the Runner to its occurrence or Dispatch;
5. advance cursor or mark Dispatch emitted;
6. write audit events;
7. commit before Scheduler can claim the Runner.

Task completion transaction:

1. lock Runner, source Dispatch, and Task Instance;
2. persist summary and request metrics;
3. record actual start/completion and prompt hash;
4. verify shared payload hashes;
5. mark failure/cancellation terminal and cancel descendants, or release ready children;
6. aggregate Task Instance state;
7. commit atomically.

## Time semantics

All persisted instants are timezone-aware UTC. User-facing calendar schedules retain an
IANA timezone for preview and daylight-saving behavior. Relative task delay is anchored to
actual Provider request completion, not submission time or planned time.

If a dependency finishes after a child's nominal schedule, the child becomes overdue and
is eligible immediately. Export preserves planned due time and actual timestamps; analysis
must not replace one with the other.

## Misfire and overlap

Misfire policy is explicit and bounded. Planner records skipped occurrences and adjustment
events instead of silently shifting history. Overlap policy controls whether due plan
occurrences queue while an earlier Runner is active. Task graphs do not use plan overlap;
their dependency DAG is the causal policy.

## Fairness and concurrency

Multiple Backend processes may run Planner loops. Row locks and uniqueness constraints
provide idempotency. Scheduler separately prefers Campaigns with fewer running Runners,
preventing one large Campaign from monopolizing all slots.

Ray Actors request resources independently. Planner does not reserve all Actors for a
Campaign or Runner and does not participate in Ray placement.

## Lifecycle aggregation

Campaign status is derived from Runner, RunnerPlan, Task Instance, and Dispatch state:

- running Runner -> `running`;
- queued Runner -> `queued`;
- active plan, blocked/pending task, or pending Dispatch -> `planned`;
- paused plans without active execution -> `paused`;
- all bounded work terminal -> `completed` or `cancelled`.

Campaign outcome remains separate: `pending`, `succeeded`, `partial_failed`, `failed`,
`cancelled`, or `no_runs`.

## Recovery

Planner and Scheduler restarts reconstruct state from PostgreSQL. Pending work remains
durable. Running Runners use heartbeat-based stale recovery. Cancellation propagates to
queued/running Runners, active plans, pending/blocked Dispatches, and task instances.

Version 1.0.0 starts from the schema in `sql/postgresql/init.sql`. Pre-release rows and
tables are outside the supported contract; do not rely on `create_all` to translate them.

## Operational checks

```bash
llmperfctl planner runtime
llmperfctl planner preview -f examples/example-runner-plan.yaml
llmperfctl planner events RUNNER_PLAN_ID
llmperfctl scheduler status
llmperfctl campaign status CAMPAIGN_ID
```

Investigate queue growth using plan cursors, pending Dispatch due times, dependency states,
Scheduler capacity, performance-guard state, PostgreSQL latency, and Ray resources. Never
solve a dependency problem by holding a Worker asleep.

## Verification

Pure scheduling and time calculations belong in unit tests. Transactional idempotency,
concurrent claims, task dependency release, and fairness require the explicitly configured
PostgreSQL test suite. Default tests must not connect to production or silently substitute
SQLite.

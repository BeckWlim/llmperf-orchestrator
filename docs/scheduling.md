# RunnerPlan scheduling

RunnerPlans materialize bounded time-based occurrences as ordinary queued Runners.
Use task definitions for causal dependencies and independent trials; see [task rules](tasks.md).
Planner owns recurrence, RuntimeCoordinator owns graph readiness, and Scheduler owns execution.
Waiting schedules and task delays reserve no Worker or request slots.

## Plan definition

Interval recurrence uses `recurrence: {kind: interval, every: 1}` for a one-second
interval, or `every: {value: 500, unit: ms}` for half a second. Bare numbers default to
seconds; `ms`, `s`, `m`, `h`, and `d` are supported. `every_seconds` is rejected.

A RunnerPlan stores a Runner template and recurrence state:

- timezone, interval or calendar recurrence, and start/end bounds;
- a finite maximum occurrence count or end time;
- overlap policy and misfire grace;
- next/last fire time, occurrence cursor, emitted/skipped counts, and lifecycle status.

[The RunnerPlan example](../examples/example-runner-plan.yaml) materializes two small
Runners one second apart. Preview uses the same time calculation as persistence, including
daylight-saving adjustments. Persisted instants are UTC; calendar definitions retain their
IANA timezone. A failed Runner does not rewind a plan cursor.

## Materialization and persistence

Campaign creation atomically persists immediate Runners, plans, and compiled task records.
Planner materializes due plan occurrences and records their cursor/events transactionally.
Occurrence identity is unique per `(runner_plan_id, plan_occurrence)`. PostgreSQL row locks
and uniqueness constraints prevent duplicate materialization.

Task nodes use the separate runtime path: RuntimeCoordinator reads committed progress,
evaluates dependencies/delays, and asks the repository to activate eligible execution nodes.
The durable table is `execution_nodes`; export and CLI contracts may call these records
Dispatches. There is no live `benchmark_runner_dispatches` queue. See
[durable records](runtime.md#durable-records).

## Overlap and misfires

Overlap policy controls whether a due occurrence can queue while a previous Runner is
active. Misfire grace bounds how late an occurrence may be handled. Skipped occurrences
and adjustments are recorded rather than silently rewriting history. Task definitions do
not use plan overlap settings; their compiled dependencies control eligibility.

## Admission and recovery

The Backend holds a fenced durable lease. The runtime ready queue rotates among Campaigns;
Scheduler bounds active Runners, and the shared request router bounds in-flight requests.
A schedule does not reserve actors or bypass resource admission. The public capacity
configuration is documented in [Runtime](runtime.md#capacity).

After restart, pending work is reconstructed from PostgreSQL. Interrupted executing attempts
become indeterminate and fail their Runners without automatic replay. Cancellation applies
to the Campaign's plans, pending nodes, and queued/running Runners. See
[recovery and cutover](runtime.md#recovery-and-cutover) for schema and ownership requirements.

## Operational checks

```bash
llmperfctl planner preview -f examples/example-runner-plan.yaml
llmperfctl planner runtime
llmperfctl planner status RUNNER_PLAN_ID
llmperfctl planner events RUNNER_PLAN_ID
llmperfctl scheduler status
llmperfctl campaign status CAMPAIGN_ID
```

Campaign lifecycle `completed` describes terminal work, not universal success. Inspect
`outcome`, Runner errors, plan events, and request metrics. Investigate queue growth using
cursors, due times, dependency states, Scheduler capacity, PostgreSQL latency, and Ray
resources. Waiting inside a Worker is not a scheduling mechanism.

Pure recurrence calculations belong in unit tests. Transactional idempotency, cancellation,
and recovery require the explicitly configured disposable PostgreSQL suite; see
[Testing](testing.md).

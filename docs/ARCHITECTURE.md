# LLMPerf Architecture

## 1. Design boundary

LLMPerf separates workload authoring, compilation, scheduling, execution, and analysis.
The runtime has one execution primitive: an atomic Runner. Experiment vocabulary such as
Prime, Warm, Probe, retention, and promotion is metadata, never Planner behavior.

```text
YAML authoring syntax
  matrix / sequence / repeat / parallel / invoke
                    │
                    ▼
             Workload Compiler
                    │
      finite CompiledTaskGraph (invoke DAG)
                    │
                    ▼
  PostgreSQL Dispatch queue -> Planner -> Runner -> Worker -> Provider
                    │
                    ▼
      export v6 -> normalized evidence -> autonomous HTML rendering
```

This is a deliberate compatibility break. The former protocol registry, per-experiment
models, state machines, cache analyzer, and `benchmark_protocol_*` schema are removed.

## 2. Control plane

FastAPI validates operator input and resolves Backend-owned Provider Profiles. Provider
URLs and credentials never enter task YAML. Campaign creation is atomic: invalid input or
an admission failure creates no partial workload.

The performance guard estimates the fully expanded graph before acceptance:

- compiled Runner count;
- Provider request count;
- input/output token budget;
- per-Runner and effective concurrency;
- available Ray Actor capacity.

Unknown benchmark defaults remain visible as admission warnings. A `repeat` is finite and
bounded; there is no runtime loop capable of escaping static accounting.

## 3. Workload Compiler

`task_definitions` is compile-time syntax. It is not a second scheduler.

### 3.1 Atomic node

Every compiled node contains:

- stable `node_id` inside one task instance;
- dependency Dispatch IDs;
- `after_seconds` relative to completion of all dependencies;
- one resolved Runner template;
- semantic `role` tag;
- logical `payload_id` and derived `payload_seed`;
- matrix dimensions and zero-based `trial_index`.

The compiler forces task Runners to `concurrent_requests: 1` and
`max_completed_requests: 1`.

### 3.2 Composition

- `matrix` creates Cartesian coordinates.
- `trials` creates independent samples at each coordinate.
- `sequence` carries the current dependency frontier forward.
- `repeat` expands to a serial chain; count and interval may reference dimensions.
- `parallel` creates siblings from the same frontier; the next sequence item joins all
  siblings.
- `invoke` is the only runtime node type.

The Planner receives only the resulting DAG and therefore does not change when new
experiments are composed.

### 3.3 Deterministic payloads

Generated payloads use a deterministic seed derived from:

```text
SHA-256(global seed, sorted matrix coordinates, trial index, seed namespace)
```

Different trials and namespaces form independent random families. Multiple nodes that
reference the same payload share the same seed. The Worker records the actual
`prompt_hash`; persistence stores the first hash for a payload and fails the instance if a
later replay differs. Deterministic input does not assert Provider cache behavior—it only
makes that behavior measurable.

The payload boundary can later host other deterministic generators or immutable dataset
samplers without changing graph or Planner semantics.

## 4. Persistence model

PostgreSQL is the source of truth.

| Table | Responsibility |
|---|---|
| `benchmark_campaigns` | workload ownership and metadata |
| `benchmark_runner_plans` | recurring Runner cursors |
| `benchmark_task_definitions` | submitted recipe and frozen Runner template |
| `benchmark_task_instances` | matrix/trial state, payload hashes, node outcomes |
| `benchmark_runner_dispatches` | dependency DAG, due time, node lineage, Runner link |
| `benchmark_runners` | execution state and resolved benchmark |
| `benchmark_requests` | per-request metrics |
| `benchmark_events` | Runner lifecycle audit trail |

Root Dispatches begin `pending`. Dependent Dispatches begin `blocked`. After a successful
node, Repository verifies payload identity, records request start/completion timestamps,
and releases each child only when every dependency succeeded. The child due time is:

```text
max(actual dependency completion timestamps) + after_seconds
```

Any failed/cancelled node makes the instance terminal and cancels un-emitted descendants.
No task-specific branch exists in this transaction.

## 5. Planner and Scheduler

Planner claims due Dispatch rows with PostgreSQL row locks and materializes their Runner
templates. Waiting Dispatches consume no Scheduler slot, Worker, Ray Actor, or Provider
connection.

Scheduler claims queued Runners fairly across Campaigns, starts isolated Ray work, records
heartbeats, and persists terminal results. PostgreSQL uniqueness constraints prevent
duplicate plan occurrences, duplicate task nodes, and duplicate Runner binding when
multiple Backend processes compete.

## 6. Worker and observability

The Worker builds one prompt from `payload_seed`, sends it through the selected Provider
adapter, and persists:

- request start and completion timestamps;
- input/output token counts and TTFT/E2E metrics;
- Provider cache hit/miss counters when available;
- `prompt_hash` and local tokenizer count;
- summary, error, timeout, stdout, and stderr evidence.

Provider counters are accounting evidence. TTFT deltas and speedup are performance
evidence. Reports must keep these concepts separate and expose counter coverage and sample
size.

## 7. Export and reporting

Campaign export v6 contains:

- `campaign` and `aggregate`;
- `runner_plans`;
- `task_definitions` and `task_instances`;
- `dispatches` with full lineage;
- generic joined `task_analyses`;
- `runners`, optionally with request records.

The report preparation script performs schema validation, cohort normalization, task graph
joins, and metric extraction. It does not prescribe chart types or infer experiment meaning
from fixed Provider/protocol names. The rendering Agent chooses the smallest useful visual
for actual data, keeps important conclusions large, folds Runner details, and may combine
bars and lines only when their units and relationship are clear.

Stable visual rules remain centralized:

- assign Provider series by deterministic slot order using a high-contrast palette; never
  match literal Provider names;
- keep hit-rate axes within 0–100%;
- show a 1× reference for speedup;
- use “Warm acceleration” or “Warm TTFT improvement”, not “Warm is faster” as an
  unsupported absolute claim;
- align table titles and use restrained curve smoothing that cannot invent extrema;
- preserve multiple Provider series in comparisons.

## 8. Failure and restart behavior

- A CLI disconnect does not affect Campaign execution.
- A Backend restart leaves pending/blocked Dispatches and queued Runners durable.
- A stale running Runner may be safely requeued by heartbeat policy.
- A task node without `prompt_hash`, or a replay whose hash changed, fails closed.
- Missing cache counters remain missing; they are never converted to zero or a miss.
- This schema has no migration bridge from the removed protocol tables.

# LLMPerf Architecture

## 1. Design boundary

LLMPerf separates workload authoring, compilation, scheduling, execution, and analysis.
The runtime has one execution primitive: an atomic Runner. Experiment vocabulary such as
Prime, Warm, Probe, retention, and promotion is metadata, never Planner behavior.

```text
YAML authoring syntax
  instances(matrix, trials, seed) / workflow / invoke / repeat / parallel / sequence
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
    export 1.0.0 -> normalized evidence -> autonomous HTML rendering
```

Version 1.0.0 defines the first supported task-graph, persistence, export, and analysis
contract. Pre-release protocol registries and experiment-specific state machines are not
part of the release surface.

### 1.1 Versioned format inventory

| Surface | Version marker | 1.0.0 contract |
|---|---|---|
| Python distribution and CLI | package `version` | `1.0.0` |
| Backend configuration YAML | root `version` | string `"1.0.0"` |
| Runner and Campaign YAML/JSON | root `version` | string `"1.0.0"` |
| HTTP control plane | route prefix | `/api/v1` |
| RunnerPlan persisted template | `template_version` / `plan_template_version` | string `"1.0.0"` |
| Compiled task definition and instance | `format_version` | string `"1.0.0"` |
| Runner and Campaign export | root `version` | string `"1.0.0"` |
| Benchmark result envelope | root `version` | string `"1.0.0"` |
| Normalized report evidence | `analysis_version` | string `"1.0.0"` |
| Render plan and review | `render_plan_version` / `render_review_version` | string `"1.0.0"` |
| Hugging Face cache-link manifest | `schema` suffix | `llmperf-huggingface-cache-links/1.0.0` |

`llmperf.version` is the runtime authority for release, protocol, and compiled-task format
identifiers. External YAML is strict: the version is mandatory, unknown fields are rejected,
and pre-1.0.0 documents require an explicit offline migration rather than an in-process
adapter. The `/api/v1` path is the HTTP major-version boundary; request models reject an
explicit non-1.0.0 body version.

## 2. Control plane

FastAPI validates operator input and resolves Backend-owned Provider Profiles. Provider
URLs and credentials never enter task YAML. Campaign creation is atomic: invalid input or
an admission failure creates no partial workload.

Workload YAML selects only a stable Provider ID and model ID. The matching Backend Profile
owns `adapter` (`openai`, `anthropic`, `litellm`, `sagemaker`, or `vertexai`) and injects it
into the frozen Runner after validation; URLs, model names, and discovery modes never imply
an adapter. All adapters can execute ordinary or compiled Runners. Cache-analysis claims
remain conditional on the selected Provider returning comparable usage counters.

The performance guard estimates the fully expanded graph before acceptance:

- compiled Runner count;
- Provider request count;
- input/output token budget;
- per-Runner and effective concurrency;
- available Ray Actor capacity.

Unknown benchmark defaults remain visible as admission warnings. A `repeat` is finite and
bounded; there is no runtime loop capable of escaping static accounting.

### 2.1 Backend ownership layers

Backend support code is organized by stable ownership boundary rather than by individual
download or validation features:

- `config.py` owns dotenv discovery, Provider-only environment reloads, YAML expansion,
  validation, and the atomic active configuration store;
- `outbound.py` owns process-wide HTTP(S) proxy normalization, the Hugging Face HTTP
  client, native Xet proxy visibility, and the explicit direct-connection policy for Ray
  control traffic;
- `artifacts.py` owns the common artifact descriptor and resolver capabilities, Hugging
  Face identity rules, dataset and tokenizer materialization, and active post-download
  integrity validation.

Dataset and tokenizer caches receive the same immutable outbound policy. Their resolutions
implement one artifact descriptor protocol, so preflight validation depends on the common
contract rather than importing and type-dispatching over concrete cache implementations.
Artifact validation also exposes an NDJSON stream of path-free byte progress, heartbeat,
and terminal result events. This keeps Hugging Face filesystem paths inside the Backend
while allowing a remote CLI to render transfer progress on stderr.
Worker runtime constants remain in `worker.py`; importing a Worker does not initialize the
Backend artifact stack merely to obtain its dataset handoff key. Worker-side token counting
retains its separate execution dependency on Transformers.

## 3. Workload Compiler

`task_definitions` is a typed compile-time DSL. It is not a second scheduler. Its dedicated
design is documented in [TASK_COMPILER_ARCHITECTURE.md](TASK_COMPILER_ARCHITECTURE.md).

### 3.1 Atomic node

Every logical compilation row contains:

- stable `node_id` inside one task instance;
- logical dependency node IDs;
- `after_seconds` relative to completion of all dependencies;
- semantic `role` tag;
- logical `payload_id` and derived `payload_seed`;
- matrix dimensions and zero-based `trial_index`.

Final assembly maps logical dependencies to Dispatch IDs and attaches one resolved Runner
template to each node.

The compiler forces task Runners to `concurrent_requests: 1` and
`max_completed_requests: 1`.

### 3.2 Composition

- `instances.matrix` creates Cartesian coordinates.
- `instances.trials` creates independent samples at each coordinate.
- `instances.seed` identifies deterministic payload families.
- top-level `workflow` carries the current dependency frontier forward;
- `invoke`, `repeat`, `parallel`, and nested `sequence` map directly to immutable
  `BaseNode` subclasses;
- `repeat` expands any nested node to a bounded serial chain;
- `parallel` expands nested branches from the same frontier and implicitly joins their
  outgoing frontiers;
- `invoke` remains the only runtime node type.

Expansion first produces a UUID-free `CompilationTable` of logical instance and node rows.
Only `TaskAssembler` assigns Task Instance and Dispatch UUIDs and converts logical
dependencies into persistence identities.

The Planner receives only the resulting DAG and therefore does not change when new
experiments are composed, provided that the experiment stays inside the finite,
admission-time-known execution envelope described below.

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

The payload boundary resolves artifact location independently from record decoding. A
dataset's required `adapter` names a registered prompt-record decoder: `sharegpt` extracts
any non-empty first conversation value, `sharegpt-user` restricts first-turn roles to
`human` or `user`, `document-text` maps every non-empty Parquet or Arrow `text` row to one
complete document, `text` maps non-empty text-file lines, and `builtin-sonnet` owns the
packaged fallback and its instruction. External adapters use Hugging Face Datasets to
normalize JSON, Parquet, Arrow, or text artifacts into a persistent Arrow-backed row index.
ShareGPT JSON arrays and JSONL are incrementally parsed into that index because the
upstream JSON builder otherwise performs a full read for arrays; other supported formats
use standard builders. The in-memory selection state contains shuffled row positions
rather than a second copy of all prompt text. No orchestration layer branches on a dataset
brand. All adapters expose one indexed record interface without changing graph or Planner
semantics. Dataset `sample` mode
preserves a whole record. `concatenate` mode walks a
seeded shuffle without replacement until the corpus is exhausted, then starts a newly
shuffled cycle only as needed to fill the requested token budget. A truncated final record
is still consumed in that cycle. The Worker records a text-free selection manifest (source
adapter, record indices, corpus cycles, segment sizes, seed, and manifest hash);
persistence verifies that evidence alongside `prompt_hash` on replay.

### 3.4 Supported experiment envelope

The 1.0 task model natively supports finite, open-loop experiments whose complete graph
is known before execution. This includes parameter matrices, repeated trials, ordered
phases, fan-out/fan-in layouts, completion-anchored delays, deterministic payload replay,
and provider or model comparison cohorts. A request-backed observer also fits this model
when one bounded invocation can produce terminal evidence in the normal Runner result.

The current model does not provide observation-conditioned branches, dynamic graph
growth, adaptive or unbounded loops, quorum or optional dependencies, continuous
side-channel sampling, or node kinds other than bounded invocation. Dependency release
uses an all-predecessors-succeeded rule, and a failed predecessor cancels its descendants.
Consequently, the statement that new experiments do not require Planner changes applies
to new compositions within this envelope, not to every possible observer protocol.

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

### 6.1 Extension horizon for complex observers

Future observer support should preserve the current separation between compilation,
dependency planning, execution, and reporting instead of adding experiment-specific
branches to the Planner. The intended extension seams are:

- a versioned, provider-neutral observer evidence envelope containing the observer kind,
  measurement interval, subject references, quality flags, and immutable artifact hashes;
- a typed execution capability for observers that cannot be represented as a normal
  request-backed Runner, while retaining one bounded unit of work per dispatch;
- generic dependency completion policies for optional, quorum, or continue-on-error joins;
- bounded compilation epochs for adaptive experiments, where a controller appends a
  versioned graph fragment transactionally rather than running an implicit loop inside a
  Worker or the Planner; and
- immutable artifact references for carrying observations between epochs without
  coupling orchestration state to provider-specific payloads.

Any such extension must expose a worst-case node, request, token, duration, and resource
budget to admission control. It must also retain deterministic identifiers and durable
evidence so that restart, replay, and audit properties remain intact.

## 7. Export and reporting

Campaign export 1.0.0 contains:

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

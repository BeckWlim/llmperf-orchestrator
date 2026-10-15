# LLMPerf Architecture

See [Runtime, storage, and multi-turn execution](RUNTIME.md) for the implemented
database layout, shared request capacity, message history, and recovery behavior.

## 1. Design boundary

LLMPerf separates workload authoring, compilation, scheduling, execution, and analysis.
The runtime has one execution primitive: a bounded Runner. Experiment vocabulary such as
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
  PostgreSQL metadata -> Runtime coordinator -> Scheduler -> Shared Ray actors -> Provider
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

Provider definitions live in one versioned YAML or JSON document with a `providers` map.
`LLMPERF_PROVIDERS_FILE` selects its path; otherwise the Backend discovers one
`providers.yaml`, `providers.yml`, or `providers.json` beside its environment file. Explicit
relative paths resolve against that directory. A selected file is authoritative for all
profiles; the legacy environment profile parser is used only when no file is selected or
discovered. Missing explicit files, ambiguous discovery, duplicate keys/normalized IDs,
unknown fields fail validation without a partial reload. Provider startup errors are isolated:
the API starts with an empty registry and a public configuration error, recoverable by reload.
`api_key` stores an inline secret; alternatively `api_key_env` references a source variable.
Empty or missing keys are allowed. Both sources cannot be specified together.
Optional `worker_api_key_env` and
`worker_api_base_env` specify adapter destinations. Source and destination credential
variables for all profiles, including removed profiles, are stripped before injecting the
selected profile into a Worker. Secret values are not part of public profile documents.
`provider reload` rereads the complete document and credentials and swaps one validated
registry generation; existing Runner connection snapshots remain unchanged.

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

- `provider_files.py` owns strict Provider YAML/JSON decoding and credential-reference resolution;
- `config.py` owns dotenv discovery, Provider source selection and reloads, YAML expansion,
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

Prompt construction and cache-probe mutation use the tokenizer before timed execution.
The default `benchmark.output_token_count=provider` never tokenizes responses. Request
actors do not load tokenizers; SageMaker and Vertex prediction schemas without usage return
unknown output counts. After request threads join and the finish timestamp is captured,
the Runner reconciles metrics using server usage. Input counts fall back to the prepared
local estimate with explicit provenance. Missing output usage stays null, including derived
token throughput and TPOT, and cannot be replaced by SSE-event counts or output budgets.
Summary output-token coverage counts successful requests with known counts; any missing
count makes overall output throughput null. Distribution statistics use known counts only.
Optional `output_token_count=local` retains generated text until this finalization phase
and counts output tokens once in the Runner, outside its measured duration. Provider
counts remain separate. Both ordinary Runners and cache probes follow this boundary.

This separation is informed by SGLang's
[benchmark metrics finalization](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py),
which calculates benchmark duration before retokenizing outputs. LLMPerf's default skips
that retokenization entirely; local counting is explicitly requested.

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

Each logical invoke sends exactly one request and completes independently. The runtime
shares request actors across eligible invokes; it does not impose a static group barrier.
Optional conversation names bind dependency-ordered turns to committed message history.

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

## 4. Persistence and runtime ownership

[Runtime storage](RUNTIME.md#durable-records) documents the concrete schema and ownership.
PostgreSQL stores metadata, checkpoints, attempts, immutable messages, results, and audit
records. `execution_nodes` replaces the Dispatch queue as durable node progress. A schema
revision check prevents the previous database from scheduling updated code.

## 5. Planner and Scheduler

Planner materializes bounded schedule occurrences. RuntimeCoordinator computes dependencies
and due times, reconciles durable state, and admits eligible Runners through a fair in-memory
queue. A bounded event bus wakes it after committed results. Waiting dependencies and delays
reserve no physical request capacity. The single Backend owner holds a fenced durable lease.

## 6. Worker and observability

### Request and actor concurrency

Eight shared Ray actors with eight requests per actor provide 64 request slots by default.
`live_slots` and `busy_slots` report physical request capacity and execution; active Runner
handles have an independent bound. Atomic invokes use lightweight Backend handles and cached
initialization tokenizers. Batch Runners retain a Ray task. Both execute HTTP requests through
the shared pool with isolated per-request adapter state and credentials.

Each compiled invoke completes independently. The runtime combines ready work across
Runners at the actor pool without a static group barrier. Multi-turn invokes use an
instance-scoped conversation name and dependency-ordered immutable history versions.
Successors become eligible only after result, messages, and checkpoint commit atomically.
See [configuration and examples](RUNTIME.md#capacity).

### Request evidence

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
- An interrupted executing Runner becomes indeterminate; automatic replay is disabled.
- A task node without `prompt_hash`, or a replay whose hash changed, fails closed.
- Missing cache counters remain missing; they are never converted to zero or a miss.
- This schema has no migration bridge from the removed protocol tables.

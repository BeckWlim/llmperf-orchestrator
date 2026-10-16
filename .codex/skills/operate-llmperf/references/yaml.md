# LLMPerf workload YAML

## Choose the smallest durable shape

- `runners`: immediate independent benchmark points.
- `runner_plans`: bounded recurring or calendar Runner templates.
- `task_definitions`: finite causal graphs compiled from atomic invokes.

One Campaign may contain all three. Do not create experiment-specific runtime protocols.

```yaml
version: "1.0.0"
campaign:
  name: experiment-name
  description: optional
  tags: {}
runners: []
runner_plans: []
task_definitions: []
wait: false
```

`wait` is a CLI field and is removed before API validation.

For repeated parallel batches, add `campaign.rounds: {count: 100, seed: 20261014}`
and supply the ordinary Runner templates in `runners`. Each round gets deterministic
seeds derived from the base seed, round, and template index. The next round waits until
all members of the current round are terminal. This shape cannot include RunnerPlans or
task definitions. Safety checks count the full expansion (at most 10,000 Runners), not
just templates. `metadata.runner_round` is reserved for Backend expansion.

Optional `LLMPERF_PREPARATION_WORKERS` enables bounded chunk generation for ordinary
bundled-sonnet and external `concatenate` workloads. Tokenization calibrates corpus estimates once per Runner before
timed sending; chunk generation during sending does not tokenize. Local input counts are
estimates and server usage remains authoritative. See `docs/runtime.md` for CPU budgeting.

## Runner and Worker template

Provider profiles own endpoints and credentials. Workload YAML contains only `provider`
and exact `model` IDs.

```yaml
timeout: 60
concurrent_requests: 1
provider: aliyun
model: deepseek-v4-pro
tokenizer:
  id: deepseek-ai/DeepSeek-V3
  revision: main
  use_fast: true
mean_output_tokens: 8
stddev_output_tokens: 0
output_token_count: provider
additional_sampling_params:
  temperature: 0
inputs:
  default:
    tokens: 2048
    dataset:
      source: huggingface
      id: organization/sharegpt
      filename: sharegpt.json
      revision: 0123456789abcdef0123456789abcdef01234567
      adapter: sharegpt-user
    dataset_prompt_mode: concatenate
    dataset_seed: 11111
request:
  repeat: 20
```

`dataset.adapter` is required and independent from artifact `source`. Use
`sharegpt-user` for a user-input corpus: it accepts only `conversations[0]` entries whose
normalized `from` role is `human` or `user`. The legacy `sharegpt` adapter accepts any
non-empty first conversation value. Use `document-text` when every non-empty Parquet or
Arrow `text` row is one complete document, including FineWeb-family artifacts. Use `text`
for one non-empty prompt per text-file line. ShareGPT adapters support JSON, Parquet, and
Arrow artifacts; `document-text` supports Parquet and Arrow; `text` supports text files.
External adapters normalize through Hugging Face Datasets into a persistent Arrow-backed
index, then pass through the same seeded construction and evidence pipeline. LLMPerf
incrementally materializes ShareGPT JSON arrays and JSONL because the upstream builder
fully reads array inputs; Parquet, Arrow, and text use their standard builders. Omit
`dataset` to select the bundled `src/llmperf/datasets/sonnet.txt` adapter. In `sample` mode, intact
records must already fit the requested token range.
`concatenate` deterministically shuffles first turns, consumes each record once before
cycling, joins diverse records, and truncates only the final segment to the requested token
budget. Concatenation requires `dataset_repeat_count: 1`; exact repetition belongs in a
compiled task through a shared payload ID.

External concatenation supports `shared_prefix_tokens`; intact-record `sample` mode does
not. With preparation workers enabled, this reuses the existing dataset loader, calibrates
from at most 128 records (4,096 characters each) during initialization, and constructs
bounded estimated-length chunks from the full corpus. Switching among installed dataset
adapters requires only workload YAML and cached artifacts, without dataset-specific runtime
changes. Without preparation workers, concatenation retains exact local token counting
during upfront preparation.

`output_token_count: provider` is the default. The tokenizer prepares prompts before timed
execution; output counts come only from server usage. Missing output counts and derived
token throughput/TPOT remain null, while request counts and latency remain usable. Inspect
`output_token_count_coverage` before comparing token throughput. `output_token_count: local`
retokenizes outputs after timed execution using the selected tokenizer; it is useful for
endpoints without usage, but local and server counts can differ. These fields also apply
to task/plan Runner templates. Actors never tokenize individual responses.

For cross-Runner replay tasks, use fixed token lengths and an explicit tokenizer resolved
to an immutable revision. Dataset-backed tasks are accepted only after the Backend resolves
the dataset revision to a Hugging Face commit hash. All registered Provider adapters may
execute compiled tasks; cache conclusions still require comparable Provider counter
evidence. Each invoke sends one request and retains an independent completion boundary. Runtime
request slots share actors dynamically across Runners and instances. Internal `adapter`,
resolved tokenizer provenance and `task_context` are Backend-owned fields.

## Task definition grammar

Durations use `after`, `every`, `timeout`, or cache-probe `delay`: a number means seconds,
and `{value: 500, unit: ms}` specifies units explicitly (`ms`, `s`, `m`, `h`, `d`).
Payloads use `seed`. Old suffixed duration keys, `seed_namespace`, and flat `kind`/`sequence`
authoring are rejected.

Runner owns Provider/model/tokenizer, generation settings, and named `inputs`. Each input
contains text or dataset settings and a new-input token target (`tokens: 128` or
`tokens: {mean: 128, stddev: 16}`). Reused history is additional context, not part of that
target. `request` owns only execution logic and message composition: `input`, `repeat`,
`steps`, dependencies, and timing. Steps select a Runner input by name. Omitted input
references select `default`. Omitting `repeat` and `steps` creates one Worker.
`repeat` copies the complete step graph; it is planned work, not retries.
`timeout` and optional `concurrent_requests` also belong to Runner.
The old `benchmark`, `max_submissions`, `request_steps`, and `history_from` authoring
fields are rejected. Internal resolved execution records still use `benchmark`.

The canonical authoring reference is `docs/tasks.md`; runnable feature coverage is indexed
in `docs/examples.md`. Use `instances.trials` with one `run` for independent batches; use
`repeat` only for serial batches. A failed node terminates its whole TaskInstance and cancels
queued siblings, so independent failure boundaries require separate instances.

Use `run: {name: batch}` for a whole multi-request Runner. It inherits the definition's
`runner.request` template and Runner execution settings. Put it
inside `repeat` for serial batches and use independent `instances.matrix` families for
independent progression. Batch-only definitions omit `payloads`; `invoke` still requires
an explicit payload and always sends one request. `run` supports `role` and `after`,
but not conversation or payload fields. Both node kinds advance on committed success.

Batch dataset seeds use `batch-v1`: task seed, sorted matrix coordinates, trial index,
and expanded node path (including repeat indices). Task seeds accept unsigned 64-bit
integers. Legacy invoke payload seeds remain unchanged. See
`examples/example-batch-sequence.yaml`. Eight families with 100 batches of 256 requests
compile to 800 nodes and 204,800 target requests, with peak concurrency 64 when each batch
allows eight requests. Counts inside a Runner must be included in safety estimates.

`runner.concurrent_requests`, when specified, bounds outstanding Workers within a Runner.
Omitting it leaves admission to Scheduler without a Runner window. Backend
`scheduler.ray_actor_max_concurrency` bounds each shared actor (default eight). Eight actors
provide 64 slots; `LLMPERF_RUNNERS_MAX=64` allows 64 atomic Runners to fill them. Dependencies
and delays reserve no slots. Static preview groups are singletons.

For multi-turn or merged inputs, use `depends_on` on request steps or atomic invokes:
`[{step: first, reuse_messages: true}, {step: second, reuse_messages: true}]`.
The Runner waits for every dependency to succeed, concatenates marked source message lists
in that order, and appends the current input. Duplicates and structured content are retained.
`depends_on: [first]` waits without message reuse. Failed or cancelled predecessors cause
Runner-owned cancellation of unsubmitted descendants; independent branches can continue.
No named conversation object is needed. Outer `run` dependencies only wait; define message
reuse within its `request.steps`. Payloads may contain literal `text` in addition to `seed`.
See `examples/example-multiturn.yaml`, `examples/example-message-merge.yaml`, and `docs/runtime.md`.

```yaml
task_definitions:
  - name: repeated-hit-surface
    instances:
      matrix:
        warmup_count: [0, 1, 2, 4]
        quiet_seconds: [0, 60, 300]
      trials: 8
      seed: 20260818
    payloads:
      replay: {seed: replay}
      cold: {seed: cold-control}
    workflow:
      - invoke:
          name: prime
          payload: replay
      - repeat:
          name: warmups
          count: $warmup_count
          every: 0
          node:
            invoke:
              name: warmup
              payload: replay
      - parallel:
          name: observation
          after: $quiet_seconds
          branches:
            - invoke:
                name: probe
                payload: replay
            - invoke:
                name: cold
                role: cold_control
                payload: cold
    runner:
      label: repeated-hit-surface
      metadata: {purpose: repeated-cache-hit}
```

### Matrix and trials

`instances` owns every independent expansion dimension. `instances.matrix` is a Cartesian
product of named integer dimensions, and `instances.trials` creates independent samples at
every coordinate. `instances.seed` controls deterministic payload families. A scalar task
value can be a literal integer or a readable `$dimension_name` reference. Flat top-level
`matrix`, `trials`, and `seed` fields are invalid. `instances` is one object rather than a
list; its plural name describes the generated TaskInstance set.

The compiler and performance guard account for expanded nodes, not YAML line count:

```text
instances = product(matrix dimension sizes) × trials
requests  = sum(batch request targets) + number of expanded invoke nodes
```

### Workflow and atomic invoke

`workflow` is an implicit top-level sequence. Every list entry is a one-key primitive
mapping, so the YAML hierarchy mirrors the compiler node hierarchy.

```yaml
- invoke:
    name: warm
    payload: replay
    after: $delay_seconds
```

- `name` contributes to the stable logical node path within an instance.
- `role` is optional and defaults to `name`; it is analysis metadata only.
- `payload` references a declared logical payload.
- `after` starts after every dependency completes.

### Repeat

```yaml
- repeat:
    name: observations
    count: $observation_count
    every: $interval_seconds
    node:
      invoke:
        name: warm
        payload: replay
```

Repeat expands its nested node to a serial chain. The nested node may itself be a sequence,
parallel group, or repeat. Count may be zero and is bounded at compile time. Never encode
an unbounded runtime loop.

### Parallel and join

```yaml
- parallel:
    name: observation
    after: 300
    branches:
      - invoke:
          name: probe
          payload: replay
      - invoke:
          name: cold
          role: cold_control
          payload: cold
```

All branches share the incoming dependency frontier. Each branch may be any primitive. The
next workflow step waits for every outgoing branch frontier, forming an implicit join.

### Nested sequence

Use `sequence` when one parallel or repeated branch needs multiple ordered operations:

```yaml
- sequence:
    name: verification
    steps:
      - invoke: {name: first, payload: replay}
      - invoke: {name: second, payload: replay}
```

Expanded node IDs are hierarchical, for example `observation.verification.first` and
`observations.2.warm`. UUIDs are assigned only after the logical compilation table is
complete; they are not part of task authoring or graph semantics.

## Deterministic random payloads

Generated payload content is pseudo-random but reproducible. Its seed is derived from:

```text
global task seed + sorted matrix coordinates + trial index + payload seed
```

Use cases include bundled-sonnet prompt families, deterministic sampling or concatenation
from an immutable external artifact, and independent controls. Reusing one payload ID in
Prime/Warm/Probe guarantees the same derived seed; different namespace or trial produces an
independent value. Persistence compares both the actual `prompt_hash` and the dataset
selection evidence on every replay and fails closed on mismatch.

Prompt selection evidence contains the source adapter, seed, stable record indices,
per-segment token/character counts, corpus cycle, and a manifest hash, but never prompt
text. For external sources, the frozen Runner also keeps the dataset ID, filename, and
resolved commit revision. Together these fields make every source auditable without copying
dataset content into Campaign exports.

## Common compositions

Passive retention: matrix on delay, independent instance per delay/trial, then Prime and a
delayed parallel Warm/Cold pair.

Access-conditioned residency: Prime followed by a Repeat chain. This measures retention
under repeated access and must not be described as passive TTL.

Repeated-hit behavior: matrix on warmup count and quiet window, Prime, Repeat Warmup, then
delayed Probe/Cold. Compare warmup counts only at like-for-like quiet windows.

Multi-Provider comparison: submit one task definition per resolved Provider/model template
using identical matrix, trials, seed namespaces, token shape, and workflow. Provider identity
is a cohort dimension, not a compiler condition.

## Safety checklist

- Prove the Provider/model with a one-request smoke first.
- Freeze tokenizer and any dataset revision.
- Fix sampling parameters when measuring exact replay.
- Use a unique family per matrix coordinate/trial; do not let one point warm another.
- Bound matrix size, trials, repeat count, RunnerPlan occurrences, and total duration.
- Keep the requested experiment within six hours when that is the stated upper bound.
- Run `validate_workload.py` with active Scheduler/Ray capacity before submission.
- Treat missing Provider counters as unknown, not zero.
- Do not retry an ambiguously sent cache-family node with the same payload.

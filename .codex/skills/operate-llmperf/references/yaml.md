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

## Runner benchmark

Provider profiles own endpoints and credentials. Workload YAML contains only `provider`
and exact `model` IDs.

```yaml
benchmark:
  provider: aliyun
  model: deepseek-v4-pro
  tokenizer:
    id: deepseek-ai/DeepSeek-V3
    revision: main
    use_fast: true
  timeout_seconds: 60
  concurrent_requests: 1
  max_completed_requests: 20
  mean_input_tokens: 2048
  stddev_input_tokens: 0
  shared_prefix_tokens: 0
  dataset:
    source: huggingface
    id: organization/sharegpt
    filename: sharegpt.json
    revision: 0123456789abcdef0123456789abcdef01234567
    adapter: sharegpt-user
  dataset_prompt_mode: concatenate
  dataset_repeat_count: 1
  dataset_seed: 11111
  mean_output_tokens: 8
  stddev_output_tokens: 0
  additional_sampling_params:
    temperature: 0
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
`dataset` to select the bundled `src/llmperf/sonnet.txt` adapter. In `sample` mode, intact
records must already fit the requested token range.
`concatenate` deterministically shuffles first turns, consumes each record once before
cycling, joins diverse records, and truncates only the final segment to the requested token
budget. Concatenation requires `dataset_repeat_count: 1`; exact repetition belongs in a
compiled task through a shared payload ID.

For cross-Runner replay tasks, use fixed token lengths and an explicit tokenizer resolved
to an immutable revision. Dataset-backed tasks are accepted only after the Backend resolves
the dataset revision to a Hugging Face commit hash. All registered Provider adapters may
execute compiled tasks; cache conclusions still require comparable Provider counter
evidence. The compiler overwrites task nodes to one request at concurrency one. Its internal
`adapter`, resolved tokenizer provenance, and `task_context` are Backend-owned execution
fields. They are not part of submitted YAML, and unknown fields are rejected.

## Task definition grammar

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
      replay: {seed_namespace: replay}
      cold: {seed_namespace: cold-control}
    workflow:
      - invoke:
          name: prime
          payload: replay
      - repeat:
          name: warmups
          count: $warmup_count
          every_seconds: 0
          node:
            invoke:
              name: warmup
              payload: replay
      - parallel:
          name: observation
          after_seconds: $quiet_seconds
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
requests  = sum(expanded invoke nodes across instances)
```

### Workflow and atomic invoke

`workflow` is an implicit top-level sequence. Every list entry is a one-key primitive
mapping, so the YAML hierarchy mirrors the compiler node hierarchy.

```yaml
- invoke:
    name: warm
    payload: replay
    after_seconds: $delay_seconds
```

- `name` contributes to the stable logical node path within an instance.
- `role` is optional and defaults to `name`; it is analysis metadata only.
- `payload` references a declared logical payload.
- `after_seconds` starts after every dependency completes.

### Repeat

```yaml
- repeat:
    name: observations
    count: $observation_count
    every_seconds: $interval_seconds
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
    after_seconds: 300
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
global task seed + sorted matrix coordinates + trial index + payload seed_namespace
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

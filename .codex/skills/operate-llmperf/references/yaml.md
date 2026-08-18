# LLMPerf workload YAML

## Choose the smallest durable shape

- `runners`: immediate independent benchmark points.
- `runner_plans`: bounded recurring or calendar Runner templates.
- `task_definitions`: finite causal graphs compiled from atomic invokes.

One Campaign may contain all three. Do not create experiment-specific runtime protocols.

```yaml
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
  llm_api: openai
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
  mean_output_tokens: 8
  stddev_output_tokens: 0
  additional_sampling_params:
    temperature: 0
```

For cross-Runner replay tasks, use generated prompts, fixed token lengths, OpenAI-compatible
metrics, and an explicit/model-bound tokenizer resolved to an immutable revision. The
compiler overwrites task nodes to one request at concurrency one. `task_request` is
compiler-owned and must never appear in submitted YAML.

## Task definition grammar

```yaml
task_definitions:
  - name: repeated-hit-surface
    matrix:
      warmup_count: [0, 1, 2, 4]
      quiet_seconds: [0, 60, 300]
    trials: 8
    seed: 20260818
    payloads:
      replay: {seed_namespace: replay}
      cold: {seed_namespace: cold-control}
    sequence:
      - {kind: invoke, id: prime, role: prime, payload: replay}
      - kind: repeat
        id: warmups
        count: {dimension: warmup_count}
        interval_seconds: 0
        invoke: {kind: invoke, id: warmup, role: warmup, payload: replay}
      - kind: parallel
        after_seconds: {dimension: quiet_seconds}
        invokes:
          - {kind: invoke, id: probe, role: probe, payload: replay}
          - {kind: invoke, id: cold, role: cold_control, payload: cold}
    runner:
      label: repeated-hit-surface
      metadata: {purpose: repeated-cache-hit}
      benchmark: {}
```

### Matrix and trials

`matrix` is a Cartesian product of named integer dimensions. `trials` creates independent
samples at every coordinate. A scalar task value can be a literal integer or
`{dimension: name}`.

The compiler and performance guard account for expanded nodes, not YAML line count:

```text
instances = product(matrix dimension sizes) × trials
requests  = sum(expanded invoke nodes across instances)
```

### Atomic invoke

```yaml
- kind: invoke
  id: warm
  role: warm
  payload: replay
  after_seconds: {dimension: delay_seconds}
```

- `id` is topology identity within an instance.
- `role` is free-form analysis metadata; it cannot alter scheduling.
- `payload` references a declared logical payload.
- `after_seconds` starts after every dependency completes.

### Repeat

```yaml
- kind: repeat
  id: observations
  count: {dimension: observation_count}
  interval_seconds: {dimension: interval_seconds}
  invoke: {kind: invoke, id: warm, role: warm, payload: replay}
```

Repeat expands to a serial chain. Count may be zero and is bounded at compile time. Never
encode an unbounded runtime loop.

### Parallel and join

```yaml
- kind: parallel
  after_seconds: 300
  invokes:
    - {kind: invoke, id: probe, role: probe, payload: replay}
    - {kind: invoke, id: cold, role: cold_control, payload: cold}
```

All siblings share the incoming dependency frontier. The next sequence step waits for all
siblings, forming an implicit join.

## Deterministic random payloads

Generated payload content is pseudo-random but reproducible. Its seed is derived from:

```text
global task seed + sorted matrix coordinates + trial index + payload seed_namespace
```

Use cases include randomly generated prompt families, deterministic sampling from a future
immutable dataset generator, and independent controls. Reusing one payload ID in
Prime/Warm/Probe guarantees the same derived seed; different namespace or trial produces an
independent value. Persistence compares the actual `prompt_hash` on every replay and fails
closed on mismatch.

Do not use nondeterministic random selection for causal experiments. If dataset sampling is
added to the compiler later, it must record the immutable dataset identity, selected item,
seed, and prompt hash.

## Common compositions

Passive retention: matrix on delay, independent instance per delay/trial, then Prime and a
delayed parallel Warm/Cold pair.

Access-conditioned residency: Prime followed by a Repeat chain. This measures retention
under repeated access and must not be described as passive TTL.

Repeated-hit behavior: matrix on warmup count and quiet window, Prime, Repeat Warmup, then
delayed Probe/Cold. Compare warmup counts only at like-for-like quiet windows.

Multi-Provider comparison: submit one task definition per resolved Provider/model template
using identical matrix, trials, seed namespaces, token shape, and sequence. Provider identity
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

# KV-Cache Observability and Experiment Design

## Scope

LLMPerf measures externally observable cache behavior from Provider responses and latency.
It cannot directly prove a Provider's internal cache topology, eviction policy, routing, or
storage tier. Reports must separate observations from inference.

The evidence model has three independent layers:

1. input identity: deterministic payload seed plus runtime prompt hash;
2. accounting: Provider-reported hit/miss counters and their coverage;
3. performance: paired or controlled TTFT/E2E measurements.

A matching input does not guarantee a cache hit. A reported hit does not guarantee a
material speedup. A lower TTFT without accounting counters is performance evidence, not
proof of the Provider's accounting category.

## Request instrumentation

Persist per request:

- client start and completion UTC timestamps;
- TTFT, end-to-end latency, inter-token latency, and output throughput;
- local input/output token counts;
- normalized Provider cache hit/miss tokens when available;
- raw counter availability/validity flags;
- prompt hash and task metadata;
- bounded error code/message and timeout state.

Never persist raw prompts in reports. Provider-private fields require explicit normalization
and tests before they become stable metrics.

## Counter normalization

Counter presence is distinct from a numeric zero. Missing, partial, negative, or
inconsistent counters remain unavailable and contribute quality flags.

For complete counters, pooled token hit ratio is:

```text
sum(hit_tokens) / (sum(hit_tokens) + sum(miss_tokens))
```

Do not average per-request percentages when denominators differ. Report:

- requests with any cache counter;
- requests with complete hit/miss counters;
- invalid/incomplete counter requests;
- token denominator and counter coverage;
- request hit probability under an explicit threshold rule.

Hit ratio axes are bounded to 0–100%.

## Within-Runner cache probe

`cache_probe` is appropriate when Prime/Warm or prefix/mutation requests can execute inside
one bounded Runner. The probe owns its short dependency plan and produces paired analysis.

Typical modes include exact-repeat and controlled prefix/mutation comparisons. For each
prompt family, preserve causal order:

```text
Prime -> Warm 1 -> Warm 2
```

Different families may run concurrently. A failed or ambiguous Prime skips that family's
dependent requests. Keep Provider retries disabled because retrying can change the cache
state being measured.

Use the paired verdict, sample count, counter coverage, and confidence interval where
available. Do not infer a global Provider TTL from one short local probe.

## Cross-Runner task graphs

Long waits and repeated access use `task_definitions`. Workload Compiler lowers matrix,
sequence, repeat, and parallel syntax into atomic single-request Runner nodes. Planner only
handles dependencies and due times.

### Deterministic random payloads

Generated payload seed is derived from global seed, sorted matrix coordinates, trial index,
and payload namespace. Different trials create independent random prompt families. Prime,
Warmup, Warm, and Probe nodes that reference the same payload replay identical input. Cold
controls use a different namespace.

Repository stores the first prompt hash for each payload and fails the Task Instance if a
later replay differs. This validates the experiment input even when the generation library
or execution process changes.

### Passive retention

Each `delay × trial` must be an independent Task Instance so a short-delay Warm cannot
refresh a long-delay sample. A useful graph is:

```text
Prime -> delay -> (Warm || Cold Control)
```

Warm and Cold Control are siblings with a common causal frontier. They may complete in
either order. The Task Instance becomes completed only when all required nodes succeeded.

Use actual request timestamps, not merely planned delay. For latency acceleration, compare
contemporaneous Cold Control and Warm:

```text
Warm acceleration = Cold Control TTFT / Warm TTFT
```

Show a 1x no-acceleration reference. Prime/Warm TTFT across a long wall-clock window is not
the retention speed control.

### Access-conditioned residency

A repeat chain models periodically accessed state:

```text
Prime -> Warm 1 -> Warm 2 -> ... -> Warm N
```

Count and interval may be matrix dimensions. This measures retention under access and must
not be described as passive TTL. Preserve every actual inter-node interval and cache/TTFT
observation.

### Repeated-hit promotion hypothesis

To explore whether repeated hits affect later retention, use a bounded matrix:

```text
warmup_count × quiet_seconds × trial
Prime -> Warmup × N -> quiet -> (Probe || Cold Control)
```

Compare different warmup counts only at the same quiet window and compatible Provider/model
cohort. The result is repeat-conditioned retention. Evidence may be consistent with
multiple cache tiers, but external measurements alone cannot establish internal tier names
or mechanisms.

## Experimental controls

Keep constant within a comparison cohort:

- Provider Profile and exact model;
- input/output token shape;
- tokenizer ID and immutable revision;
- concurrency and sampling parameters;
- payload generation method;
- task topology and control relationship.

Randomize independent family order when order could confound results, but keep randomization
deterministic and recorded. Separate Provider comparisons into aligned series; never merge
incompatible denominators or treat Provider names as semantic categories.

Use enough independent trials for variability. A single point supports only a directional
observation. Record routing, time-of-day, concurrency, and failures that could affect cache
affinity.

## Boundary for more complex observers

The current architecture is well suited to request-bound, statically planned observation:
fixed probes, control cohorts, repeated trials, parallel provider comparisons, timed
retention checks, and derived report metrics can all be expressed as a finite task graph.
This covers the present KV-cache promotion, retention, and residency experiments without
requiring cache-specific behavior in the Planner or Scheduler.

It does not yet make continuous telemetry collectors, response-dependent probes, adaptive
stopping, dynamic sampling rates, or provider-internal cache events first-class. Those
experiments need a versioned observer-result contract and, where the next graph depends on
previous evidence, bounded compilation epochs. Provider-private telemetry may strengthen
an inference, but it must remain separately identified from portable request evidence and
must not become an implicit requirement for the generic experiment model.

## Fine-grained hit analysis

Use both accounting and TTFT:

- complete token hit ratio shows the fraction of input tokens reported reused;
- request hit probability shows how often a chosen hit criterion is met;
- TTFT distribution shows performance effect;
- Cold/target ratio gives an interpretable acceleration scale;
- counter coverage determines whether accounting conclusions are representative.

A combined bar/line view may show TTFT improvement and hit ratio at the same delay or
warmup-count points when their grain is identical. It is one option, not a mandatory chart.
Aligned panels are preferable when a dual axis would imply a false relationship.

## Multi-Provider reporting

Comparable Providers should appear as separate lines or grouped marks on the same semantic
axis. Assign styles by sorted normalized identity using the stable generic palette, never by
literal Provider-name matching. Reuse the mapping in every chart and add markers/dashes
when color alone is fragile.

Curves are visual guides only. Use shape-preserving interpolation that cannot overshoot
observed points, keep points visible, and preserve gaps for missing data.

## Reliability and failure semantics

- Zero completed requests is a failed Runner.
- Missing cache counters are unknown, not misses.
- Prompt-hash mismatch fails the Task Instance.
- Failed/cancelled nodes cancel blocked and pending descendants.
- Ambiguously sent requests are not retried in the same payload family.
- Campaign lifecycle status and aggregate outcome remain separate.
- Pending long-delay nodes keep Campaign lifecycle planned.

Reports must surface failed/cancelled Runners, timeouts, request errors, counter gaps,
pending nodes, actual-vs-planned timing, and sample size.

## Six-hour experiment boundary

When an experiment is capped at six hours, bound the maximum quiet/delay point, repeat
intervals, trial count, Provider request count, and expected queue latency so every planned
node can finish inside the window. Validate the fully expanded graph against active
Scheduler and Ray capacity before submission. A six-hour wall-clock cap does not justify
unbounded loops or hidden retries.

## Export and audit

Campaign export version 1.0.0 preserves task definitions, dimensions, trials, dependency
topology, planned/actual timing, payload hashes, Runner summaries, and optional request
records. The report preparation pipeline normalizes this evidence without imposing a fixed
chart layout. Final HTML should emphasize the strongest conclusions, fold Runner detail,
and remain traceable to the source export.

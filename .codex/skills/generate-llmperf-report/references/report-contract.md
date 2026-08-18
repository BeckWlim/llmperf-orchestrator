# LLMPerf report data contract

## Pipeline boundary

`prepare_report_data.py` owns deterministic extraction and normalization. The reporting
Agent owns interpretation, chart selection, layout, and HTML assembly. Do not move fixed
experiment dashboards back into the pipeline.

Accepted inputs:

- Runner export version 1;
- Campaign export version 6.

Campaign v6 includes `task_definitions`, `task_instances`, `dispatches`, `task_analyses`,
and `runners`. It deliberately has no protocol-specific analysis collection.

The normalized document has:

- `overview`: counts, lifecycle, failures, completed requests;
- `cohorts`: comparable Provider/model/token/concurrency groups;
- `task_definitions`: submitted compile-time recipes;
- `evidence.task_graphs`: instance dimensions, trial, payload hashes, topology, timing, and
  joined Runner metrics;
- `evidence.runner_cache_probes`: within-Runner paired cache-probe evidence;
- `runners`: flat normalized detail.

Missing values remain JSON null or absent. Never coerce missing counters to zero.

## Generic task semantics

One Task Instance represents one matrix coordinate and trial. Each node is one atomic
request Runner. Interpret fields as follows:

- `role`: author-provided semantic label, not runtime behavior;
- `payload_id`: logical generated input family;
- `payload_seed`: deterministic materialization seed;
- `payload_hashes`: runtime proof that repeated payload references were identical;
- `dependencies`: causal predecessor Dispatch IDs;
- `planned_after_seconds`: requested delay after all predecessors completed;
- actual timestamps: observed request start/completion anchors;
- `dimensions` and `trial_index`: experiment comparison coordinates.

Do not branch on Provider names, compiler labels, or a fixed list of roles. Infer meaningful
comparisons from the graph and explain that inference in the report.

## Cache and latency semantics

Provider cache counters and latency answer different questions:

- token hit ratio describes Provider-reported cache accounting;
- request hit probability describes the fraction of requests satisfying an explicit hit
  rule;
- TTFT delta or control/target TTFT ratio describes observed latency improvement;
- a hit counter alone does not prove speedup;
- speedup above 1× means the target node had lower TTFT than its control.

For pooled token hit ratio, use complete counters only:

```text
sum(hit_tokens) / (sum(hit_tokens) + sum(miss_tokens))
```

Report counter coverage and sample size. Do not average percentages when token denominators
differ.

For a cross-Runner comparison, pair nodes only when Provider, model, token shape,
concurrency, tokenizer, matrix coordinates, and intended control relationship are
compatible. A contemporaneous Cold Control can support:

```text
acceleration = Cold Control TTFT / Warm-or-Probe TTFT
```

Show a labeled 1× reference. A Prime separated by a retention interval is not a valid
contemporaneous speed control.

Repeated access changes the state being measured. If Warmups precede a Probe, describe the
result as access-conditioned or repeat-conditioned retention, not passive TTL. Preserve
each repeated node and the actual inter-node intervals.

## Chart-neutral evidence rules

Before rendering, inspect cardinality, missingness, uncertainty, and compatible dimensions.
Choose the smallest visual set that supports the conclusions. Lines, bars, intervals,
heatmaps, small multiples, aligned panels, and mixed encodings are options, not mandatory
templates.

A bar-line combination is acceptable when both measures share the same experimental grain
and the joint view materially clarifies their relationship—for example TTFT improvement and
cache hit ratio at the same delay points. Avoid mixed charts that merely stack available
metrics, hide nulls, or imply correlation through arbitrary dual-axis scaling.

For multiple Providers, preserve separate series on compatible axes. Resolve styles once
per report by sorting normalized identities and assigning slots from
`assets/provider-palette.json`; never match literal Provider names. Use marker/dash
redundancy when needed.

Quantitative guardrails:

- hit-rate axes are bounded to 0–100%;
- acceleration charts include 1×;
- connecting curves must be shape-preserving and cannot invent extrema;
- observed points remain visible and null gaps remain gaps;
- “Warm acceleration” or “Warm TTFT improvement” is preferred over “Warm is faster”;
- numeric table columns and titles align consistently;
- important conclusion charts receive greater visual area;
- Runner detail is collapsed with `<details>` by default, while failures stay visible.

## Reliability and claims

Always surface failed/cancelled Runners, request errors, timeouts, pending nodes, payload hash
failures, and counter gaps. Small samples support directional language unless confidence
intervals or an equivalent statistical test are available.

Do not expose prompts, credentials, Authorization headers, private endpoints, raw stdout,
or unbounded stderr. Final HTML must be self-contained, printable, and traceable to the
source export and normalized analysis artifact.

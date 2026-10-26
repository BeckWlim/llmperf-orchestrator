# Workload examples

Choose the smallest example that matches the measurement. Set its Provider and exact model
to a configured profile before running. Examples intentionally use small bounded workloads;
increase counts only after checking Scheduler capacity and the fully expanded request budget.
Each file has a distinct main role. Family recall includes independent and sequential
batches; matrix phases cover round barriers; message merge includes multi-turn history.

## Availability and load

| Example | Purpose | Target requests |
|---|---|---:|
| [Smoke](../examples/example-smoke.yaml) | One request to verify inference and persistence | 1 |
| [Load comparison](../examples/example-campaign.yaml) | Immediate serial and concurrent Runners in one Campaign | 4 |

## Scheduling and batch composition

| Example | Purpose | Target requests |
|---|---|---:|
| [Family recall](../examples/example-family-recall.yaml) | Two stable prefix families × two independent trials × two serial batches | 16 |
| [Seed pool](../examples/example-seed-pool.yaml) | Two sequential groups × four Runners; one deterministic prefix draw per Runner | 32 |
| [Fixed cohort recall](../examples/example-cohort-recall.yaml) | Four prefixes visited once per wave, two lanes, and two delayed recall waves | 12 |
| [Matrix phases](../examples/example-matrix-phases.yaml) | Two families × load/pressure/read × two cycles; stable target prefixes | 12 |
| [RunnerPlan](../examples/example-runner-plan.yaml) | Two time-based occurrences one second apart | 2 |

These examples have different ordering contracts. `trials` creates independent workflows;
`repeat` serializes iterations inside each workflow; `campaign.rounds` waits for the whole
round; matrix phases wait for every member of the preceding phase to become terminal.
Fixed cohort recall enumerates every seed once per wave; seed-pool draws sample with
replacement and may miss or revisit seeds. RunnerPlans follow time-based recurrence.
See [task rules](tasks.md) and [RunnerPlan scheduling](scheduling.md).

## Cache measurements and conversations

| Example | Purpose | Target requests |
|---|---|---:|
| [Paired cache probe](../examples/example-cache-probe.yaml) | Two Prime/Warm pairs inside one Runner, with paired cache analysis | 4 |
| [Cache replay](../examples/example-cache.yaml) | Sequence, repeat, delay matrix, parallel Warm/Cold, and exact payload replay | 16 |
| [Dataset](../examples/example-dataset.yaml) | Pinned external corpus and deterministic concatenation with replay | 3 |
| [Message merge](../examples/example-message-merge.yaml) | Multi-turn history from two independent requests, an ordered join, and a delay | 3 |

The cache example combines the former retention, residency, and promotion examples.
`warmup_count: 0` observes a quiet interval after prime; positive warmup counts introduce
repeated access before that interval. For a repeated observation chain, use the repeat
primitive and completion-anchored `every` described in [task rules](tasks.md).
Keep these measurement conditions distinct when interpreting results; see
[cache measurement](cache-measurement.md).

The dataset example defaults to 2,048 new-input tokens. Set `inputs.default.tokens: 65536`
for a 64K generation target after checking Provider and Scheduler capacity.
For larger load, increase a Runner's request target and concurrency after the smoke check,
using the limits and capacity calculations in [Runtime](runtime.md#capacity).

The 128-token family, cohort, and matrix examples demonstrate composition and seed
identity. They do not establish cache hits or eviction. Increase input sizes only after
checking the Provider's context limits and cache behavior. Cache probes also need enough
trials and available Provider counters before their results support a cache conclusion.

## Validate and run

For a standalone Runner, use `llmperfctl runner start -f FILE --wait`. For a Campaign:

```bash
llmperfctl campaign preview -f examples/example-family-recall.yaml --debug
llmperfctl campaign validate -f examples/example-family-recall.yaml
llmperfctl scheduler status
# Starts inference:
llmperfctl campaign start -f examples/example-family-recall.yaml --wait
```

The counts above are targets, not guaranteed successful completions. Provider errors,
timeouts, cancellation, and instance failures can reduce completed requests. Preview and
artifact validation do not prove inference availability. Use the smoke example first.

## Inspect the evidence

After a Campaign finishes, inspect its outcome and export the per-request measurements:

```bash
llmperfctl campaign status CAMPAIGN_ID
llmperfctl runner logs RUNNER_ID
llmperfctl campaign export CAMPAIGN_ID --include-requests -o output/example-results.json
```

Use the returned Campaign ID and a Runner ID from its status. Check aggregate outcome
as well as lifecycle status: `completed` does not imply every Runner succeeded.
Compare concurrency, latency, errors, and token-count coverage in the exported results;
for cache experiments, also inspect prompt identity and cache-counter coverage.
The [report skill](../.codex/skills/generate-llmperf-report/SKILL.md) turns those exports
into HTML reports. Provider isolation and runtime guards are configured in the Backend;
see [Provider setup](usage.md) and [runtime controls](runtime.md#performance-guard).

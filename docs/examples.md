# Workload examples

Choose the smallest example that matches the measurement. Set its Provider and exact model
to a configured profile before running. Examples intentionally use small bounded workloads;
increase counts only after checking Scheduler capacity and the fully expanded request budget.

## Availability and load

| Example | Purpose | Target requests |
|---|---|---:|
| [Smoke](../examples/example-smoke.yaml) | One request to verify inference and persistence | 1 |
| [Mixed Campaign](../examples/example-campaign.yaml) | Immediate Runner comparison plus an atomic replay workflow | 6 |

## Scheduling and batch composition

| Example | Purpose | Target requests |
|---|---|---:|
| [Independent batches](../examples/example-batch-parallel.yaml) | Matrix × trials; one independent batch per instance | 16 |
| [Sequential batches](../examples/example-batch-sequence.yaml) | Two independent families, each with two serial batches | 16 |
| [Campaign rounds](../examples/example-rounds.yaml) | Two Runner templates repeated twice with a barrier between rounds | 8 |
| [RunnerPlan](../examples/example-runner-plan.yaml) | Two time-based occurrences one second apart | 2 |

These examples have different ordering contracts. `trials` creates independent workflows;
`repeat` serializes iterations inside each workflow; `campaign.rounds` waits for the whole
round; RunnerPlans follow time-based recurrence. See [task rules](tasks.md) and
[RunnerPlan scheduling](scheduling.md).

## Cache measurements and conversations

| Example | Purpose | Target requests |
|---|---|---:|
| [Cache replay](../examples/example-cache.yaml) | Sequence, repeat, delay matrix, parallel Warm/Cold, and exact payload replay | 16 |
| [Dataset](../examples/example-dataset.yaml) | Pinned external corpus and deterministic concatenation with replay | 3 |
| [Multi-turn](../examples/example-multiturn.yaml) | One Runner, two Workers, inherited history, and a millisecond delay | 2 |
| [Message merge](../examples/example-message-merge.yaml) | Two independent requests feeding one request through ordered message reuse | 3 |

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

## Validate and run

For a standalone Runner, use `llmperfctl runner start -f FILE --wait`. For a Campaign:

```bash
llmperfctl campaign preview -f examples/example-batch-parallel.yaml --debug
llmperfctl campaign validate -f examples/example-batch-parallel.yaml
llmperfctl scheduler status
# Starts inference:
llmperfctl campaign start -f examples/example-batch-parallel.yaml --wait
```

The counts above are targets, not guaranteed successful completions. Provider errors,
timeouts, cancellation, and instance failures can reduce completed requests. Preview and
artifact validation do not prove inference availability. Use the smoke example first.

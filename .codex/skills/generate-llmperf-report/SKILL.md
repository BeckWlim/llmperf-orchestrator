---
name: generate-llmperf-report
description: "Prepare auditable LLMPerf analysis data and create style-consistent, self-contained HTML reports. Use when Codex needs to export or interpret Campaign/Runner records, maintain the report data pipeline, analyze KV-cache retention or speedup, compare providers or rounds, select evidence-led charts, or produce a shareable benchmark report."
---

# Generate LLMPerf Report

Build the evidence model deterministically, then let the reporting Agent choose the
narrative and visual structure that best fits the experiment.

## Prepare the analysis model

Use [scripts/prepare_report_data.py](scripts/prepare_report_data.py) as the deterministic
pipeline CLI:

```bash
.venv/bin/python .codex/skills/generate-llmperf-report/scripts/prepare_report_data.py \
  --campaign-id CAMPAIGN_ID --output /tmp/analysis.json

.venv/bin/python .codex/skills/generate-llmperf-report/scripts/prepare_report_data.py \
  --input campaign.json --output /tmp/analysis.json
```

The analysis document normalizes Runner summaries, reliability, comparable cohorts,
compiled task graphs, dimensions, payload identity, actual timing, cache counters, and
latency evidence without prescribing an experiment interpretation or chart layout.
Inspect it before choosing charts or writing conclusions.

Keep `--include-requests` off unless request distributions or outliers require it. Never
pass credentials through command arguments. Use existing `llmperfctl` authentication.

## Render for the evidence

Choose the HTML structure after inspecting the analysis model. Do not impose a fixed chart
list or dashboard grid.

- Lead with the few conclusions that answer the experiment question.
- Give decisive charts full-width or otherwise greater visual priority.
- Compare multiple Providers as separate series on the same meaningful axis when cohorts
  are compatible; preserve stable colors, legends, null gaps, and units. Assign colors by
  normalized Provider identity, not series order. Resolve the mapping from the machine-
  readable [assets/provider-palette.json](assets/provider-palette.json): sort unique
  normalized identities once, assign the generic series slots in declared order, and reuse
  that map in every chart. The skill must not special-case Provider names. Add the specified
  marker shapes or dash patterns when lines overlap or color-only identification is fragile.
- Choose the axis from experiment semantics: delay for retention, scheduled offset for
  residency, concurrency for load comparisons, or round/time for repetition.
- Make quantitative bounds and decision references explicit: hit-ratio axes end at 100%,
  and speedup charts include a labeled 1× no-acceleration reference. Use “Warm
  acceleration” or “Warm TTFT improvement” for ratios above 1×; do not label this state
  “Warm is faster”.
- Smooth lines only with a shape-preserving curve that cannot overshoot the observations.
  Show the observed points and state that connecting curves are visual guides, not extra
  measurements; preserve gaps for null values.
- Treat chart forms as an evidence-dependent grammar, not a catalog to exhaust. Before
  rendering, write a short chart brief for each candidate: the claim it supports, data
  grain, comparison dimension, metric semantics, uncertainty, missingness, and intended
  visual priority. Choose the smallest complementary set from lines, bars, points,
  intervals, heatmaps, small multiples, aligned panels, or a justified mixed encoding.
  Compose metrics only when they share a meaningful grain, cohort, or comparison axis and
  become easier to interpret together. A bar-line cache/TTFT view is one possible example,
  not a required pattern. Reject combinations that merely stack available metrics, require
  too many legends or axes, obscure nulls, or imply correlation through arbitrary scaling.
  Label every encoding and reference directly; split into aligned panels when that is
  clearer than a single combined chart.
- Omit charts that add no decision value. Use tables when exact mappings are clearer.
- Align table headers with their data: descriptive columns left, numeric columns right,
  and intentionally centered categorical columns centered in both header and body.
- Put Runner detail in a collapsed `<details>` bar by default; keep failures visible in a
  separate diagnostic section.
- Inline [assets/report-theme.css](assets/report-theme.css) to keep typography,
  spacing, colors, KPI cards, chart containers, tables, badges, and print behavior
  consistent. Extend its classes when the evidence needs a different visualization.

There is intentionally no fixed HTML generator. Build the self-contained HTML from the
analysis model, chosen evidence, and shared theme. This keeps experiment-specific rendering
decisions with the reporting Agent instead of embedding them in a compatibility surface.

## Preserve statistical semantics

Read [references/report-contract.md](references/report-contract.md) completely before
interpreting an unfamiliar schema, changing the pipeline, or rendering cache evidence.

- For Runner-local `cache_probe`, use its paired Prime/Warm analysis and verdict.
- For compiled tasks, derive experiment meaning from task dimensions, role tags,
  dependencies, payload IDs, and actual timestamps. Do not branch on a compiler or
  Provider name.
- Verify exact replay from a shared payload ID and persisted payload hash before comparing
  Prime/Warm/Probe nodes. Use contemporaneous controls at the same graph grain when
  computing acceleration; do not use Prime across a long wall-clock window as the control.
- Preserve every repeated node and compare like-for-like matrix coordinates. Label
  access-conditioned results separately from passive retention.
- Compute pooled token hit ratio from summed complete hit/miss tokens. Keep request hit
  probability separate and disclose counter coverage.
- Preserve null as unavailable. Never coerce missing counters or metrics to zero.
- Split incompatible Provider/model/concurrency/token/tokenizer/task-shape cohorts.
- Describe small-sample latency results as directional unless the exported analysis
  contains the required statistical evidence for a stronger verdict.

## Validate the deliverable

Verify the HTML is non-empty and self-contained, the analysis JSON is valid, and counts
match the source export. Report failed/cancelled Runners, request errors, timeouts,
counter gaps, pending task nodes, and whether acceleration was actually confirmed.
Keep stdout/stderr, prompts, credentials, Authorization headers, and private endpoints out
of both artifacts.

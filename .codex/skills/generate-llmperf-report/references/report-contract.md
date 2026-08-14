# LLMPerf HTML report contract

## Contents

1. Supported exports
2. Metric mapping
3. KV-cache interpretation
4. Quality and comparison rules
5. Report presentation contract

## 1. Supported exports

### Campaign export version 5

Create with:

```bash
llmperfctl campaign export CAMPAIGN_ID -o campaign.json
```

The root contains `campaign`, `aggregate`, `runner_plans`, `runners`,
`protocol_definitions`, `protocol_instances`, `dispatches`, and
`protocol_analyses`. Cache-retention curves are protocol analyses rather than
specialized Campaign fields. `cache-residency/v1` curves are access-conditioned;
render their geographic scheduled time, planned offset, and actual Prime-to-Warm
delay separately, and never label them as passive TTL.
`aggregate.status` is lifecycle state; `aggregate.outcome` is aggregate execution
result. Each Runner stores benchmark configuration under `benchmark`, aggregate
benchmark output under `summary`, and optional request records under `requests`.

The default export omits `requests` but retains `summary.results` and
`summary.cache_probe_analysis`. Add `--include-requests` only when the report
requires request distributions, outlier inspection, or paired request evidence.

### Runner export version 1

Create with:

```bash
llmperfctl runner export RUNNER_ID -o runner.json
```

The root contains `runner` metadata and `results`. The benchmark summary is
`results.summary`; individual metrics are `results.requests`. Runner export is
available only when a succeeded or failed Runner has a persisted summary.

## 2. Metric mapping

Use these summary paths. Missing values remain unavailable rather than zero.

| Report concept | Summary path |
|---|---|
| Started requests | `results.num_requests_started` |
| Completed requests | `results.num_completed_requests` |
| Request errors | `results.number_errors` |
| Error rate | `results.error_rate` |
| TTFT | `results.ttft_s` |
| End-to-end latency | `results.end_to_end_latency_s` |
| Request output throughput | `results.request_output_throughput_token_per_s` |
| Overall output throughput | `results.mean_output_throughput_token_per_s` |
| Inter-token latency | `results.inter_token_latency_s` |
| KV-cache aggregate | `results.kv_cache` |
| Paired cache evidence | `cache_probe_analysis` |
| Timeout flag | `timed_out` and `cache_probe_analysis.quality_flags.timed_out` |

Latency distribution objects expose `min`, `max`, `mean`, `stddev`, and
`quantiles.p25/p50/p75/p90/p95/p99`. Multi-round charts should graph per-Runner
summary values rather than pool incompatible quantiles. Label cross-round
statistics as medians or ranges across Runners, not request-level percentiles.

For reliability, sum started/completed/error counts across comparable Runners.
The Campaign export field `aggregate.completed_request_count` is the number of
persisted request records in the current implementation; do not present it as
the count of successful model responses.

## 3. KV-cache interpretation

Prefer paired warm-request cache evidence from `summary.cache_probe_analysis`.
Use:

- `cache.weighted_token_hit_ratio` for the warm token hit ratio;
- `cache.counter_coverage` to disclose measurement coverage;
- `speedup.p50` for median `prime_ttft / warm_ttft`;
- `paired_ttft_delta_s.p50` for median `prime_ttft - warm_ttft`;
- `paired_ttft_delta_s.confidence_interval` for latency evidence;
- `quality_flags` and `paired_samples` for evidence quality.

Verdict meanings:

| Verdict | Allowed conclusion |
|---|---|
| `confirmed_external` | Cache hits and statistically positive paired latency improvement were observed. |
| `accounting_confirmed` | Provider cache counters confirm reuse; paired latency improvement is not established. |
| `latency_inferred` | Latency evidence is positive but provider counters are unavailable. |
| `not_observed` | Adequate counters were present but no cache hit was observed. |
| `inconclusive` | Evidence coverage or sample quality is insufficient. |

Compute a Campaign-wide warm hit ratio only by summing complete warm hit and miss
tokens, then dividing hit tokens by their sum. Do not average per-Runner ratios.

## 4. Quality and comparison rules

- Keep lifecycle and outcome distinct. `completed` does not imply `succeeded`.
- A succeeded Runner may contain request errors or a degraded outcome.
- Surface all failed/cancelled Runners, request errors, timed-out summaries,
  skipped dependent cache requests, tokenizer mismatches, and invalid counters.
- Report cache coverage next to cache hit ratio. A high ratio with low coverage is
  weak evidence.
- Do not compare Runners as one series when provider, model, concurrency, input or
  output token targets, tokenizer provenance, cache-probe mode, or sampling
  parameters differ. Split into cohorts or disclose the mismatch.
- Do not infer provider performance from infrastructure failures such as Ray OOM,
  scheduler cancellation, artifact resolution, proxy errors, or local timeouts.
- Bound diagnostic messages and HTML-escape them. Never render raw stdout/stderr,
  prompt text, credentials, Authorization headers, or private endpoints.

## 5. Report presentation contract

A professional report contains:

1. Title, experiment ID, provider/model scope, generation time, and export schema.
2. Executive findings written with evidence strength and caveats.
3. KPI cards for Runner outcome, request reliability, latency, throughput, and
   KV-cache behavior when those metrics exist.
4. Accessible inline SVG charts with units, legends, null gaps, and point values.
5. A chronological Runner table that links round, status, request counts, latency,
   throughput, cache evidence, and verdict.
6. Data-quality and failure diagnostics separated from model-performance claims.
7. Methodology/provenance notes sufficient to reproduce the export.

Generate one self-contained UTF-8 HTML file. Do not require a web server, CDN,
remote font, external stylesheet, or JavaScript library.

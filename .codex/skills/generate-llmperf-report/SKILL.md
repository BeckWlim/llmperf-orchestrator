---
name: generate-llmperf-report
description: "Generate professional, self-contained HTML analysis reports with charts from LLMPerf experiment records. Use when Codex needs to export a Campaign or Runner with llmperfctl, turn Campaign export v3 or Runner export v1 JSON into an offline HTML report, compare multi-round latency/throughput/KV-cache behavior, summarize request reliability and failures, or prepare an auditable benchmark report for review or sharing."
---

# Generate LLMPerf Report

Turn durable LLMPerf records into one offline HTML file with executive findings,
data-quality warnings, inline SVG charts, per-Runner metrics, and failure diagnostics.

## Generate the report

1. Prefer a Campaign ID for multi-round analysis. Use a Runner ID only for a
   single execution. Reuse an existing export JSON when one is already available.
2. Run [scripts/generate_report.py](scripts/generate_report.py). It invokes
   `llmperfctl` without a shell, or reads the supplied JSON directly.
3. Keep the default summary-only Campaign export unless request-level analysis is
   explicitly needed. Add `--include-requests` only for that case.
4. Verify that the output exists, is non-empty, and the script reports the
   expected Runner/request counts. Return a clickable path to the HTML file.
5. State material quality limits alongside the file: failed/cancelled Runners,
   degraded request outcomes, timeouts, incomplete cache-counter coverage, and
   whether cache acceleration was actually confirmed.

Use one of these forms from the repository root:

```bash
.venv/bin/python .codex/skills/generate-llmperf-report/scripts/generate_report.py \
  --campaign-id CAMPAIGN_ID --output reports/campaign.html

.venv/bin/python .codex/skills/generate-llmperf-report/scripts/generate_report.py \
  --runner-id RUNNER_ID --output reports/runner.html

.venv/bin/python .codex/skills/generate-llmperf-report/scripts/generate_report.py \
  --input campaign-report.json --output reports/campaign.html
```

Pass global CLI connection options before the LLMPerf command with repeatable
`--llmperfctl-arg`, for example:

```bash
.venv/bin/python .codex/skills/generate-llmperf-report/scripts/generate_report.py \
  --campaign-id CAMPAIGN_ID --output reports/campaign.html \
  --llmperfctl-arg=--url --llmperfctl-arg=http://127.0.0.1:8000
```

Never pass tokens through command arguments or include them in the report. Use
the existing `llmperfctl` authentication discovery or environment configuration.

## Interpret results conservatively

- Keep Campaign lifecycle `status` separate from aggregate `outcome`.
  `completed/partial_failed` means the workload ended with at least one failure.
- Treat Runner `succeeded` as successful persistence of a benchmark summary, not
  proof that every request succeeded or that the target request count completed.
- Distinguish `num_requests_started`, `num_completed_requests`, and
  `number_errors`; do not label persisted request records as successful requests.
- Claim externally observed KV-cache acceleration only for
  `cache_probe_analysis.verdict=confirmed_external`. `accounting_confirmed`
  confirms provider cache accounting but not a statistically supported latency win.
- Preserve null as unavailable. Never coerce unknown cache counters or missing
  latency statistics to zero.
- Compare like-for-like Runners. Call out differing provider, model, concurrency,
  prompt/token settings, tokenizer provenance, or cache-probe configuration.

## Customize or diagnose

Read [references/report-contract.md](references/report-contract.md) completely
before changing the generator, interpreting an unfamiliar export schema, adding
new charts, or writing a custom narrative beyond the generated findings.

The generated HTML must remain self-contained: inline CSS and SVG only, no CDN,
remote JavaScript, tracking pixels, external fonts, or embedded secrets. Keep raw
stdout/stderr and request prompt text out of the report; include only bounded,
HTML-escaped diagnostic summaries.

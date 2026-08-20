# Evidence-locked rendering review

- [Trust boundary](#trust-boundary)
- [Rendering sequence](#rendering-sequence)
- [Render plan shape](#render-plan-shape)
- [Review the final report](#review-the-final-report)
- [Approve design-system changes separately](#approve-design-system-changes-separately)

## Trust boundary

Begin every rendering pass by reopening the normalized `analysis_version: 1.0.0` document.
Use that document as the only factual source for report claims, available fields,
comparisons, and experiment semantics.

Previous conversation may contribute only:

- the current report question or audience;
- an explicit presentation constraint, such as language or page size;
- an explicitly approved shared design rule already stored in this skill.

Do not reuse a prior claim, chart type, layout, metric, data relationship, or conclusion
unless the current analysis independently supports it. Do not infer that a field exists
because it appeared in an earlier report. When a user assertion conflicts with normalized
evidence, show the discrepancy instead of bending the rendering to match the assertion.

## Rendering sequence

1. Generate a structural inventory with `scripts/review_render_plan.py --inventory`.
2. Inspect the analysis itself for values, cohorts, missingness, and task-graph semantics.
3. Write a render plan with `evidence_policy: normalized-analysis-only` and the inventory's
   `analysis_sha256`.
4. Bind every claim to one or more analysis paths. Bind every chart to claims and data
   paths. Use `[]` for array traversal, for example
   `evidence.task_graphs[].nodes[].runner.ttft_p50`.
5. Run the plan review and resolve all errors before writing HTML.
6. Render from the reviewed plan and current analysis, not from conversational memory.
7. Embed the analysis hash and plan IDs in the HTML, then rerun the review with `--html`.
8. Reopen the final HTML and perform the human review below.

The render plan is a working artifact, not a second source of truth. If it conflicts with
the analysis, change the plan. If the analysis file changes, its hash invalidates the plan.

## Render plan shape

Use this minimal structure:

```json
{
  "render_plan_version": "1.0.0",
  "analysis_sha256": "<sha256 from inventory>",
  "evidence_policy": "normalized-analysis-only",
  "objective": "Decision the report should support",
  "claims": [
    {
      "id": "claim-retention",
      "statement": "Bounded statement to present",
      "evidence_paths": ["evidence.task_graphs[].nodes[].runner.cache_hit_ratio"],
      "qualification": "Missingness, sample-size, or compatibility limit"
    }
  ],
  "charts": [
    {
      "id": "chart-retention",
      "claim_ids": ["claim-retention"],
      "data_paths": [
        "evidence.task_graphs[].dimensions.delay_seconds",
        "evidence.task_graphs[].nodes[].runner.cache_hit_ratio"
      ],
      "grain": "One compatible task node per delay and trial",
      "comparison_dimension": "Delay within a compatible cohort",
      "metric_semantics": "Provider-reported complete-token hit ratio",
      "uncertainty": "Show observed points; describe small samples as directional",
      "missingness": "Preserve nulls as gaps",
      "visual_priority": "primary"
    }
  ]
}
```

Use `primary` or `supporting` for `visual_priority`. Tables and prose may carry claim IDs
without a chart. Keep IDs unique and stable within one rendering pass.

## Review the final report

Perform these checks after the deterministic review succeeds:

- Trace every headline, KPI, finding, annotation, and chart back to its claim binding and
  current analysis values.
- Confirm grouping uses compatible cohorts and the intended task-graph grain.
- Confirm labels distinguish observed, derived, adapted, and unavailable values.
- Confirm nulls remain gaps or unavailable markers and do not become zero.
- Confirm axes, units, denominators, controls, intervals, and sample sizes match metric
  semantics.
- Confirm failures, timeouts, counter gaps, pending nodes, and compatibility limitations
  remain visible.
- Remove any decorative visualization that lacks a decision-supporting claim.
- Confirm the report is self-contained, printable, keyboard-readable, and understandable
  without color alone.
- Search for prompts, credentials, Authorization headers, private endpoints, stdout, and
  unbounded stderr before delivery.

## Approve design-system changes separately

Keep report approval separate from skill approval. Before changing shared CSS, palettes,
templates, or rendering rules, show a concrete proposal containing:

- the exact reusable rule or token change;
- why it generalizes across experiments and Providers;
- which shared files would change;
- a representative before/after effect and known tradeoffs.

Require an unambiguous authorization to persist that proposal into the skill. Statements
such as "this report looks good" or "use this for the current report" approve only the
deliverable. After approval, add the smallest provider-neutral rule, test it, validate the
skill, and avoid storing report-specific values, Provider names, or conclusions.

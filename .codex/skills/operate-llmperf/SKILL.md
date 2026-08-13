---
name: operate-llmperf
description: "Configure, operate, diagnose, and modify this LLMPerf project. Use when working with llmperfctl commands, Runner or Campaign YAML, bounded RunnerPlans and geographic-time scheduling, Provider Profiles, tokenizer or dataset resolution, KV-cache probes, Campaign lifecycle/outcome interpretation, Scheduler/Planner/Worker behavior, exports, logs, PostgreSQL persistence, or repository tests and implementation conventions."
---

# Operate LLMPerf

Use the repository's durable execution model consistently when authoring workloads,
running commands, diagnosing failures, or changing implementation.

## Load the relevant reference

- Read [references/yaml.md](references/yaml.md) completely before creating,
  reviewing, or changing Runner/Campaign/RunnerPlan YAML.
- Read [references/cli.md](references/cli.md) completely before operating the
  Backend or CLI, diagnosing a run, interpreting status, or exporting results.
- Read [references/engineering.md](references/engineering.md) completely before
  changing source, database behavior, tests, architecture, or project docs.
- Read more than one reference when the task crosses those boundaries.

## Preserve the execution model

Apply these invariants:

1. Treat a Campaign as the durable workload boundary. It may contain immediate
   Runners, bounded RunnerPlans, or both.
2. Treat a RunnerPlan as a template. The Planner materializes due occurrences as
   ordinary queued Runners; it never occupies a Scheduler slot while waiting.
3. Treat the Scheduler as the queue consumer and Worker owner. A Worker is a
   temporary subprocess for exactly one claimed Runner.
4. Persist control state and benchmark results in PostgreSQL. Do not add SQLite
   fallback behavior or in-memory authoritative state.
5. Keep Provider endpoints and credentials in Backend-owned profiles. Put only
   the stable provider ID and model ID in workload YAML.
6. Distinguish Campaign lifecycle `status` from aggregate execution `outcome`.
   Never infer that `completed` means every Runner succeeded.

## Follow the task workflow

1. Inspect `git status --short` and preserve unrelated user changes.
2. Read the active Pydantic models and CLI help when exact accepted fields or
   options matter; repository documentation may lag implementation.
3. Use the current Aliyun `deepseek-v4-pro` examples as the operational baseline.
4. Validate YAML structure before submission. Remember that Backend validation
   may resolve Provider, tokenizer, and dataset artifacts before accepting the
   complete workload.
5. For implementation work, make the smallest coherent cross-layer change:
   model, persistence, API, CLI, docs, and focused tests as applicable.
6. Run focused tests, then the complete test suite. Do not silently substitute a
   different database or provider.
7. Report whether Backend restart, PostgreSQL configuration, artifact download,
   or credentials are required for the change to take effect.

## Diagnose from the first concrete failure

- Inspect `runner status --summary`, then `runner logs` for Worker/Ray exceptions.
- Treat HTTP status nested through the Backend as the provider response, not
  necessarily as a Backend authentication failure.
- Treat `/models` discovery as catalog visibility only; prove inference support
  with a bounded smoke Runner.
- For long `-w` commands, remember that logs are rendered by HTTP polling and
  change detection; exiting the CLI stops observation, not durable execution.
- Avoid masking a failed request as a zero-request success. Preserve the first
  request error and Worker process details in the Runner result.

## Keep outputs safe

- Never print or persist API keys, bearer tokens, or private keys.
- Use `llmperf-backend config set ... --stdin` for secrets.
- Keep JSON/table output on stdout and operational logs on stderr.
- Use `--log-level debug` only when HTTP timing and response metadata are needed;
  credentials must remain redacted.

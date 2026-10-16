# Documentation

Use one guide for each boundary. YAML rules live in the task guide; runtime behavior and
storage live in the runtime guide. Implementation plans that preceded the current code
have been consolidated into these guides; their original text remains in Git history.

| Start here | Covers |
|---|---|
| [Usage and configuration](usage.md) | Installation, Providers, CLI operations, datasets, and metrics |
| [Examples](examples.md) | One catalog of runnable YAML, request budgets, and selection guidance |
| [Task compiler and YAML rules](tasks.md) | Instances, workflows, Runner templates, seeds, limits, failure boundaries, and compiler status |
| [Architecture](architecture.md) | Component ownership, control plane, and reporting |
| [Runtime and storage](runtime.md) | Request capacity, durable records, multi-turn history, and recovery |
| [RunnerPlan scheduling](scheduling.md) | Recurrence, timezones, misfires, and overlap policies |
| [KV-cache measurement](cache-measurement.md) | Retention, residency, promotion, controls, and evidence interpretation |
| [Testing](testing.md) | Test ownership and opt-in PostgreSQL/Ray checks |

Deployment instructions remain beside their assets in
[deploy/systemd](../deploy/systemd/README.md). Report production instructions remain in the
[report skill](../.codex/skills/generate-llmperf-report/SKILL.md).

## Proposed development

[Runner tasks, atomic Workers, and messaging](plans/runner-worker-model.md) defines the
proposed complete-task Runner, atomic Worker steps, unified pressure/multi-turn scheduling,
dynamic creation, and reliable messaging boundary. It includes implementation phases and
acceptance checks; it is a plan, not the current runtime contract.

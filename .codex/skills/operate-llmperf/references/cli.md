# LLMPerf CLI Operations

## Preflight

Use read-only checks first:

```bash
llmperfctl health
llmperfctl scheduler status
llmperfctl planner runtime
llmperfctl provider list
llmperfctl provider models PROVIDER_ID
```

Provider catalog visibility does not prove inference. Run a one-request smoke before a
larger workload. Do not mutate configuration unless requested and authorized.

## Runner operations

```bash
llmperfctl runner start -f examples/example-smoke.yaml
llmperfctl runner start -f examples/example-smoke.yaml --wait
llmperfctl runner status RUNNER_ID
llmperfctl runner logs RUNNER_ID
llmperfctl runner cancel RUNNER_ID
llmperfctl runner export RUNNER_ID -o runner.json
```

A successful command submission is not a successful benchmark. Inspect Runner terminal
status, outcome, completed requests, errors, timeout, and logs. Use `--include-requests`
only when request distributions or outliers are needed.

## Campaign and task operations

```bash
llmperfctl campaign preview -f examples/example-cache-promotion.yaml
llmperfctl campaign preview -f examples/example-cache-promotion.yaml \
  --limit 50 --node-limit 200 --debug
llmperfctl campaign validate -f examples/example-campaign.yaml --artifact-timeout 3600
llmperfctl campaign start -f examples/example-campaign.yaml
llmperfctl campaign start -f examples/example-cache-promotion.yaml --wait
llmperfctl campaign list
llmperfctl campaign status CAMPAIGN_ID
llmperfctl campaign cancel CAMPAIGN_ID
llmperfctl campaign export CAMPAIGN_ID -o campaign.json
```

`campaign preview` sends the Campaign document to the Backend's authoritative TaskCompiler
and displays a bounded ASCII expansion without resolving artifacts or writing Campaign,
TaskInstance, Dispatch, or Runner records. The summary always reports total expanded
instances and nodes separately from immediate Runners; `--limit` bounds displayed instances across the Campaign,
`--node-limit` bounds displayed nodes per instance, `--debug` adds deterministic payload
seeds, and `--json` returns the registered structured projection.

Run `campaign validate` before a long-context Campaign whose Backend-owned tokenizer or
dataset may not be cached. It resolves all workload references, fully reads and hashes the
materialized cache files, rejects `.incomplete`, empty, unreadable, or changing artifacts,
and returns integrity evidence without creating a Campaign, RunnerPlan, Dispatch, or
Runner. Dataset validation additionally executes the configured adapter, persists its
normalized Arrow index through Hugging Face Datasets, and returns the adapter and usable
record count. Its workload projection reports the same immediate Runner, RunnerPlan,
TaskDefinition, expanded TaskInstance, and TaskNode counts as preview/start.
`--artifact-timeout` is deliberately separate from the ordinary Backend
request timeout. After validation succeeds, `campaign start` should use raw-artifact and
Arrow-index cache hits. While the Backend downloads an uncached artifact, the validation
endpoint streams path-free byte counts and heartbeats; `llmperfctl` renders them as a
dynamic downloaded/total byte counter on terminal stderr, without a progress bar. When
stderr is redirected, every event becomes a structured `completed_bytes/total_bytes` log
record. The final human or JSON result remains on stdout.

Campaign YAML may contain immediate `runners`, bounded `runner_plans`, and compiled
`task_definitions`. The CLI forwards task recipes; Backend validation, artifact resolution,
safety assessment, compilation, and persistence remain authoritative.
On successful submission, the CLI logs one authoritative workload summary that separates
immediate Runners, RunnerPlans, TaskDefinitions, expanded TaskInstances, and TaskNodes.
TaskNodes are durable Dispatches and become ordinary Runners only when their dependencies
and due times are satisfied.
Every Runner or Campaign YAML document declares `version: "1.0.0"`; other or missing
versions fail locally before submission.

Campaign `status` describes lifecycle. `outcome` describes aggregate execution and is one
of `pending`, `succeeded`, `partial_failed`, `failed`, `cancelled`, or `no_runs`. A completed
Campaign may have `partial_failed` outcome. Wait commands return exit code 2 for an
unsuccessful terminal outcome while preserving durable results.
Campaign list, status, wait, and cancellation projections retain TaskInstance and Dispatch
counts; a materialized Runner count alone is not treated as the size of a compiled task.

Campaign export version 1.0.0 contains `task_definitions`, `task_instances`, `dispatches`,
generic `task_analyses`, and `runners`. A Task Instance with completed parents and future
children remains lifecycle `planned`. Experiment meaning comes from dimensions, role tags,
payload identity, dependency topology, and actual timestamps—not fixed experiment names.

## RunnerPlan operations

```bash
llmperfctl planner preview -f examples/example-runner-plan.yaml
llmperfctl planner status RUNNER_PLAN_ID
llmperfctl planner events RUNNER_PLAN_ID
llmperfctl planner pause RUNNER_PLAN_ID
llmperfctl planner resume RUNNER_PLAN_ID
llmperfctl planner cancel RUNNER_PLAN_ID
```

Preview wall-clock/DST behavior and bound occurrences before submission. A RunnerPlan waits
without occupying a Scheduler slot. Planner materializes due occurrences as normal queued
Runners; Scheduler owns execution.

## Waiting and logs

`--wait` polls durable state and logs only snapshot changes. Retain the durable ID so a CLI
disconnect can reconnect without affecting execution. Use bounded poll intervals and
timeouts appropriate to the workload; a client timeout does not cancel the Campaign.

Inspect the first failed Runner before editing the workload:

```bash
llmperfctl runner status RUNNER_ID
llmperfctl runner logs RUNNER_ID
```

Do not dump complete stdout/stderr through ordinary status commands.

## Output modes

- Default: human-readable projected table or summary.
- `--json`: the same stable projection as JSON.
- `--full`: an expanded allow-listed diagnostic projection.
- `logs`: bounded Worker streams.
- `export -o`: versioned complete artifact written to a file.

Operational progress and durable IDs go to stderr. Queries use stdout. Never parse the
human table when `--json` or export is available.

`scheduler status` follows the same boundary: its default view is a compact Scheduler,
Ray-capacity, and performance-guard summary; `scheduler status --json` serializes the same
allow-listed projection. Human-readable utilization ratios use two-decimal percentages;
JSON retains the original 0-1 numeric ratios. Backend Ray addresses and private resource
labels are excluded.

## Failure triage

- 401/403: verify CLI signing config and Provider key boundary.
- 404/unknown model: verify exact Profile/model and API base path.
- 422: inspect every validation location; no partial workload should exist.
- zero completed requests: inspect first request error and Worker logs.
- queued growth: inspect Scheduler slots, performance guard, Ray capacity, and Campaign
  fairness before increasing capacity.
- pending task nodes: inspect dependency status, due time, and payload-hash failure.
- configuration changes not visible: distinguish Provider hot reload from full Backend
  restart settings.

Never place tokens, keys, private-key material, Provider endpoints, or database credentials
in command arguments or report artifacts.

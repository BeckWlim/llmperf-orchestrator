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
llmperfctl campaign start -f examples/example-campaign.yaml
llmperfctl campaign start -f examples/example-cache-promotion.yaml --wait
llmperfctl campaign list
llmperfctl campaign status CAMPAIGN_ID
llmperfctl campaign cancel CAMPAIGN_ID
llmperfctl campaign export CAMPAIGN_ID -o campaign.json
```

Campaign YAML may contain immediate `runners`, bounded `runner_plans`, and compiled
`task_definitions`. The CLI forwards task recipes; Backend validation, artifact resolution,
safety assessment, compilation, and persistence remain authoritative.
Every Runner or Campaign YAML document declares `version: "1.0.0"`; other or missing
versions fail locally before submission.

Campaign `status` describes lifecycle. `outcome` describes aggregate execution and is one
of `pending`, `succeeded`, `partial_failed`, `failed`, `cancelled`, or `no_runs`. A completed
Campaign may have `partial_failed` outcome. Wait commands return exit code 2 for an
unsuccessful terminal outcome while preserving durable results.

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

---
name: operate-llmperf
description: "Turn benchmark requirements into safe, runnable LLMPerf workloads and operate this project end to end. Use when Codex needs to create or review Runner/Campaign YAML, select or configure Backend Provider Profiles, verify models and credentials, run smoke/load/KV-cache/retention/residency tests, operate llmperfctl, diagnose Runner or Campaign failures, interpret or export results, define CLI/API input-output contracts, or modify Scheduler/Planner/Worker/PostgreSQL implementation."
---

# Operate LLMPerf

Translate the user's measurement goal into the smallest bounded workload that can
answer it, prove the Provider with a smoke request, then scale deliberately.

## Load the relevant reference

- Read [references/yaml.md](references/yaml.md) completely before creating,
  reviewing, or changing Runner/Campaign/RunnerPlan YAML.
- Read [references/provider.md](references/provider.md) completely before checking,
  creating, changing, or diagnosing a Provider Profile or model selection.
- Read [references/cli.md](references/cli.md) completely before operating the
  Backend or CLI, diagnosing a run, interpreting status, or exporting results.
- Read [references/engineering.md](references/engineering.md) completely before
  changing source, database behavior, tests, architecture, or project docs.
- Read [references/io.md](references/io.md) completely before changing CLI/API input,
  validation, projection, rendering, logging, redaction, or export behavior.
- Read more than one reference when the task crosses those boundaries.

## Preserve the execution model

Apply these invariants:

1. Treat a Campaign as the durable workload boundary. It may contain immediate
   Runners, bounded RunnerPlans, compiled task definitions, or any combination.
2. Treat a RunnerPlan as a template. The Planner materializes due occurrences as
   ordinary queued Runners; it never occupies a Scheduler slot while waiting.
3. Treat the Scheduler as the queue consumer and Worker owner. A Worker is a
   Scheduler-local Ray execution handle for exactly one claimed Runner, not an OS
   subprocess.
4. Keep that one-Runner Worker resource boundary. The Scheduler is the only Ray driver.
   Every Worker submits one retry-free Ray task and obtains at least one Runner-owned LLM
   client actor. Actors request resources independently and release them with their request
   slot. Keep every client actor serial (`max_concurrency=1`) and exclusive to one Runner;
   never initialize Ray or connect PostgreSQL inside a Worker task.
5. Persist control state and benchmark results in PostgreSQL. Do not add SQLite
   fallback behavior or in-memory authoritative state.
6. Keep Provider endpoints and credentials in Backend-owned profiles. Put only
   the stable provider ID and model ID in workload YAML.
7. Treat an atomic single-request Runner as the only compiled-task execution primitive.
   The Workload Compiler owns matrix/repeat/parallel expansion; the Planner only handles
   dependencies and due times and must not interpret role names.
8. Distinguish Campaign lifecycle `status` from aggregate execution `outcome`.
   Never infer that `completed` means every Runner succeeded.

## Run the performance safety gate

Before submission, read `scheduler status` and validate with the active slot/Ray shape:

```bash
.venv/bin/python .codex/skills/operate-llmperf/scripts/validate_workload.py FILE \
  --scheduler-slots SLOTS --ray-num-cpus CPUS --ray-actor-num-cpus ACTOR_CPUS
```

Treat a non-zero result as a stop condition. Report planned Runner count, estimated
Provider requests, mean-token budget, maximum Runner concurrency, effective concurrency,
and warnings. In addition:

- verify the shared Ray actor capacity can satisfy the effective concurrency; every
  Runner must use Ray, but no Runner may create its own runtime;
- verify concurrent Campaigns receive fair claims while Ray independently queues their
  actors; do not require all actors of one Runner to become ready simultaneously;
- account for all RunnerPlan occurrences and compiled invoke nodes, not only YAML item count;
- inspect host CPU/memory, PostgreSQL latency, Ray worker churn, and queue growth before
  increasing slots; a 1x1 Runner can still be expensive when its runtime is heavyweight;
- treat `scheduler status.performance_guard.tripped=true` as a runtime circuit breaker:
  new claims stop, queued work stays durable, and running Runners are not killed;
- treat `scheduler status.ray_runtime.claim_blocked=true` as Ray Object Store pressure;
  wait for ObjectRefs to be persisted/released before increasing slots;
- when a Campaign causes resource pressure, cancel the Campaign before restarting the
  Backend so queued Runners and pending Dispatches cannot resume unexpectedly.

Do not add blanket Provider retries to performance or cache probes. Retry only a bounded,
explicit set of transient failures with exponential backoff and jitter. Never retry 4xx
authentication/model/parameter errors. For compiled cache tasks, do not retry a node in the
same payload family after an ambiguous send: mark the instance failed and, if repetition is
required, create a new independent instance/seed so the retry cannot warm or refresh the
cache being measured. For load/reliability measurements, keep retries disabled by default
because retries hide the error rate and increase offered load.

## Translate requirements into a workload

Establish these facts from the request or local context:

- measurement: availability, latency/throughput, concurrency comparison, scheduled
  repetition, exact-repeat cache, passive retention, or access-conditioned residency;
- target: Provider Profile ID and exact model ID;
- scale: input/output token distributions, concurrency, request/trial count, delays,
  recurrence bounds, timeout, tokenizer/dataset, and export needs;
- action: create/review YAML only, or configure and run it.

Make low-risk assumptions when they preserve the user's intent. Ask only when a missing
choice changes the experiment meaning, creates substantial Provider spend, or requires a
credential/configuration mutation the user has not authorized.

Select the durable shape:

- Use one Runner for a bounded smoke or single load point.
- Use Campaign `runners` to compare several immediate configurations.
- Use a bounded RunnerPlan for repeated wall-clock or interval measurements.
- Use `cache_probe` for within-Runner exact-repeat or prefix/mutation comparisons.
- Use `task_definitions` for cross-Runner causal experiments. Compose only the atomic
  `invoke` node with bounded `matrix`, `sequence`, `repeat`, and `parallel` syntax.
- Express passive retention as independent delay-matrix instances; express access-
  conditioned residency as a repeat chain; express repeated-hit behavior as a
  `warmup_count × quiet_seconds` matrix. Role strings are analysis tags only.
- Use the same payload ID for exact Prime/Warm/Probe replay and a different payload
  namespace for controls. Random sampling must be deterministic and hash-verified.

## Follow the end-to-end workflow

1. Inspect `git status --short` and preserve unrelated user changes.
2. Inspect current examples, active Pydantic models, and CLI help when exact fields or
   options matter; checked-in documentation may lag implementation.
3. Check `health`, Scheduler, Planner, Provider list, and exact model visibility. Configure
   a Provider only when necessary and authorized; never place its URL/key in workload YAML.
4. Create a short, bounded smoke YAML first. Estimate its Provider request count and run it
   before any longer or more expensive workload when execution is in scope.
5. Create the requested YAML from the proven smoke configuration. Keep stochastic fields
   fixed when the experiment requires reproducibility and bound every plan/task expansion.
6. Validate YAML locally with the performance safety command above using the active
   Scheduler slot count and shared Ray address (if configured).
   Preview RunnerPlan occurrences. Remember submission may resolve Provider, tokenizer, and
   dataset artifacts before it accepts the complete workload.
7. Run only when requested or clearly included in the task. Use `-w` for observation, retain
   the durable ID, and inspect the final `status` and `outcome`; CLI exit alone is not proof.
8. On failure, inspect `runner status --summary` and dedicated `runner logs` from the first
   failed Runner before changing the workload.
9. For implementation work, make the smallest coherent cross-layer change:
   model, persistence, API, CLI, docs, and focused tests as applicable.
10. Run focused tests, then the complete test suite. Do not silently substitute a
   different database or provider.
11. Report assumptions, YAML path, Provider/model, estimated and actual request counts,
    durable IDs, lifecycle status, outcome, failures, exports, and whether restart,
    PostgreSQL configuration, artifact download, credentials, or further paid runs remain.

## Keep examples direct

- Name every file under `examples/` as `example-<main-purpose>.yaml`.
- Keep the name concise and use no more than three `-` characters in the filename.
- Make the checked-in defaults finish promptly and return observable results when
  run with a configured Provider. Periodic and TTL examples must use short bounded
  timings; geographic-time capabilities use a short relative schedule by default.
  Document longer production timings in comments instead of making users wait.

## Control cost and authority

- If the user asks only for YAML, do not configure a Provider or submit it.
- If the user asks to run a test, treat the required bounded Provider calls as in scope;
  call out the request budget before a materially larger multi-round or multi-delay run.
- Never invent, display, log, or commit credentials. Have the user supply secrets through
  `llmperf-backend config set ... --stdin` or configure them manually.
- Provider changes require a Backend restart. Do not restart an externally managed service
  unless the user authorized it; state the exact remaining action instead.
- Do not infer inference support from `/models`; require a successful 1x1 smoke Runner.

## Diagnose from the first concrete failure

- Inspect `runner status --summary`, then `runner logs` for Worker/Ray exceptions.
- Treat HTTP status nested through the Backend as the provider response, not
  necessarily as a Backend authentication failure.
- Treat `/models` discovery as catalog visibility only; prove inference support
  with a bounded smoke Runner.
- For long `-w` commands, remember that logs are rendered by HTTP polling and
  change detection; exiting the CLI stops observation, not durable execution.
- Avoid masking a failed request as a zero-request success. Preserve the first
  request error and Worker task details in the Runner result.

## Keep outputs safe

- Never print or persist API keys, bearer tokens, or private keys.
- Use `llmperf-backend config set ... --stdin` for secrets.
- Keep JSON/table output on stdout and operational logs on stderr.
- Use `--log-level debug` only when HTTP timing and response metadata are needed;
  credentials must remain redacted.

## Preserve the input-output boundary

Apply the decode -> validate -> resolve -> persist input pipeline and the authoritative
record -> command compatibility adapter -> whitelist projector -> render/export output
pipeline in [references/io.md](references/io.md). Register every CLI request explicitly;
never use an identity adapter, a generic raw fallback, or let a renderer accept dict/list
responses. Keep default views stable and filtered, make `--full` a larger whitelist rather
than a raw escape hatch, and update adapters, help, redaction, and boundary tests together.

# llmperf-orchestrator

Durable orchestration and observability for reproducible LLM API performance
and KV-cache experiments.

LLMPerf turns one-off benchmark commands into recoverable Campaigns: workloads
are validated once, scheduled as durable Runners, executed by isolated Workers,
and persisted before results are exported as JSON or professional HTML reports.

## Architecture

```text
                              +-----------------------+
                              |      llmperfctl       |
                              | submit/query/export   |
                              +-----------+-----------+
                                          |
                                       HTTP/JSON
                                          v
+----------------------------------------------------------------------------------+
|                               llmperf-backend                                    |
|                                                                                  |
| +-----------------------+  +---------------------+  +--------------------------+ |
| | FastAPI control plane |  | Planner             |  | Scheduler (N slots)      | |
| | auth/validate/compile |  | due work -> Runner  |  | claim via DB + supervise | |
| +-----------+-----------+  +----------+----------+  +-------------+------------+ |
+-------------|-------------------------|---------------------------|--------------+
              | atomic transactions     | materialize               | Ray handle
              |                         |                           v
              |                         |               +-----------+----------+
              |                         |               | Worker (one/Runner)  |
              |                         |               | Ray task + Actors    |
              |                         |               +-----+------------+---+
              |                         |                     |            |
              |                         |            results  |            | API
              |                         |                     |            v
              |                         |                     |    +---------------+
              |                         |                     |    | Provider API  |
              |                         |                     |    +---------------+
              v                         v                     v
+----------------------------------------------------------------------------------+
|                         PostgreSQL -- source of truth                            |
| Campaigns | RunnerPlans/cursors | Runner queue/events | summaries/request metrics|
+----------------------------------------------------------------------------------+
```

## Technical highlights

- **Recoverable orchestration** — Campaigns, plan cursors, Runner state, events,
  and results survive CLI disconnects and Backend restarts.
- **Capacity-efficient scheduling** — RunnerPlans wait without occupying a
  Scheduler slot, Worker, Ray runtime, or Provider connection. PostgreSQL row
  locks and occurrence uniqueness support safe multi-Backend competition.
- **Concurrent Campaign fairness** — claims prefer Campaigns with fewer running
  Runners while Ray independently queues request Actors, so one Campaign does not
  monopolize Scheduler slots or require all of its Actors to start together.
- **Fail-closed performance guard** — every Runner uses isolated Ray actors on one
  shared embedded or external runtime; workload admission bounds actor capacity,
  Runner fan-out, Provider requests, token budget, and effective concurrency.
- **Reproducible experiments** — Provider/model selections, benchmark parameters,
  and immutable tokenizer and dataset revisions are frozen into every Runner.
- **Composable workload input** — a bounded Workload Compiler expands matrix,
  sequence, repeat, and parallel YAML into atomic invoke DAGs. It is an input-layer
  component; Planner remains responsible for durable due-work materialization.
- **Secret isolation** — endpoints and credentials live in Backend-owned Provider
  Profiles and never need to appear in workload YAML or exported reports.
- **KV-cache evidence, not guesses** — deterministic prime/warm pairs, normalized
  Provider cache counters, timing phases, bootstrap confidence intervals, and
  explicit evidence verdicts distinguish cache accounting from proven speedup.
- **Durable retention curves** — compiled task instances preserve dependencies,
  deterministic payload identity, and timing checkpoints in PostgreSQL; the Planner
  emits long-delay warm/control Runners without holding a Worker while TTL time elapses.
- **Audit-ready reporting** — aggregate and per-request metrics are persisted in
  PostgreSQL; lightweight status views, JSON exports, and self-contained HTML
  reports are derived from the same durable records.

## Quick start

### 1. Install

Requirements: Python 3.10 or newer and PostgreSQL.

```bash
git clone git@github.com:BeckWlim/llmperf-orchestrator.git
cd llmperf-orchestrator

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The default installation includes the OpenAI-compatible and Vertex AI REST
adapters. Install only the optional integrations you use:

```bash
# LiteLLM-backed providers, including Anthropic
python -m pip install -e '.[litellm]'

# Amazon SageMaker
python -m pip install -e '.[sagemaker]'

# Standalone correctness benchmark (`python -m llmperf.llm_correctness ...`)
python -m pip install -e '.[correctness]'
```

Initialize a local database:

```bash
createdb llmperf
psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
```

Version 1.0.0 is the first supported schema. Initialize it from
`sql/postgresql/init.sql`; pre-release database layouts are not accepted or migrated.

### 2. Configure a Provider and start the Backend

The included examples use the Backend Provider Profile `aliyun` and model
`deepseek-v4-pro`. Set `ALIYUN_API_KEY` in your shell, then persist the Backend
configuration without placing the key in shell arguments:

```bash
llmperf-backend config set DATABASE_URL postgresql+asyncpg:///llmperf
llmperf-backend config set LLMPERF_PROVIDER_ALIYUN_URL \
  https://dashscope.aliyuncs.com/compatible-mode/v1
printf '%s' "$ALIYUN_API_KEY" | \
  llmperf-backend config set LLMPERF_PROVIDER_ALIYUN_KEY --stdin
llmperf-backend config set LLMPERF_DEFAULT_PROVIDER aliyun
llmperf-backend config list
```

Start the service:

```bash
llmperf-backend
```

For persistent Ubuntu operation after the virtual environment and PostgreSQL schema
are ready, render the template from [`deploy/systemd/`](deploy/systemd/README.md)
into `/etc/systemd/system`. The
[`llmperf-backend.service.template`](deploy/systemd/llmperf-backend.service.template)
starts the existing `.venv` directly as a configured non-root user and never stores
Provider credentials.

`llmperfctl` loads its own optional user environment from
`~/.config/llmperf/cli.env` (or `$XDG_CONFIG_HOME/llmperf/cli.env`). Configure a
remote Backend without exporting values in every shell:

```bash
llmperfctl config set LLMPERF_URL http://127.0.0.1:12666
llmperfctl config set LLMPERF_PRIVATE_KEY /home/user/.ssh/id_rsa
llmperfctl config path
llmperfctl config list
```

Exported process variables take precedence. Set `LLMPERF_CLI_ENV_FILE` to use a
different file. `config set` creates the selected file with mode `0600`;
Backend-facing commands reject an explicitly selected missing file, while the
local `config` commands remain available to inspect or repair it.

In another terminal, verify the control plane and Provider catalog:

```bash
source .venv/bin/activate
llmperfctl health
llmperfctl provider list
llmperfctl provider models aliyun
llmperfctl provider models aliyun --json
```

Provider queries default to terminal-friendly tables and summaries; pass `--json`
explicitly for the stable JSON projection. `provider list` shows at most three
representative models for static Profiles in configuration order and does not trigger
remote discovery for dynamic catalogs. After changing persisted `LLMPERF_PROVIDER_*`
settings, run `llmperfctl provider reload` to atomically refresh Provider Profiles without
restarting the Backend. Reload does not change PostgreSQL, Scheduler, Planner, Ray, listen
addresses, or general Backend configuration. Running Runners retain their connection
snapshot; newly claimed Runners use the new generation. A rejected candidate leaves the
current generation unchanged.

The example YAML uses `deepseek-v4-pro`; if the catalog returns a different exact
model ID, update the example before continuing.

Provider catalog visibility does not guarantee inference access. Run the smoke
test before submitting a longer experiment:

```bash
llmperfctl runner start -f examples/example-smoke.yaml --wait
```

### 3. Run a multi-round Campaign

Preview and submit the bounded two-round RunnerPlan example:

```bash
llmperfctl planner preview -f examples/example-runner-plan.yaml
.venv/bin/python .codex/skills/operate-llmperf/scripts/validate_workload.py \
  examples/example-runner-plan.yaml --scheduler-slots 1
llmperfctl campaign start -f examples/example-runner-plan.yaml
```

Copy the returned `campaign_id`, then inspect the Campaign and its Runners:

```bash
llmperfctl campaign status CAMPAIGN_ID
llmperfctl runner list --limit 20
```

The CLI may exit without affecting execution; Campaign, RunnerPlan, and Runner
state remain durable in PostgreSQL. Mutation and wait commands write operational
state changes to stderr and do not dump response JSON to stdout by default.
Use `status`, `list`, `logs`, or an explicit export command when output is needed.

### 4. Export results and prepare an HTML report

```bash
llmperfctl campaign export CAMPAIGN_ID -o campaign-report.json

python .codex/skills/generate-llmperf-report/scripts/prepare_report_data.py \
  --input campaign-report.json \
  --output /tmp/campaign-analysis.json
```

From Codex, the same workflow can be requested with:

```text
Use $generate-llmperf-report to generate a professional HTML report for
Campaign CAMPAIGN_ID.
```

The deterministic pipeline produces chart-neutral evidence. The reporting Agent
chooses the HTML structure and charts after inspecting that evidence, while shared
theme and palette assets keep style stable. Add `campaign export --include-requests`
only when request-level distributions or outliers are needed.

## Included examples

| File | Purpose |
|---|---|
| [`examples/example-smoke.yaml`](examples/example-smoke.yaml) | Minimal 1x1 Provider and persistence check |
| [`examples/example-campaign.yaml`](examples/example-campaign.yaml) | Immediate Runners plus a compiled deterministic replay task |
| [`examples/example-runner-plan.yaml`](examples/example-runner-plan.yaml) | Two bounded Runners materialized one second apart |
| [`examples/example-cache-retention.yaml`](examples/example-cache-retention.yaml) | Delay matrix with deterministic Prime/Warm replay and cold controls |
| [`examples/example-cache-residency.yaml`](examples/example-cache-residency.yaml) | Fixed-interval repeated payload observations |
| [`examples/example-cache-promotion.yaml`](examples/example-cache-promotion.yaml) | Repeated-hit count × quiet-window task matrix |
| [`examples/example-sharegpt-long.yaml`](examples/example-sharegpt-long.yaml) | Immutable ShareGPT-backed 64K replay and cold control |

Workload YAML selects stable Provider and model IDs only. Provider URLs, keys,
artifact caches, and service settings belong to the Backend configuration.

`task_definitions` is compile-time composition, not a second scheduler. The compiler
expands a finite `matrix`/`sequence`/`repeat`/`parallel` recipe into single-request
`invoke` nodes. Reusing one logical payload produces deterministic random input from
the task seed, matrix coordinates, trial index, and payload namespace; runtime
`prompt_hash` validation proves Prime/Warm replay identity. Planner sees only generic
dependencies and due times and never branches on role or experiment names.

Large-context workloads can use Backend-resolved Hugging Face artifacts instead of
expanding the small bundled sonnet corpus. Artifact location and record schema are separate:
`source: huggingface` resolves the immutable file, while the required `adapter` selects a
prompt-record decoder. `sharegpt` extracts `conversations[0].value`; `text` treats each
non-empty line as one prompt. The bundled sonnet is another explicit adapter, not an
implicit ShareGPT fallback. All adapters feed the same indexed-record loading, seeded
construction, and evidence pipeline. Set `dataset_prompt_mode: sample` to benchmark intact
records, or `concatenate` to assemble diverse records to the requested token budget without
reusing a record until the corpus is exhausted. A new seeded cycle begins only after
exhaustion. Dataset-backed compiled tasks require an immutable resolved dataset revision and
persist text-free selection evidence in addition to the prompt hash.

## Operational essentials

```bash
# Control plane
llmperfctl health
llmperfctl scheduler status
llmperfctl planner runtime

# Runners
llmperfctl runner status RUNNER_ID
llmperfctl runner logs RUNNER_ID
llmperfctl runner list --status failed

# Campaigns and plans
llmperfctl campaign list
llmperfctl campaign status CAMPAIGN_ID
llmperfctl planner events RUNNER_PLAN_ID
llmperfctl campaign export CAMPAIGN_ID -o campaign-report.json
```

Campaign lifecycle `status` and aggregate execution `outcome` are independent.
For example, `status=completed, outcome=partial_failed` means all bounded work
finished but at least one Runner failed. Unsuccessful outcomes return CLI exit
code `2`, while durable results remain available for diagnosis and export.

## Documentation

- [Architecture and persistence](docs/ARCHITECTURE.md)
- [RunnerPlan and geographic-time scheduling](docs/RUNNER_PLANNER_ARCHITECTURE.zh-CN.md)
- [External KV-cache observability](docs/KVCACHE_OBSERVABILITY_TECHNICAL_REPORT.zh-CN.md)
- [HTML report Project Skill](.codex/skills/generate-llmperf-report/SKILL.md)
- [Backend environment template](.env.template)

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
```

PostgreSQL integration tests are opt-in and require an explicitly disposable
database whose name contains `test`:

```bash
createdb llmperf_test
export LLMPERF_TEST_DB=postgresql+asyncpg:///llmperf_test
pytest -q -m postgresql
```

## Acknowledgements

This project builds on [ray-project/llmperf](https://github.com/ray-project/llmperf).
We thank its original authors and contributors for creating and open-sourcing the
LLM API benchmarking foundation used here.

## License

[Apache License 2.0](LICENSE.txt)

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
| | auth/validate/status  |  | due plan -> Runner  |  | claim via PG + supervise | |
| +-----------+-----------+  +----------+----------+  +-------------+------------+ |
+-------------|-------------------------|---------------------------|--------------+
              | atomic transactions     | materialize               | one process
              |                         |                           v
              |                         |               +-----------+----------+
              |                         |               | Worker (one/Runner)  |
              |                         |               | LLMPerf + Ray Actors |
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
- **Reproducible experiments** — Provider/model selections, benchmark parameters,
  and immutable tokenizer and dataset revisions are frozen into every Runner.
- **Secret isolation** — endpoints and credentials live in Backend-owned Provider
  Profiles and never need to appear in workload YAML or exported reports.
- **KV-cache evidence, not guesses** — deterministic prime/warm pairs, normalized
  Provider cache counters, timing phases, bootstrap confidence intervals, and
  explicit evidence verdicts distinguish cache accounting from proven speedup.
- **Audit-ready reporting** — aggregate and per-request metrics are persisted in
  PostgreSQL; lightweight status views, JSON exports, and self-contained HTML
  reports are derived from the same durable records.

## Quick start

### 1. Install

Requirements: Python 3.9 or newer and PostgreSQL.

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

# Standalone llm_correctness.py benchmark
python -m pip install -e '.[correctness]'
```

Initialize a local database:

```bash
createdb llmperf
psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
```

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

In another terminal, verify the control plane and Provider catalog:

```bash
source .venv/bin/activate
llmperfctl health
llmperfctl provider list
llmperfctl provider models aliyun
```

The example YAML uses `deepseek-v4-pro`; if the catalog returns a different exact
model ID, update the example before continuing.

Provider catalog visibility does not guarantee inference access. Run the smoke
test before submitting a longer experiment:

```bash
llmperfctl runner start -f examples/test-smoke.yaml --wait
```

### 3. Run a multi-round Campaign

Preview and submit the bounded eight-round RunnerPlan example:

```bash
llmperfctl planner preview -f examples/runner-plan.yaml
llmperfctl campaign start -f examples/runner-plan.yaml
```

Copy the returned `campaign_id`, then inspect the Campaign and its Runners:

```bash
llmperfctl campaign status CAMPAIGN_ID
llmperfctl runner list --limit 20
```

The CLI may exit without affecting execution; Campaign, RunnerPlan, and Runner
state remain durable in PostgreSQL.

### 4. Export results and generate an HTML report

```bash
llmperfctl campaign export CAMPAIGN_ID -o campaign-report.json

python .codex/skills/generate-llmperf-report/scripts/generate_report.py \
  --input campaign-report.json \
  --output reports/campaign-report.html
```

From Codex, the same workflow can be requested with:

```text
Use $generate-llmperf-report to generate a professional HTML report for
Campaign CAMPAIGN_ID.
```

The report is a self-contained HTML file with inline SVG charts, Runner-level
metrics, data-quality warnings, and failure diagnostics. Add
`campaign export --include-requests` only when request-level records are needed.

## Included examples

| File | Purpose |
|---|---|
| [`examples/test-smoke.yaml`](examples/test-smoke.yaml) | Minimal 1x1 end-to-end Provider and persistence check |
| [`examples/test-campaign.yaml`](examples/test-campaign.yaml) | Immediate cold-versus-repeated ShareGPT KV-cache Campaign |
| [`examples/runner-plan.yaml`](examples/runner-plan.yaml) | Eight bounded KV-cache rounds emitted every 30 seconds |

Workload YAML selects stable Provider and model IDs only. Provider URLs, keys,
artifact caches, and service settings belong to the Backend configuration.

## Operational essentials

```bash
# Control plane
llmperfctl health
llmperfctl scheduler status
llmperfctl planner runtime

# Runners
llmperfctl runner status RUNNER_ID --summary
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

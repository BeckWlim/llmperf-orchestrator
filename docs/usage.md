# Usage and configuration guide

Detailed setup, workload, and operations guidance. Start with the [README](../README.md)
for a quick overview, or see [Architecture](architecture.md) for the execution model.
All shell commands below run from the repository root.

## Technical highlights

- **Recoverable orchestration** — Campaigns, plan cursors, Runner state, events,
  and results survive CLI disconnects and Backend restarts.
- **Capacity-efficient scheduling** — RunnerPlans wait without occupying a
  Scheduler slot, Worker, Ray runtime, or Provider connection. PostgreSQL row
  locks and occurrence uniqueness support safe multi-Backend competition.
- **Concurrent Campaign fairness** — claims prefer Campaigns with fewer running
  Runners while Ray independently queues request Actors, so one Campaign does not
  monopolize Scheduler slots or require all of its Actors to start together.
- **Runtime protection with optional campaign budgets** — static validation and
  concurrency checks stay on. Whole-submission Runner, request, and token totals
  are advisory by default; enable campaign admission explicitly to enforce them.
  See [settings, defaults, and rejection meanings](runtime.md#performance-guard).
  Runners share Ray request actors with per-request credential isolation.
- **Reproducible experiments** — Provider/model selections, benchmark parameters,
  and immutable tokenizer and dataset revisions are frozen into every Runner.
- **Composable workload input** — a bounded Workload Compiler maps readable `workflow`
  primitives to an immutable node hierarchy, expands a UUID-free logical DAG table, and
  assembles atomic invokes only at the persistence boundary. Planner remains responsible
  for durable due-work materialization.
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

## Installation and setup

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

# Standalone correctness benchmark (`python -m llmperf.benchmark.correctness ...`)
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

The quick start uses the Backend Provider Profile `aliyun` and model
`deepseek-v4-pro`. Set `ALIYUN_API_KEY` in your shell, then persist the Backend
configuration without placing the key in shell arguments:

```bash
llmperf-backend config set DATABASE_URL postgresql+asyncpg:///llmperf
cp src/llmperf_backend/configuration/configs/providers.example.yaml ~/.config/llmperf/providers.yaml
printf '%s' "$ALIYUN_API_KEY" | \
  llmperf-backend config set ALIYUN_API_KEY --stdin
llmperf-backend config set LLMPERF_DEFAULT_PROVIDER aliyun
llmperf-backend config list
```

Edit the single `~/.config/llmperf/providers.yaml` file to define all Provider Profiles.
`providers.json` is also supported with the same structure. Fields include `adapter`,
`base_url`, `api_key_env` (a credential variable name), `models` (a list), and nested
`discovery` settings. Optional `worker_api_base_env` and `worker_api_key_env` select the
adapter's destination environment variables. Set `api_key` directly in this Backend-owned
file, or use `api_key_env` to reference `backend.env` or the process environment. These two
fields are mutually exclusive. Empty keys and unset referenced variables are allowed;
OpenAI-compatible requests omit Authorization when no key is configured. Process values
take precedence for environment references. Protect Provider files containing keys like
other credential files; keys are excluded from public Provider responses.

Set `LLMPERF_PROVIDERS_FILE` to select another YAML/JSON file; relative paths resolve beside
`backend.env`. Otherwise the Backend finds one `providers.yaml`, `providers.yml`, or
`providers.json` beside that environment file. Multiple matches require explicit selection.
Provider definitions load only from this file. `LLMPERF_PROVIDER_<ID>_*` fields in
`backend.env` or the process environment no longer create profiles or override file fields.
Without a Provider file, Provider loading fails. Invalid files,
unknown fields, and duplicate IDs/keys reject the candidate. At startup, Provider loading
errors leave the control plane available with no active profiles and a configuration error
visible in `llmperfctl provider list`. Fix the file and run `llmperfctl provider reload`.
A failed reload keeps the previously active profiles. Core database and authentication
configuration still require valid settings.

#### Migrating environment-defined Providers

Move each profile into `providers.yaml` before updating the Backend. Preserve its exact ID,
endpoint, adapter, model catalog, and discovery policy. All profiles must be in the file;
file and environment definitions are never merged.

| Removed environment field | Provider YAML field |
|---|---|
| `LLMPERF_PROVIDER_<ID>_URL` | `base_url` |
| `..._ADAPTER` | `adapter` |
| `..._MODELS` (comma-separated) | `models` (YAML list) |
| `..._DISCOVERY` | `discovery.mode` |
| `..._PATH` | `discovery.path` |
| `..._TTL` | `discovery.cache_ttl_seconds` (integer) |
| `..._URLVAR` | `worker_api_base_env` |
| `..._KEYVAR` | `worker_api_key_env` |
| `..._KEY` | `api_key_env: EXISTING_KEY_VARIABLE` or private `api_key` |

An existing `..._KEY` variable may remain in `backend.env` when the YAML explicitly names
it in `api_key_env`; its name alone no longer defines a Provider. Remove other obsolete
profile variables after migration. `LLMPERF_DEFAULT_PROVIDER`, `LLMPERF_DEFAULT_MODEL`, and
backend settings remain environment configuration.

For a local chat endpoint without model discovery:

```yaml
version: "1.0.0"
providers:
  local:
    adapter: openai
    base_url: http://127.0.0.1:8001/v1
    models: [exact-model-id]
    discovery:
      mode: static
```

After installing the updated code, run `llmperfctl provider reload`, then inspect
`llmperfctl provider list` and `llmperfctl provider models PROVIDER_ID`. File edits and
credential rotation can be reloaded without restarting; installing new code requires the
normal service update.

Start the service:

```bash
llmperf-backend
```

For persistent Ubuntu operation after the virtual environment and PostgreSQL schema
are ready, render the template from [`deploy/systemd/`](../deploy/systemd/README.md)
into `/etc/systemd/system`. The
[`llmperf-backend.service.template`](../deploy/systemd/llmperf-backend.service.template)
starts the existing `.venv` directly as a configured non-root user and never stores
Provider credentials.

`llmperfctl` loads its own optional user environment from
`~/.config/llmperf/cli.env` (or `$XDG_CONFIG_HOME/llmperf/cli.env`). Configure a
remote Backend without exporting values in every shell:

```bash
llmperfctl config set LLMPERF_URL http://127.0.0.1:8000
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
remote discovery for dynamic catalogs. After changing the Provider file or its referenced credentials, run `llmperfctl provider reload` to atomically refresh Provider Profiles without
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

Use `runner_groups` for readable matrices across globally ordered phases:

```yaml
campaign:
  name: phased-study
  rounds:
    count: 8
    seed: date
    prefix_seed_scope: family
    phases: [load, pressure, read]
runner_groups:
  - name: target
    phase: load
    matrix: &families
      group: [1, 2]
      member: [1, 2, 3, 4]
    runner: &target
      provider: your-provider
      model: your-model
      concurrent_requests: 8
      inputs:
        default: {tokens: 4096, shared_prefix_tokens: 3276}
      request: {repeat: 1}
  - name: pressure
    phase: pressure
    matrix:
      member: [1, 2, 3, 4, 5, 6, 7, 8]
    runner:
      <<: *target
      inputs:
        default: {tokens: 4096, shared_prefix_tokens: 0}
  - name: target
    phase: read
    matrix: *families
    runner: *target
```

Each phase completes before the next becomes eligible. Reusing group name and
coordinates preserves the prefix across phases/cycles; suffixes change. Phase names
carry no built-in cache behavior. `seed: date` is frozen per API request; use an integer
for reproducible preview/submission. `campaign preview` shows the resolved seed,
phase order, matrix sizes and cycle count without preparing or submitting requests.
The expanded limit is 10,000 Runners. See [limitations](limitations.md#matrix-phases-and-long-running-experiments)
for recovery/failure semantics and the optional larger `workloads/*-recall.yaml` profiles.

For independent family sequences, use the bounded
[`example-batch-sequence.yaml`](../examples/example-batch-sequence.yaml). Each `run` step
dispatches a whole Runner with its configured request count and concurrency. A repeated
step waits for the preceding Runner's committed success, while other family instances
advance independently. This differs from `campaign.rounds`, which retains a barrier
across every family. Batch-only task definitions do not need `payloads`.

Preview and submit the bounded two-round RunnerPlan example:

```bash
llmperfctl planner preview -f examples/example-runner-plan.yaml
.venv/bin/python .codex/skills/operate-llmperf/scripts/validate_workload.py \
  examples/example-runner-plan.yaml --scheduler-slots 1
llmperfctl campaign preview -f examples/example-cache.yaml --debug
llmperfctl campaign validate -f examples/example-runner-plan.yaml \
  --artifact-timeout 3600
llmperfctl campaign start -f examples/example-runner-plan.yaml
```

`campaign validate` resolves every Backend-owned tokenizer and dataset, then fully reads
and SHA-256 hashes the materialized cache files without creating benchmark work. Dataset
validation also runs the selected adapter, materializes its normalized Hugging Face Arrow
index, and reports the usable record count. Use it before `campaign start` for large
uncached ShareGPT or tokenizer artifacts; the dedicated artifact timeout does not change
the normal Backend request timeout. During uncached downloads, the Backend streams
path-free absolute byte counts and heartbeats. On a terminal, `llmperfctl` dynamically
refreshes one downloaded/total byte counter without drawing a progress bar; for redirected
stderr it writes one structured byte record per event. Projected text/JSON on stdout is
never contaminated.

`campaign preview` performs Backend-authoritative schema validation and TaskCompiler
expansion without downloading artifacts or writing database records. It renders a bounded
ASCII dependency view by default; use `--limit`, `--node-limit`, `--debug`, or `--json` for
more instances, more nodes per instance, deterministic payload seeds, or structured output.

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

## Workload examples and task definitions

Use the [example catalog](examples.md) to choose a small workload and inspect its request
budget. [Task compiler and YAML rules](tasks.md) is the reference for instance expansion,
workflow composition, batch versus atomic execution, seeds, limits, and failure boundaries.
Provider URLs, keys, artifact caches, and service settings remain Backend-owned.

Large-context workloads can use Backend-resolved Hugging Face artifacts instead of
expanding the small bundled sonnet corpus. Artifact location and record schema are separate:
`source: huggingface` resolves the immutable file, while the required `adapter` selects a
prompt-record decoder. `sharegpt` preserves the legacy behavior of accepting any non-empty
`conversations[0].value`; `sharegpt-user` accepts only first turns whose normalized `from`
role is `human` or `user`; `document-text` treats every non-empty Parquet or Arrow `text`
row as one complete document; `text` treats each non-empty text-file line as one prompt.
External files are normalized into a disk-backed Hugging Face Arrow index. ShareGPT JSON
arrays and JSONL files are incrementally parsed because the upstream builder fully reads
top-level arrays; Parquet, Arrow, and text continue through the standard builders. Prompt
selection shuffles integer row positions and reads text lazily instead of constructing an
in-memory copy of the corpus. Set `HF_DATASETS_CACHE` to choose the persistent Arrow cache
directory.

The bundled sonnet is another explicit adapter, not an implicit ShareGPT fallback. All
adapters feed the same seeded construction and evidence pipeline. Set
`dataset_prompt_mode: sample` to benchmark intact records, or `concatenate` to assemble
diverse records to the requested token budget without reusing a record until the corpus is
exhausted. A new seeded cycle begins only after exhaustion. Dataset-backed compiled tasks
require an immutable resolved dataset revision and persist text-free selection evidence in
addition to the prompt hash.

## Request concurrency and token counts

`runner.concurrent_requests` bounds requests within an ordinary Runner. The shared Ray
pool defaults to eight actors with eight requests each, providing 64 slots across Runners.
The active Runner bound is independent; its default is 64 for atomic workloads. Waiting
turns and delays use no request slots. See [runtime configuration](runtime.md#capacity).

The Backend checks required tables and columns without an application schema-version
marker. Older protocol/Dispatch layouts require explicit migration or a fresh database.
Stop the old Backend before switching database configuration and preserve a backup.

Tokenization runs during prompt preparation, before timed request execution. By default,
`benchmark.output_token_count: provider` uses server-reported output usage and performs no
local output tokenization. Missing usage leaves output count, total count, token throughput,
and TPOT unavailable (`null`); request rates and measured latencies remain available.
Input counts prefer server usage, with the prepared local prompt count as a labeled fallback.
Stream chunks and requested `max_tokens` are never substituted for output token counts.
Results include `token_count_sources` per request and `output_token_count_coverage` in the
summary; overall token throughput is unavailable if any successful request lacks a count.

Set `benchmark.output_token_count: local` to retokenize outputs after all requests finish,
outside the measured duration. These local counts depend on the selected tokenizer and may
differ from server counts, which remain separately available. This option also works in
Runner templates used by tasks and plans. The standalone benchmark exposes
`--output-token-count provider|local`.

The OpenAI-compatible adapter requests streaming usage by default. For endpoints that do
not accept `stream_options`, override it with `additional_sampling_params: {stream_options: null}`;
without returned usage, use latency/request-rate metrics or opt into local counting.

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

- [Architecture and persistence](architecture.md)
- [RunnerPlan and geographic-time scheduling](scheduling.md)
- [External KV-cache observability](cache-measurement.md)
- [HTML report Project Skill](../.codex/skills/generate-llmperf-report/SKILL.md)
- [Backend environment template](../.env.template)

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

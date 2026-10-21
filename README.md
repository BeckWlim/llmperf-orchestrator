# llmperf-orchestrator

Reproducible LLM API load tests and KV-cache experiments, with durable scheduling,
configurable concurrency, and per-request results stored in PostgreSQL.

## Main features

- **Durable workloads:** submit individual Runners, scheduled rounds, or Campaigns;
  track their state independently of the CLI session.
- **Flexible concurrency:** a shared Ray actor pool serves independent requests across
  Runners; eight actors support 64 request slots by default. Up to 1024 active Runners
  share eight preparation threads, released while their Workers execute.
- **Family prefix reuse:** stable prefixes across trials with suffixes derived from
  family seed, trial, and request. See [responsibilities and limitations](docs/limitations.md).
- **Matrix phases:** repeat named Runner matrices behind global completion barriers;
  retain family prefixes across bounded cycles with a recorded date-based or explicit seed.
  Define experiment roles in [workload YAML](docs/usage.md#3-run-a-multi-round-campaign).
- **Composable experiments:** combine sequences, parallel branches, repeats, delay
  matrices, and deterministic payload replay. Task `run` steps execute multi-request
  Runners, letting families advance through independent batch sequences.
- **Multi-turn conversations:** pass committed message history between ordered invokes.
- **Performance evidence:** measure latency, throughput, errors, and available cache
  counters; export JSON and prepare HTML reports.
- **Controlled execution:** static validation and runtime guards bound concurrent load;
  whole-campaign budgets are advisory by default, with explicit admission opt-in.
  Provider credentials stay in Backend configuration.
- **Provider configuration isolation:** define `api_key` or `api_key_env` in
  `providers.yaml`, including empty keys. Invalid Provider configuration leaves the
  control plane available; failed reloads retain the active profiles.

Tokenization runs during prompt preparation. By default, output token counts come from
server usage; missing counts remain unavailable.

## Architecture

```mermaid
flowchart LR
    CLI["llmperfctl"] --> API
    subgraph Backend["Backend"]
        API["API + task compiler"]
        Planner["Planner"]
        Runtime["Runtime coordinator + reconciliation notifier"]
        Scheduler["Scheduler"]
    end
    API --> DB[("PostgreSQL")]
    Planner -->|"Scheduled occurrences"| DB
    DB -->|"Recover and reconcile"| Runtime
    Runtime -->|"Ready requests"| Scheduler
    Scheduler -->|"Execute"| Ray["Shared Ray actors"]
    Ray -->|"Requests"| Provider["LLM Provider API"]
    Ray -->|"Results"| Scheduler
    Scheduler -->|"Persist results"| DB
    Scheduler -->|"Committed event"| Runtime
```

PostgreSQL holds metadata, checkpoints, results, and conversation history. The runtime
coordinator manages dependencies; Ray executes requests under the Scheduler's supervision.

## Quick start

Requires Python 3.10+, a running local PostgreSQL instance, and an LLM Provider API key.
The database commands assume your PostgreSQL role matches your login user and can create
databases. The example below uses Aliyun's OpenAI-compatible API.

### Install

```bash
git clone https://github.com/BeckWlim/llmperf-orchestrator.git
cd llmperf-orchestrator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
createdb llmperf
psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
```

### Configure and start

```bash
llmperf-backend config set DATABASE_URL postgresql+asyncpg:///llmperf
cp src/llmperf_backend/configuration/configs/providers.example.yaml ~/.config/llmperf/providers.yaml
```

The copied file includes the Aliyun endpoint. Enter your API key without saving it in
shell history (Bash):

```bash
read -rsp 'Aliyun API key: ' ALIYUN_API_KEY; echo
printf '%s' "$ALIYUN_API_KEY" | \
  llmperf-backend config set ALIYUN_API_KEY --stdin
unset ALIYUN_API_KEY
llmperf-backend config set LLMPERF_DEFAULT_PROVIDER aliyun
llmperf-backend
```

Leave the Backend running. In another terminal, enter the same repository directory:

```bash
source .venv/bin/activate
llmperfctl config set LLMPERF_URL http://127.0.0.1:8000
llmperfctl health
llmperfctl provider list
llmperfctl provider models aliyun
```

For another Provider or Backend address, see the [configuration guide](docs/usage.md).

### Run and inspect

Open `examples/example-smoke.yaml` and set `benchmark.model` to an exact model ID from
the catalog above that your account can use. Then send one short request:

```bash
llmperfctl runner start -f examples/example-smoke.yaml --wait
llmperfctl runner list --limit 5
llmperfctl runner status RUNNER_ID
```

Use the Runner ID printed by the start command. A successful smoke test ends with
`succeeded`; use `llmperfctl runner logs RUNNER_ID` to diagnose a failure. The first run
may download the default tokenizer.

To run scheduled rounds, update the Provider/model in `examples/example-runner-plan.yaml`,
then submit and export a Campaign:

```bash
llmperfctl planner preview -f examples/example-runner-plan.yaml
llmperfctl campaign validate -f examples/example-runner-plan.yaml
llmperfctl campaign start -f examples/example-runner-plan.yaml
llmperfctl campaign status CAMPAIGN_ID
llmperfctl campaign export CAMPAIGN_ID -o results.json
```

Use the ID returned by `campaign start`. Add `--include-requests` to export individual
measurements, or use the [report skill](.codex/skills/generate-llmperf-report/SKILL.md)
to prepare an HTML report.

## Examples and documentation

- [Documentation index](docs/README.md) · [Setup and operations](docs/usage.md)
- [Core workload examples](docs/examples.md) · [Task compiler and YAML rules](docs/tasks.md)
- [Architecture and source layout](docs/architecture.md) · [Runtime and storage](docs/runtime.md)
- [Performance guard: defaults, budgets, and runtime controls](docs/runtime.md#performance-guard)
- [systemd deployment](deploy/systemd/README.md) · [Testing](docs/testing.md)

Existing installations: the Backend checks required tables and columns. For older
database layouts, follow the [cutover guidance](docs/runtime.md#recovery-and-cutover).
Provider definitions now require a Provider file; migrate former `backend.env` profile
fields using the [migration table](docs/usage.md#migrating-environment-defined-providers).

## Development

```bash
python -m pip install -e '.[dev]'
python -m basedpyright
pytest -q
```

Runner owns Provider/model/tokenizer, generation settings, and named `inputs`.
`request` defines Worker execution logic: repetition, steps, dependencies, timing, and
message composition. Runner handles failure cancellation; Scheduler owns request slots.
See the [definition rules](docs/tasks.md#runner-data-and-request-logic) and
[message merge example](examples/example-message-merge.yaml).

## Credits and license

Built on [ray-project/llmperf](https://github.com/ray-project/llmperf).
Licensed under [Apache 2.0](LICENSE.txt).

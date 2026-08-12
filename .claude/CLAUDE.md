# LLMPerf Engineering Guide

## Project intent

This repository keeps the original LLMPerf benchmark engine and adds a durable,
asynchronous orchestration layer for GLM/KVCache studies. PostgreSQL is the
source of truth. JSON is an explicit export format, never an intermediate
result store for backend-managed runs.

Read `docs/ARCHITECTURE.md` before changing backend boundaries, task state, or
database models.

## Package boundaries

- `src/llmperf`: upstream benchmark clients, request launchers, and metrics.
- `src/llmperf_backend`: FastAPI, YAML configuration, async persistence,
  Scheduler, provider registry/model discovery, and the database-writing
  calculation worker.
- `src/llmperf_cli`: lightweight remote client. It may use the standard library
  and PyYAML, but must not import `llmperf_backend`, FastAPI, SQLAlchemy, Ray, or
  benchmark implementation modules.
- `token_benchmark_ray.py`: legacy/manual benchmark entry and reusable metric
  calculation function. Backend Runners call the calculation function through
  `llmperf_backend.worker`; they do not use its file-output path.

## Invariants

1. Every execution is a persistent Runner with one immutable `runner_id`.
2. A `campaign_id` groups zero or more Runners for aggregate export. Human-readable
   `label` values are descriptive and are not identity keys.
3. Valid state transitions are:
   `queued -> running -> succeeded|failed|cancelled` and
   `running -> queued` only during stale/shutdown recovery.
4. A CLI Campaign plan with multiple Runners must use the transactional Campaign
   batch endpoint. Do not replace it with best-effort sequential submission.
5. Claim work inside a database transaction with `FOR UPDATE SKIP LOCKED`.
   Never replace the database claim with an in-memory-only queue.
6. Summary and individual request metrics are committed in the same transaction.
   Respect `cancel_requested` under a row lock before committing results.
7. Backend-managed workers must not write result JSON files. Add new export
   formats at the API/CLI export boundary.
8. Do not run Ray or benchmark calculation on the FastAPI event loop. Keep it in
   the supervised worker subprocess.
9. Do not send the database URL into Ray actors. The Worker removes
   `LLMPERF_WORKER_DATABASE_URL` before creating Ray's runtime environment.
10. Do not return database passwords from configuration APIs. Preserve URL
   redaction when changing config responses.
11. Do not grant application roles PostgreSQL superuser privileges.
12. Provider API keys and endpoint selection are server-owned. Submitted YAML,
    Runner rows, API responses, and JSON exports may reference `provider` and
    `model`, but must never contain profile credentials.
13. The Scheduler must construct a Worker environment containing only the selected
    provider's credential variables. Never forward the complete
    `LLMPERF_PROVIDER_*` environment into Worker or Ray processes.
14. Model discovery may call only the endpoint bound to a configured provider.
    Do not accept arbitrary discovery URLs or API keys from CLI/API payloads.
15. Worker exit code zero is not benchmark success. Zero completed model
    requests must persist metrics as a failed run; partial success must be
    represented as a degraded structured outcome.
16. Keep captured stdout/stderr out of default wait output. CLI wait operations
    must expose a compact first error and return non-zero for failed/cancelled
    runs, while an explicit full view remains available.

## Compatibility

- Python: `>=3.8,<3.11`.
- Code must avoid Python 3.10-only union syntax and other newer syntax.
- Pydantic helpers in `llmperf_backend.models` intentionally support Pydantic
  1.x and 2.x.
- Runtime PostgreSQL access is asynchronous through SQLAlchemy and asyncpg.
- Tests use SQLite/aiosqlite only as an isolated repository/API test backend;
  PostgreSQL remains the production source of truth.

## Configuration

- Default YAML: `src/llmperf_backend/configs/default.yaml`.
- Override path: `LLMPERF_BACKEND_CONFIG`.
- Database URL: `DATABASE_URL` through a YAML environment placeholder.
- CLI endpoint: `LLMPERF_URL`.
- Provider profiles use the canonical
  `LLMPERF_PROVIDER_<ID>_URL|KEY|ADAPTER|MODELS|...` contract. Do not add legacy
  aliases or implicit `OPENAI_API_*` fallback profiles. Task YAML chooses
  `provider` and `model`; the backend overwrites client-supplied `llm_api` from
  the selected adapter.
- Provider profile and dotenv changes require a backend restart. The YAML
  config reload endpoint does not reload credentials or discovery settings.
- Trusted CLI private key: `LLMPERF_PRIVATE_KEY`. The CLI signs short-lived
  RS256 tokens; the service stores only `auth.public_key_path`.
- `LLMPERF_TOKEN` remains available for externally generated short-lived tokens.
- Benchmark defaults reload for newly created Runners. Server, database pool, and
  Scheduler settings require process restart.

## Database changes

The current development schema can be bootstrapped with SQLAlchemy
`create_all`. This does not alter existing tables. When changing a deployed
schema, add a real migration before relying on new columns or constraints.
Update the data-model section in `docs/ARCHITECTURE.md` and add repository tests
with each schema change.

The canonical PostgreSQL bootstrap file is `sql/postgresql/init.sql`. Keep it in
sync with ORM models. Never seed a required superuser row: bootstrap public-key
authentication must continue to work with empty `users` and
`trusted_client_keys` tables.

## Validation commands

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m py_compile src/llmperf_backend/*.py src/llmperf_cli/*.py
python -m llmperf_cli --help
```

Real PostgreSQL tests are opt-in and destructive only to a dedicated database
whose name contains `test`:

```bash
LLMPERF_TEST_DATABASE_URL='postgresql+asyncpg:///llmperf_test' \
  python -m pytest -q -m postgresql tests/test_postgresql_integration.py
```

For a PostgreSQL smoke test:

```bash
export DATABASE_URL='postgresql+asyncpg:///llmperf'
llmperf-backend
llmperfctl health
```

## Review checklist

- Is PostgreSQL still the first durable result store?
- Are state changes transactional and idempotent?
- Can two runner processes safely observe the same queued work?
- Are cancellation and successful completion serialized by a row lock?
- Are secrets absent from API responses, subprocess arguments, and Ray runtime
  environment?
- Does each Worker receive only its selected provider's endpoint/key, and are
  model discovery requests restricted to server-configured endpoints?
- Does the service still hold only the public authentication key, and does the
  CLI automatically refresh short-lived tokens?
- Does the CLI remain independently importable without backend dependencies?
- Are single-run and campaign exports still reproducible from database records?

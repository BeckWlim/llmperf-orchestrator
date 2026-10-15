# Provider Profile Configuration and Verification

## Ownership boundary

Provider Profiles are Backend-owned. Runner and Campaign YAML contains only stable
`provider` and exact `model` IDs, never endpoints or API keys. Modify configuration only
when requested or when an authorized run cannot use the target Profile. Ask users to enter
missing keys through `--stdin`; never request keys in chat.

## Read-only inspection

```bash
llmperfctl health
llmperfctl scheduler status
llmperfctl provider list
llmperfctl provider models PROVIDER_ID
```

`provider list` exposes only public Profile metadata and up to three representative static
models without remote discovery. Use `--json` for the stable projection. Use
`provider models --refresh` only when remote refresh is necessary and authorized. Catalog
visibility does not prove inference access; a 1x1 smoke does.

## Provider document (preferred)

Store all profiles in one `~/.config/llmperf/providers.yaml` or `providers.json` file:

```yaml
version: "1.0.0"
providers:
  acme:
    adapter: openai
    base_url: https://api.example.com/v1
    api_key_env: LLMPERF_PROVIDER_ACME_KEY
    models: [model-a, model-b]
    discovery:
      mode: static
      path: /models
      cache_ttl_seconds: 300
```

Set `api_key` in the Backend-owned Provider file, or keep credential values in `backend.env`
via `llmperf-backend config set ... --stdin` or the service environment and use `api_key_env`.
The two fields are mutually exclusive. Empty keys and unset referenced variables are allowed;
OpenAI-compatible requests omit Authorization in that case. Protect inline-key files and
never print their contents. Optional `worker_api_base_env` and
`worker_api_key_env` override adapter destination variable names. Unknown fields, duplicate
keys/normalized IDs reject the complete candidate. Startup loading errors leave the control
plane available with no active profiles and a safe `configuration_error` in Provider list.
Fix the configuration and reload. Failed reloads preserve the active profiles.

`LLMPERF_PROVIDERS_FILE` selects an explicit file. Relative paths resolve beside the Backend
environment file. Otherwise one `providers.yaml`, `providers.yml`, or `providers.json` in
that directory is discovered; multiple files require explicit selection. A selected file
is authoritative, and legacy environment definition fields do not override it. Without a
Provider file, legacy environment profiles remain supported. Use `llmperfctl provider reload`
after edits or credential rotation; a failed reload preserves the active generation.

## Legacy environment Profile

Environment keys use `LLMPERF_PROVIDER_<ID>_<FIELD>`. Uppercase underscores in ID normalize
to lowercase hyphens.

```bash
llmperf-backend config set LLMPERF_PROVIDER_ACME_URL https://api.example.com/v1
llmperf-backend config set LLMPERF_PROVIDER_ACME_ADAPTER openai
printf '%s' "$ACME_API_KEY" | \
  llmperf-backend config set LLMPERF_PROVIDER_ACME_KEY --stdin
llmperf-backend config set LLMPERF_PROVIDER_ACME_DISCOVERY openai
llmperf-backend config set LLMPERF_DEFAULT_PROVIDER acme
llmperf-backend config list
```

URLs must be HTTP(S) without userinfo, query, or fragment. The default discovery path is
`/models`. Checked-in workloads should select a Provider explicitly instead of depending on
the Backend default.

Supported adapters are `openai`, `anthropic`, `litellm`, `sagemaker`, and `vertexai`.
The Profile exclusively owns adapter selection; workload YAML does not expose `llm_api`.
`LLMPERF_PROVIDER_<ID>_ADAPTER` selects it explicitly and defaults to `openai` when
omitted. Adapter selection is never inferred from URL or model name. At submission, the
Backend resolves the workload's `provider` ID, injects canonical `adapter` into the frozen
Runner benchmark, and the Worker selects the corresponding client. `DISCOVERY` is an
independent catalog policy and does not select or change the request adapter. The
`anthropic` and `litellm` adapters require the `litellm` optional dependency; `sagemaker`
requires its optional dependency.

Use a static catalog when no compatible `/models` endpoint exists:

```bash
llmperf-backend config set LLMPERF_PROVIDER_ACME_DISCOVERY static
llmperf-backend config set LLMPERF_PROVIDER_ACME_MODELS model-a,model-b
```

`DISCOVERY` is `openai`, `static`, or `disabled`. Static models are an admission allow-list;
unknown exact IDs fail with 422 and are never guessed. `PATH` overrides discovery path and
`TTL` controls cache seconds from 0 to 86400. Install optional dependencies before using an
optional adapter.

## Reload and Worker injection

Credential/environment precedence is process environment over the selected environment
file (`LLMPERF_ENV_FILE`, otherwise persisted user configuration). Working-directory `.env`
files are not loaded. The Provider document, when selected, owns all profile definitions.
`llmperfctl provider reload` validates a complete candidate and atomically replaces only
Provider Profiles and model caches. It does
not reload PostgreSQL, Scheduler, Planner, Ray, auth, listen settings, or the default
Provider. Running Runners keep their connection snapshot; later claims use the new
generation.

Worker runtime environments receive only the selected Profile's endpoint/key variables.
Never copy secrets to YAML, metadata, logs, or exports.

## Verification

1. Check health and Scheduler status.
2. Confirm Profile ID, adapter, public URL, and `api_key_configured`.
3. Confirm the exact model ID; refresh only if necessary.
4. Run one short request at concurrency one.
5. Inspect Runner summary and then logs on failure.

401 usually indicates an invalid or missing key. 404/unknown model usually indicates model
or base-path mismatch. When catalog succeeds but inference fails, trust the smoke request's
HTTP evidence. In multi-Backend deployments, reload every instance.

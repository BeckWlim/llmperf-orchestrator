# LLMPerf Input and Output Contract

## Input pipeline

All external input follows one path:

```text
CLI/YAML/JSON/environment
  -> decode
  -> strict validation
  -> resolve Backend-owned references
  -> safety assessment
  -> atomic persistence
```

- Parse YAML safely and require a top-level mapping. Strict Pydantic models reject unknown
  fields.
- The CLI owns file, argument, and HTTP boundaries. It never accesses PostgreSQL or expands
  scheduling semantics.
- Provider URLs, API keys, database credentials, and private keys belong only to Backend or
  CLI configuration. Workloads carry stable IDs.
- Accept secrets through `--stdin`, permission-controlled configuration, or environment
  variables. Never place them in arguments, logs, YAML, metadata, summaries, or exports.
- Resolve tokenizer/dataset references to Backend-owned local artifacts and immutable
  revisions before queueing.
- Persist a complete Campaign, RunnerPlan, and compiled Dispatch graph in one transaction.
  Resolution or admission failure must leave no partial workload.
- Bound file size, requests, tokens, concurrency, timeout, occurrences, graph expansion,
  and artifact resolution.

## Output pipeline

```text
PostgreSQL/API authoritative record
  -> command adapter
  -> resource projector
  -> CLIProjection
  -> centralized renderer or versioned export
```

- HTTP execution returns structured data and never prints directly.
- Register every command route explicitly. Adapters validate response shape before calling
  a projector; do not add identity adapters or raw fallbacks.
- Projectors allow-list stable fields. Do not clone a full record and remove a few keys.
- Default status/list/health output is lightweight and human-readable. `--json` serializes
  the same projection. `--full` remains an explicit expanded allow-list.
- Large complete records are available only through versioned export files.
- Worker stdout/stderr appears only through `logs` or explicit full export.
- Start, cancel, and export actions write progress and durable IDs to stderr and do not dump
  raw responses to stdout.
- Artifact transfers use path-free absolute `completed_bytes/total_bytes` events and
  heartbeats. Render a dynamic byte counter on terminal stderr and structured event records
  on redirected stderr; do not introduce terminal progress-bar rendering.
- `render_result` is the only CLI rendering-policy entry point; it must reject raw dict/list
  responses.

## Data minimization

Default projectors remove credentials, Authorization data, private-key paths, database
URLs, internal absolute paths, key rotation details, complete Worker streams, request
bodies, prompts, large summaries, and unstable implementation fields. Authorized logs,
expanded allow-lists, or versioned exports are explicit diagnostic paths; projection is not
a substitute for Backend authorization and redaction.

Errors should include bounded HTTP/Provider codes, reason, method, API path, elapsed time,
request ID, validation locations, and the first actionable message. Filter input bodies,
credential fields, oversized values, and normal-command tracebacks.

## Command modes

| Mode | stdout | stderr | Scope |
|---|---|---|---|
| default query | stable text projection | operation logs | allow-listed fields |
| `--json` | same projection as JSON | operation logs | allow-listed fields |
| `--full` | detailed projection | operation logs | expanded allow-list |
| `logs` | bounded Worker streams | operation logs | stdout/stderr |
| `export -o` | silent by default | file/result status | versioned export |
| start/cancel | silent by default | IDs and transitions | operation summary |

## Change checklist

1. Update strict models/decoders and invalid, unknown, and boundary tests.
2. Update the resource projector and verify secrets and large nested records remain absent.
3. Register new command adapters and route all display modes through the central renderer.
4. Update help, skill references, and API/export version documentation.
5. Test semantic parity between text and JSON projections, allow-listed `--full`, raw-payload
   rejection, and stdout/stderr separation.
6. Audit representative real responses for sensitive fields and size, then run focused and
   complete tests.

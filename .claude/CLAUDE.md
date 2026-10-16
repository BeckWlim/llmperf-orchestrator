# LLMPerf engineering guide

Use the maintained repository references rather than duplicating architecture rules here:

- [Documentation index](../docs/README.md)
- [Source layout and ownership](../docs/architecture.md#source-layout-and-references)
- [Task compiler and definition rules](../docs/tasks.md)
- [Runtime, persistence, and recovery](../docs/runtime.md)
- [Testing](../docs/testing.md)
- [Engineering workflow](../.codex/skills/operate-llmperf/references/engineering.md)

Python requires 3.10 or newer. PostgreSQL is the durable store; tests do not silently
substitute SQLite. The runtime coordinator owns task readiness, Scheduler owns Worker
handles and Ray capacity, and Workers never connect to PostgreSQL. Provider profiles are
Backend-owned and support explicit Provider reloads; other service settings follow the
restart requirements in the usage guide.

Keep source and schema changes consistent with `sql/postgresql/init.sql`. Run static checks,
compilation, formatting, and focused/full tests. PostgreSQL integration tests require an
explicit disposable database whose name contains `test`; never use production data.

```bash
.venv/bin/python -m basedpyright
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m black --check --workers 1 src tests
.venv/bin/pytest -q
.venv/bin/python -m llmperf_cli --help
```

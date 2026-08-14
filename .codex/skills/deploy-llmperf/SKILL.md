---
name: deploy-llmperf
description: "Guide and execute safe LLMPerf installation and deployment. Use when Codex needs to prepare a host, install Python/PostgreSQL dependencies, create the virtual environment and schema, configure Backend and Provider settings, install or update the Ubuntu systemd service, expose a remote Backend securely, configure llmperfctl clients or authentication, verify health and smoke inference, diagnose deployment failures, or hand off operational commands."
---

# Deploy LLMPerf

Deploy the checked-out project with PostgreSQL as its durable store, a non-root
Backend process, Backend-owned Provider credentials, and an explicit verification path.

## Load deployment instructions

- Read [references/deployment.md](references/deployment.md) completely before planning,
  executing, changing, or diagnosing a deployment.
- Read `deploy/systemd/README.md` and
  `deploy/systemd/llmperf-backend.service.template` before any systemd action.
- Read `.codex/skills/operate-llmperf/references/provider.md` before configuring or
  diagnosing a Provider Profile.
- Use `$operate-llmperf` after deployment when the task continues into workload YAML,
  benchmark execution, or result diagnosis.

## Establish scope before mutation

Determine:

- whether the user wants instructions, a readiness audit, or execution;
- target OS, host, repository path, service user/group, and foreground versus systemd;
- local or remote PostgreSQL and whether the database already contains valuable data;
- local-only or remote API access, authentication/TLS boundary, and CLI host;
- Provider adapter/profile/model and required optional Python extras;
- whether sudo, package installation, database creation, service enablement, and smoke
  Provider calls are authorized.

Make read-only discoveries without asking. Do not perform privileged installation,
overwrite an existing unit, alter a valuable database, expose a listening socket, or make
paid Provider calls unless the requested deployment scope authorizes that action.

## Follow the deployment workflow

1. Inspect `git status --short`; preserve user changes and record the exact commit/path.
2. Audit Python, virtual environment, PostgreSQL client/server reachability, systemd,
   ports, current Backend config path, and any existing service without changing them.
3. Select a supported topology: foreground development or persistent Ubuntu systemd.
   Do not invent Docker/Kubernetes assets; create them only when explicitly requested.
4. Install only missing prerequisites with the platform's package manager after approval.
5. Create/update `.venv` and install the project plus only required optional extras.
6. Create or verify a PostgreSQL database/least-privilege role. Apply schema only to the
   resolved target and never substitute SQLite or a production database as a test DB.
7. Configure `DATABASE_URL`, Backend host/port, caches, and Provider Profiles with
   `llmperf-backend config`; send secrets only through `--stdin`.
8. Prove the prepared environment in the foreground before installing systemd. Resolve
   imports, configuration, database, and port failures here.
9. For systemd, render the checked-in template to `/etc/systemd/system`; never start the
   template in place. Verify the rendered unit before `daemon-reload` and enable/start.
10. Verify health, Scheduler, Planner, public Provider profile, exact model catalog, and a
    bounded 1x1 smoke Runner when Provider inference verification is in scope.
11. Configure remote `llmperfctl` separately from Backend settings. Require HTTPS plus
    authentication for untrusted networks; prefer a reverse proxy over directly exposing
    Uvicorn.
12. Hand off service status/journal/restart commands, config paths, database target,
    Provider/model, smoke ID/outcome, backups, unresolved security work, and rollback steps.

## Preserve deployment invariants

- Run the Backend as the configured ordinary user, never root.
- Keep Provider keys and database credentials out of unit files, source, workload YAML,
  shell history, logs, and final responses.
- Keep PostgreSQL authoritative. A CLI disconnect or Backend restart must not erase work.
- Keep one Backend/Uvicorn process unless the architecture is deliberately changed;
  Scheduler/Planner behavior and in-memory model discovery assume this default.
- Keep Worker/Ray children in the systemd cgroup with `KillMode=control-group`.
- Treat `config set` results as restart-required. Verify the running process loaded the
  expected service user config after restart.
- Treat `/models` visibility as catalog evidence only; use a bounded smoke request to prove
  inference.

## Diagnose from the deployment boundary

Check in this order:

1. `systemctl status` and `journalctl` for process/startup failures.
2. Service `User`, `WorkingDirectory`, `ExecStart`, config path, file permissions, and port.
3. PostgreSQL URL, socket/host reachability, role/database privileges, and schema state.
4. `llmperfctl health`, Scheduler, Planner, Provider list, and model discovery.
5. Smoke Runner summary, then dedicated Runner logs for Worker/Ray/Provider failures.

Do not mask the first concrete error by repeatedly restarting or re-running paid workloads.

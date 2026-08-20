# LLMPerf Deployment Reference

## Supported shapes

- Development/verification: repository `.venv`, foreground `llmperf-backend`.
- Persistent Ubuntu: render and install `deploy/systemd/llmperf-backend.service.template`
  under a non-root service user.
- PostgreSQL is the only runtime database. Do not invent Docker/Kubernetes assets when the
  repository does not provide them.

## Read-only preflight

Resolve actual paths, users, ports, configuration, service state, and dirty worktree before
changing anything:

```bash
pwd
git status --short
python3 --version
psql --version
systemctl --version
ss -ltn
./.venv/bin/llmperf-backend config path
systemctl status --no-pager llmperf-backend.service
```

Before overwriting a deployment or changing a database, identify ownership, purpose,
backup, and rollback strategy.

## Python and PostgreSQL

LLMPerf requires Python 3.10+. Install only adapter extras that are needed.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
createdb llmperf
psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
./.venv/bin/llmperf-backend config set DATABASE_URL postgresql+asyncpg:///llmperf
```

Production should use a dedicated non-superuser role and explicit host/database/role.
Never print database passwords. SQLAlchemy `create_all` does not migrate an existing
schema. Review schema differences, active Campaigns, and backups before upgrades.

Version 1.0.0 is the first supported database contract. Do not start it against a
pre-release schema: export anything still needed, back up PostgreSQL, then initialize a
clean 1.0.0 schema. A separate database and service port is the safe parallel deployment
option.

## Persisted Backend configuration

Run configuration commands as the final service user. The default file is
`~/.config/llmperf/backend.env` with directory mode 0700 and file mode 0600.

```bash
./.venv/bin/llmperf-backend config set LLMPERF_SERVER_HOST 127.0.0.1
./.venv/bin/llmperf-backend config set LLMPERF_SERVER_PORT 8000
./.venv/bin/llmperf-backend config list
```

Store Provider keys through `--stdin`; do not place secrets in the unit. Client
`llmperfctl config` does not replace Backend configuration. Restart after non-Provider
configuration changes and verify the loaded config path.

## Foreground acceptance

Before systemd installation, start from the repository root and verify health from another
terminal. Resolve imports, PostgreSQL, ports, configuration, and artifact permissions here.

```bash
./.venv/bin/llmperf-backend config list
./.venv/bin/llmperf-backend
./.venv/bin/llmperfctl health
```

## Ubuntu systemd

Resolve the template placeholders to an absolute repository path and explicit non-root
user/group. Do not repurpose `HOME`.

```bash
llmperf_root="$PWD"
llmperf_user="$(id -un)"
llmperf_group="$(id -gn)"
sed \
  -e "s|@LLMPERF_ROOT@|$llmperf_root|g" \
  -e "s|@LLMPERF_USER@|$llmperf_user|g" \
  -e "s|@LLMPERF_GROUP@|$llmperf_group|g" \
  deploy/systemd/llmperf-backend.service.template \
  | sudo tee /etc/systemd/system/llmperf-backend.service >/dev/null
sudo chmod 0644 /etc/systemd/system/llmperf-backend.service
sudo systemd-analyze verify /etc/systemd/system/llmperf-backend.service
sudo systemctl daemon-reload
sudo systemctl enable --now llmperf-backend.service
sudo systemctl status --no-pager llmperf-backend.service
```

Inspect and obtain approval before replacing an existing unit. `KillMode=control-group`
must remain so service stop cleans up Worker/Ray processes.

## Remote access and auth

Keep the Backend on loopback when possible and use an HTTPS reverse proxy plus source
firewall restrictions. JWT signing does not replace TLS. Generate a dedicated RSA 3072+
key; the Backend receives only its public key and the private key remains mode 0600.

```bash
mkdir -p ~/.config/llmperf/keys
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 \
  -out ~/.config/llmperf/keys/ctl-private.pem
chmod 0600 ~/.config/llmperf/keys/ctl-private.pem
openssl pkey -in ~/.config/llmperf/keys/ctl-private.pem -pubout \
  -out ~/.config/llmperf/keys/ctl-public.pem
./.venv/bin/llmperf-backend config set LLMPERF_AUTH_ENABLED true
./.venv/bin/llmperf-backend config set LLMPERF_AUTH_KEY \
  /absolute/path/to/ctl-public.pem
```

Configure remote CLI URL/private key separately. Never expose unauthenticated Uvicorn on
public `0.0.0.0`.

## Acceptance and troubleshooting

```bash
llmperfctl health
llmperfctl scheduler status
llmperfctl planner runtime
llmperfctl provider list
llmperfctl provider models PROVIDER_ID
llmperfctl runner start -f examples/example-smoke.yaml -w
```

Run paid smoke inference only when authorized. Record the Runner ID, summary, and first
actionable log error.

- Unit failure: inspect User/Group, absolute paths, `.venv`, and permissions.
- Database failure: verify `postgresql+asyncpg`, socket/host, role, and schema.
- Port failure: inspect `ss -ltn` and server host/port settings.
- Stale configuration: verify service user, config path, and restart/reload boundary.
- Worker imports: reinstall editable project from the intended repository.
- Provider failure: inspect public Profile/catalog, then smoke summary/logs.
- Residual Workers: preserve `KillMode=control-group`.

For updates, inspect the worktree and commit, back up PostgreSQL, review schema changes,
reinstall dependencies, then restart and inspect journal output. Rollback must account for
code, dependencies, and schema together; never roll back only code across an incompatible
database change.

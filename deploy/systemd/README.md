# Ubuntu systemd template

`deploy/systemd/` contains deployment templates and instructions only. systemd
must not start a service file directly from this repository directory. The rendered
unit is installed as a system service under `/etc/systemd/system`, but the Backend
process runs as a configured ordinary user rather than root. The template assumes
that the repository already has a working `.venv`, the PostgreSQL schema has been
initialized, and `llmperf-backend config set ...` has created the Backend settings
for that user.

## 1. Verify the prepared environment

From the repository root, confirm the existing virtual environment can start the
Backend before installing the unit:

```bash
./.venv/bin/llmperf-backend config list
./.venv/bin/llmperf-backend
```

Stop the foreground process after the startup check. Backend credentials remain in
`~/.config/llmperf/backend.env`; the systemd unit never contains Provider keys.

## 2. Render and install the template

Render a copy into the system unit directory. The commands below use the current
login user, group, and absolute repository path; the repository template remains
unchanged:

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
```

The unit intentionally contains no `Environment=` declarations. `User` selects the
account whose `~/.config/llmperf/backend.env` is discovered by the Backend, while
`WorkingDirectory` provides the project-local `.env` fallback and Worker directory.
Backend defaults keep one Uvicorn process unless its own configuration overrides it.

Review the installed unit before starting it:

```bash
sudo systemd-analyze verify /etc/systemd/system/llmperf-backend.service
```

## 3. Start the installed service

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llmperf-backend.service
sudo systemctl status --no-pager llmperf-backend.service
```

## 4. Operate and update

```bash
sudo journalctl -u llmperf-backend.service -f
sudo systemctl restart llmperf-backend.service
sudo systemctl stop llmperf-backend.service
```

After moving the repository, render the template again, run `daemon-reload`, and
restart the service. Changes to Backend configuration also require a restart.

`Restart=always` restores the Backend after unexpected exits. `KillMode=control-group`
ensures Scheduler-owned Worker and Ray subprocesses do not survive a service stop.
The unit forces one Backend/Uvicorn process; benchmark state and results remain in
PostgreSQL.

## 5. Migrate Hugging Face caches safely

Hugging Face snapshot directories contain relative symlinks into content-addressed
`blobs/` directories. Some copy tools retain blobs and non-empty or partially populated
snapshot directories but omit some links. The mapping from repository filenames to blob
hashes cannot be inferred reliably from the remaining files or blob contents, so capture
a manifest on the healthy source before transfer:

```bash
python deploy/huggingface_cache_links.py capture \
  --cache-root ~/.cache/llmperf/tokenizers/downloads \
  --manifest tokenizer-cache-links.json
```

Transfer the cache and manifest while no artifact download is active. Preserve symlinks
with `rsync -a`, and exclude `.locks/`, `*.lock`, and `*.incomplete`. On the destination,
audit first; this command is read-only:

```bash
python deploy/huggingface_cache_links.py audit \
  --cache-root ~/.cache/llmperf/tokenizers/downloads \
  --manifest tokenizer-cache-links.json
```

Repair only missing links whose referenced blobs are already present:

```bash
python deploy/huggingface_cache_links.py repair \
  --cache-root ~/.cache/llmperf/tokenizers/downloads \
  --manifest tokenizer-cache-links.json
```

The script never downloads artifacts and never overwrites existing files or conflicting
links. Repeat the same workflow with the configured dataset cache root. A manifest must
come from the same healthy source cache; an empty destination snapshot plus anonymous
blob hashes is not sufficient to reconstruct the mapping safely.

-- LLMPerf PostgreSQL 16 schema
-- Usage:
--   psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
--
-- This script is idempotent for an empty/current schema. It creates no users:
-- the configured bootstrap public key provides first-run superuser access.

BEGIN;

CREATE TABLE IF NOT EXISTS benchmark_campaigns (
    id                  VARCHAR(36) PRIMARY KEY,
    name                VARCHAR(200) NOT NULL,
    description         TEXT,
    tags                JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by          VARCHAR(64) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_campaigns_created_at
    ON benchmark_campaigns (created_at DESC);

CREATE TABLE IF NOT EXISTS benchmark_runner_plans (
    id                      VARCHAR(36) PRIMARY KEY,
    campaign_id             VARCHAR(36) NOT NULL
                            REFERENCES benchmark_campaigns(id) ON DELETE CASCADE,
    name                    VARCHAR(200) NOT NULL,
    status                  VARCHAR(20) NOT NULL,
    timezone                VARCHAR(64) NOT NULL,
    recurrence              JSONB NOT NULL,
    overlap_policy          VARCHAR(20) NOT NULL,
    runner_template         JSONB NOT NULL,
    template_version        INTEGER NOT NULL DEFAULT 1,
    starts_at               TIMESTAMPTZ NOT NULL,
    ends_at                 TIMESTAMPTZ,
    max_occurrences         INTEGER,
    next_fire_at            TIMESTAMPTZ,
    last_fire_at            TIMESTAMPTZ,
    occurrence_cursor       INTEGER NOT NULL DEFAULT 0,
    emitted_count           INTEGER NOT NULL DEFAULT 0,
    skipped_count           INTEGER NOT NULL DEFAULT 0,
    misfire_grace_seconds   INTEGER NOT NULL DEFAULT 60,
    created_by              VARCHAR(64) NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_runner_plan_status CHECK (
        status IN ('active', 'paused', 'completed', 'cancelled')
    ),
    CONSTRAINT ck_runner_plan_boundary CHECK (
        ends_at IS NOT NULL OR max_occurrences IS NOT NULL
    ),
    CONSTRAINT ck_runner_plan_time_range CHECK (
        ends_at IS NULL OR ends_at > starts_at
    ),
    CONSTRAINT ck_runner_plan_overlap_policy CHECK (
        overlap_policy IN ('queue', 'skip')
    ),
    CONSTRAINT ck_runner_plan_occurrences CHECK (
        max_occurrences IS NULL OR max_occurrences > 0
    )
);

CREATE INDEX IF NOT EXISTS ix_runner_plan_due
    ON benchmark_runner_plans (next_fire_at)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS ix_runner_plan_campaign
    ON benchmark_runner_plans (campaign_id, created_at DESC);

CREATE TABLE IF NOT EXISTS benchmark_runners (
    id                  VARCHAR(36) PRIMARY KEY,
    campaign_id         VARCHAR(36)
                        REFERENCES benchmark_campaigns(id) ON DELETE SET NULL,
    runner_plan_id      VARCHAR(36)
                        REFERENCES benchmark_runner_plans(id) ON DELETE SET NULL,
    plan_occurrence     INTEGER,
    scheduled_for       TIMESTAMPTZ,
    plan_template_version INTEGER,
    label               VARCHAR(200),
    created_by          VARCHAR(64) NOT NULL,
    status              VARCHAR(20) NOT NULL,
    benchmark_config    JSONB NOT NULL,
    user_metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    heartbeat_at        TIMESTAMPTZ,
    cancel_requested    BOOLEAN NOT NULL DEFAULT FALSE,
    scheduler_id        VARCHAR(100),
    process_id          INTEGER,
    exit_code           INTEGER,
    summary             JSONB,
    request_count       INTEGER NOT NULL DEFAULT 0,
    error_message       TEXT,
    stdout              TEXT,
    stderr              TEXT,
    CONSTRAINT ck_benchmark_runners_status CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
    ),
    CONSTRAINT ck_benchmark_runners_request_count CHECK (request_count >= 0),
    CONSTRAINT uq_runner_plan_occurrence UNIQUE (
        runner_plan_id, plan_occurrence
    )
);

CREATE INDEX IF NOT EXISTS ix_runners_campaign_id
    ON benchmark_runners (campaign_id);
CREATE INDEX IF NOT EXISTS ix_runners_status_created_at
    ON benchmark_runners (status, created_at);
CREATE INDEX IF NOT EXISTS ix_runners_queue_created_at
    ON benchmark_runners (created_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS ix_runners_created_at
    ON benchmark_runners (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_runner_plan_time
    ON benchmark_runners (runner_plan_id, scheduled_for);

CREATE TABLE IF NOT EXISTS benchmark_request_results (
    runner_id           VARCHAR(36) NOT NULL
                        REFERENCES benchmark_runners(id) ON DELETE CASCADE,
    sequence            INTEGER NOT NULL,
    metrics             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (runner_id, sequence),
    CONSTRAINT ck_request_result_sequence CHECK (sequence >= 0)
);

CREATE INDEX IF NOT EXISTS ix_request_results_runner_id
    ON benchmark_request_results (runner_id);

CREATE TABLE IF NOT EXISTS benchmark_runner_events (
    id                  INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    runner_id           VARCHAR(36) NOT NULL
                        REFERENCES benchmark_runners(id) ON DELETE CASCADE,
    status              VARCHAR(20) NOT NULL,
    message             TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_runner_events_status CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
    )
);

CREATE INDEX IF NOT EXISTS ix_runner_events_runner_id
    ON benchmark_runner_events (runner_id, id);

CREATE TABLE IF NOT EXISTS benchmark_runner_plan_events (
    id                  INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    runner_plan_id      VARCHAR(36) NOT NULL
                        REFERENCES benchmark_runner_plans(id) ON DELETE CASCADE,
    event_type          VARCHAR(30) NOT NULL,
    occurrence          INTEGER,
    scheduled_for       TIMESTAMPTZ,
    runner_id           VARCHAR(36)
                        REFERENCES benchmark_runners(id) ON DELETE SET NULL,
    message             TEXT,
    details             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_runner_plan_event_time
    ON benchmark_runner_plan_events (runner_plan_id, created_at);

-- User identity and authorization level are independent from key material.
CREATE TABLE IF NOT EXISTS users (
    username            VARCHAR(64) PRIMARY KEY,
    display_name        VARCHAR(200),
    email               VARCHAR(320),
    role                VARCHAR(20) NOT NULL,
    enabled             BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by          VARCHAR(64) NOT NULL,
    CONSTRAINT ck_users_role CHECK (role IN ('viewer', 'operator', 'superuser'))
);

CREATE INDEX IF NOT EXISTS ix_users_enabled_role
    ON users (enabled, role);

-- A user may have multiple keys during rotation. valid_until implements the
-- grace window for previous keys without changing user identity.
CREATE TABLE IF NOT EXISTS trusted_client_keys (
    key_id              VARCHAR(32) PRIMARY KEY,
    username            VARCHAR(64) NOT NULL
                        REFERENCES users(username) ON DELETE CASCADE,
    public_key_pem      TEXT NOT NULL,
    enabled             BOOLEAN NOT NULL DEFAULT TRUE,
    valid_until         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by          VARCHAR(64) NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_trusted_keys_username
    ON trusted_client_keys (username);
CREATE INDEX IF NOT EXISTS ix_trusted_keys_validity
    ON trusted_client_keys (enabled, valid_until);

-- This table deliberately has no FK to users so audit records survive any
-- future user-deletion policy.
CREATE TABLE IF NOT EXISTS trusted_client_events (
    id                  INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    username            VARCHAR(64) NOT NULL,
    key_id              VARCHAR(32),
    action              VARCHAR(20) NOT NULL,
    actor               VARCHAR(64) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_trusted_client_events_action CHECK (
        action IN ('created', 'updated', 'rotated', 'revoked')
    )
);

CREATE INDEX IF NOT EXISTS ix_trusted_events_username
    ON trusted_client_events (username, id DESC);
CREATE INDEX IF NOT EXISTS ix_trusted_events_created_at
    ON trusted_client_events (created_at DESC);

COMMIT;

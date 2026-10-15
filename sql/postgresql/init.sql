-- LLMPerf: initialize an empty PostgreSQL database.
-- Existing schemas require a fresh database; this is not a migration.
BEGIN;


CREATE TABLE benchmark_campaigns (
    id VARCHAR(36) NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    tags JSONB NOT NULL,
    created_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id)
)

;

CREATE INDEX ix_benchmark_campaigns_created_at ON benchmark_campaigns (created_at);


CREATE TABLE execution_leases (
    id SERIAL NOT NULL,
    owner VARCHAR(100) NOT NULL,
    generation INTEGER NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT ck_lease_singleton CHECK (id = 1 AND generation > 0)
)

;


CREATE TABLE trusted_client_events (
    id SERIAL NOT NULL,
    username VARCHAR(64) NOT NULL,
    key_id VARCHAR(32),
    action VARCHAR(20) NOT NULL,
    actor VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id)
)

;

CREATE INDEX ix_trusted_client_events_username ON trusted_client_events (username);


CREATE TABLE users (
    username VARCHAR(64) NOT NULL,
    display_name VARCHAR(200),
    email VARCHAR(320),
    role VARCHAR(20) NOT NULL,
    enabled BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_by VARCHAR(64) NOT NULL,
    PRIMARY KEY (username)
)

;


CREATE TABLE benchmark_runner_plans (
    id VARCHAR(36) NOT NULL,
    campaign_id VARCHAR(36) NOT NULL,
    name VARCHAR(200) NOT NULL,
    status VARCHAR(20) NOT NULL,
    timezone VARCHAR(64) NOT NULL,
    recurrence JSONB NOT NULL,
    overlap_policy VARCHAR(20) NOT NULL,
    runner_template JSONB NOT NULL,
    template_version VARCHAR(16) NOT NULL,
    starts_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ends_at TIMESTAMP WITH TIME ZONE,
    max_occurrences INTEGER,
    next_fire_at TIMESTAMP WITH TIME ZONE,
    last_fire_at TIMESTAMP WITH TIME ZONE,
    occurrence_cursor INTEGER NOT NULL,
    emitted_count INTEGER NOT NULL,
    skipped_count INTEGER NOT NULL,
    misfire_grace_seconds INTEGER NOT NULL,
    created_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT ck_runner_plan_status CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
    CONSTRAINT ck_runner_plan_boundary CHECK (ends_at IS NOT NULL OR max_occurrences IS NOT NULL),
    CONSTRAINT ck_runner_plan_time_range CHECK (ends_at IS NULL OR ends_at > starts_at),
    CONSTRAINT ck_runner_plan_overlap_policy CHECK (overlap_policy IN ('queue', 'skip')),
    FOREIGN KEY(campaign_id) REFERENCES benchmark_campaigns (id) ON DELETE CASCADE
)

;

CREATE INDEX ix_runner_plan_campaign ON benchmark_runner_plans (campaign_id, created_at);

CREATE INDEX ix_runner_plan_due ON benchmark_runner_plans (next_fire_at) WHERE status = 'active';


CREATE TABLE benchmark_task_definitions (
    id VARCHAR(36) NOT NULL,
    campaign_id VARCHAR(36) NOT NULL,
    name VARCHAR(200) NOT NULL,
    format_version VARCHAR(16) NOT NULL,
    config JSONB NOT NULL,
    runner_template JSONB NOT NULL,
    created_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(campaign_id) REFERENCES benchmark_campaigns (id) ON DELETE CASCADE
)

;

CREATE INDEX ix_task_definition_campaign ON benchmark_task_definitions (campaign_id, created_at);


CREATE TABLE trusted_client_keys (
    key_id VARCHAR(32) NOT NULL,
    username VARCHAR(64) NOT NULL,
    public_key_pem TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_by VARCHAR(64) NOT NULL,
    PRIMARY KEY (key_id),
    FOREIGN KEY(username) REFERENCES users (username) ON DELETE CASCADE
)

;

CREATE INDEX ix_trusted_client_keys_username ON trusted_client_keys (username);


CREATE TABLE benchmark_runners (
    id VARCHAR(36) NOT NULL,
    campaign_id VARCHAR(36),
    runner_plan_id VARCHAR(36),
    plan_occurrence INTEGER,
    scheduled_for TIMESTAMP WITH TIME ZONE,
    plan_template_version VARCHAR(16),
    label VARCHAR(200),
    created_by VARCHAR(64) NOT NULL,
    status VARCHAR(20) NOT NULL,
    benchmark_config JSONB NOT NULL,
    user_metadata JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    heartbeat_at TIMESTAMP WITH TIME ZONE,
    cancel_requested BOOLEAN NOT NULL,
    scheduler_id VARCHAR(100),
    exit_code INTEGER,
    summary JSONB,
    request_count INTEGER NOT NULL,
    error_message TEXT,
    stdout TEXT,
    stderr TEXT,
    PRIMARY KEY (id),
    CONSTRAINT uq_runner_plan_occurrence UNIQUE (runner_plan_id, plan_occurrence),
    FOREIGN KEY(campaign_id) REFERENCES benchmark_campaigns (id) ON DELETE SET NULL,
    FOREIGN KEY(runner_plan_id) REFERENCES benchmark_runner_plans (id) ON DELETE SET NULL
)

;

CREATE INDEX ix_benchmark_runners_campaign_id ON benchmark_runners (campaign_id);

CREATE INDEX ix_benchmark_runners_created_at ON benchmark_runners (created_at);

CREATE INDEX ix_runner_plan_time ON benchmark_runners (runner_plan_id, scheduled_for);

CREATE INDEX ix_runners_queue_created_at ON benchmark_runners (created_at) WHERE status = 'queued';

CREATE INDEX ix_runners_running_campaign ON benchmark_runners (campaign_id) WHERE status = 'running';

CREATE INDEX ix_runners_status_created_at ON benchmark_runners (status, created_at);


CREATE TABLE benchmark_task_instances (
    id VARCHAR(36) NOT NULL,
    definition_id VARCHAR(36) NOT NULL,
    campaign_id VARCHAR(36) NOT NULL,
    format_version VARCHAR(16) NOT NULL,
    instance_key VARCHAR(512) NOT NULL,
    state VARCHAR(20) NOT NULL,
    spec JSONB NOT NULL,
    checkpoint JSONB NOT NULL,
    outcome JSONB NOT NULL,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_task_instance_key UNIQUE (definition_id, instance_key),
    CONSTRAINT ck_task_instance_state CHECK (state IN ('planned', 'active', 'completed', 'failed', 'cancelled')),
    FOREIGN KEY(definition_id) REFERENCES benchmark_task_definitions (id) ON DELETE CASCADE,
    FOREIGN KEY(campaign_id) REFERENCES benchmark_campaigns (id) ON DELETE CASCADE
)

;

CREATE INDEX ix_task_instance_campaign ON benchmark_task_instances (campaign_id, format_version, state);


CREATE TABLE benchmark_request_results (
    runner_id VARCHAR(36) NOT NULL,
    sequence INTEGER NOT NULL,
    metrics JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (runner_id, sequence),
    CONSTRAINT ck_request_result_sequence CHECK (sequence >= 0),
    FOREIGN KEY(runner_id) REFERENCES benchmark_runners (id) ON DELETE CASCADE
)

;


CREATE TABLE benchmark_runner_events (
    id SERIAL NOT NULL,
    runner_id VARCHAR(36) NOT NULL,
    status VARCHAR(20) NOT NULL,
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(runner_id) REFERENCES benchmark_runners (id) ON DELETE CASCADE
)

;

CREATE INDEX ix_benchmark_runner_events_runner_id ON benchmark_runner_events (runner_id);


CREATE TABLE benchmark_runner_plan_events (
    id SERIAL NOT NULL,
    runner_plan_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(30) NOT NULL,
    occurrence INTEGER,
    scheduled_for TIMESTAMP WITH TIME ZONE,
    runner_id VARCHAR(36),
    message TEXT,
    details JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(runner_plan_id) REFERENCES benchmark_runner_plans (id) ON DELETE CASCADE,
    FOREIGN KEY(runner_id) REFERENCES benchmark_runners (id) ON DELETE SET NULL
)

;

CREATE INDEX ix_runner_plan_event_time ON benchmark_runner_plan_events (runner_plan_id, created_at);


CREATE TABLE conversation_messages (
    instance_id VARCHAR(36) NOT NULL,
    conversation VARCHAR(64) NOT NULL,
    position INTEGER NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    runner_id VARCHAR(36) NOT NULL,
    PRIMARY KEY (instance_id, conversation, position),
    CONSTRAINT ck_message_position_role CHECK (position >= 0 AND role IN ('user', 'assistant')),
    FOREIGN KEY(instance_id) REFERENCES benchmark_task_instances (id) ON DELETE CASCADE,
    FOREIGN KEY(runner_id) REFERENCES benchmark_runners (id)
)

;


CREATE TABLE execution_nodes (
    id VARCHAR(36) NOT NULL,
    campaign_id VARCHAR(36) NOT NULL,
    runner_plan_id VARCHAR(36),
    dispatch_key VARCHAR(80),
    task_instance_id VARCHAR(36),
    node_id VARCHAR(80),
    due_at TIMESTAMP WITH TIME ZONE,
    state VARCHAR(20) NOT NULL,
    runner_template JSONB NOT NULL,
    lineage JSONB NOT NULL,
    runner_id VARCHAR(36),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    emitted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (id),
    CONSTRAINT uq_runner_dispatch_plan_key UNIQUE (runner_plan_id, dispatch_key),
    CONSTRAINT uq_runner_dispatch_task_node UNIQUE (task_instance_id, node_id),
    CONSTRAINT ck_runner_dispatch_state CHECK (state IN ('blocked', 'pending', 'emitted', 'cancelled')),
    CONSTRAINT ck_runner_dispatch_owner CHECK ((runner_plan_id IS NOT NULL AND task_instance_id IS NULL AND dispatch_key IS NOT NULL AND node_id IS NULL) OR (runner_plan_id IS NULL AND task_instance_id IS NOT NULL AND dispatch_key IS NULL AND node_id IS NOT NULL)),
    FOREIGN KEY(campaign_id) REFERENCES benchmark_campaigns (id) ON DELETE CASCADE,
    FOREIGN KEY(runner_plan_id) REFERENCES benchmark_runner_plans (id) ON DELETE CASCADE,
    FOREIGN KEY(task_instance_id) REFERENCES benchmark_task_instances (id) ON DELETE CASCADE
)

;

CREATE INDEX ix_runner_dispatch_campaign ON execution_nodes (campaign_id, created_at);

CREATE INDEX ix_runner_dispatch_due ON execution_nodes (due_at) WHERE state = 'pending';

CREATE INDEX ix_runner_dispatch_runner ON execution_nodes (runner_id);

CREATE INDEX ix_runner_dispatch_task ON execution_nodes (task_instance_id, node_id);

CREATE UNIQUE INDEX uq_runner_dispatch_plan_runner ON execution_nodes (runner_id) WHERE runner_plan_id IS NOT NULL;


CREATE TABLE task_attempts (
    id VARCHAR(36) NOT NULL,
    runner_id VARCHAR(36) NOT NULL,
    generation INTEGER NOT NULL,
    state VARCHAR(20) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (id),
    CONSTRAINT ck_attempt_state CHECK (state IN ('executing', 'succeeded', 'failed', 'cancelled', 'indeterminate')),
    UNIQUE (runner_id),
    FOREIGN KEY(runner_id) REFERENCES benchmark_runners (id) ON DELETE CASCADE
)

;

COMMIT;

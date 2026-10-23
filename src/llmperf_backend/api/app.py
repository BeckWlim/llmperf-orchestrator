"""FastAPI application for configuration and durable benchmark orchestration."""

import asyncio
from contextlib import asynccontextmanager
import copy
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.engine import make_url
from pydantic import ValidationError

from llmperf.version import PROTOCOL_VERSION
from llmperf_backend.api.progress import attach_progress
from llmperf_backend.compiler.requests import planned_worker_count
from llmperf_backend.artifacts.resolver import (
    ArtifactCaches,
    ArtifactDownloadProgress,
    ArtifactProgressCallback,
    ArtifactResolution,
    ArtifactValidationError,
    DatasetCache,
    DatasetResolutionError,
    DatasetResolver,
    TokenizerCache,
    TokenizerResolutionError,
    TokenizerResolver,
    artifact_progress_scope,
    is_immutable_huggingface_revision,
    validate_resolved_artifacts,
)
from llmperf_backend.api.auth import TokenVerifier, normalize_public_key
from llmperf_backend.configuration.loader import (
    ConfigError,
    ConfigSnapshot,
    ConfigStore,
    load_provider_profiles,
)
from llmperf_backend.configuration.models import (
    BenchmarkConfig,
    BenchmarkCampaignCreate,
    BenchmarkCampaignStart,
    BenchmarkRunnerBatchCreate,
    BenchmarkRunnerCreate,
    BenchmarkRunnerSpec,
    ResolvedBenchmarkConfig,
    RunnerPlanCreate,
    RunnerPlanPreview,
    TrustedClientWrite,
    YAMLValidationRequest,
    app_config_schema,
    dump_model,
)
from llmperf_backend.storage.repository import (
    CANCELLED,
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    Database,
    RunnerRepository,
    json_safe,
    PLAN_STATUSES,
)
from llmperf_backend.runtime.planner import Planner, preview_fires
from llmperf_backend.providers.registry import (
    ProviderConfigError,
    ProviderDiscoveryError,
    ProviderModelDiscovery,
    ProviderRegistry,
)
from llmperf_backend.runtime.scheduler import Scheduler
from llmperf_backend.compiler.safety import WorkloadSafetyError, assess_workload
from llmperf_backend.compiler.tasks import (
    TaskCompileContext,
    TaskCompiler,
    parallel_request_groups,
)

RUNNER_STATUSES = {QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED}
LOGGER = logging.getLogger(__name__)


def _compile_campaign_task_preview(
    payload: BenchmarkCampaignStart,
    instance_limit: int,
    node_limit: int,
    include_debug: bool,
) -> Dict[str, Any]:
    """Compile TaskDefinitions into a bounded, persistence-free preview."""

    definition_previews: List[Dict[str, Any]] = []
    total_instances = 0
    total_nodes = 0
    previewed_instances = 0
    for definition_index, definition in enumerate(payload.task_definitions):
        recipe = definition.recipe()
        compile_context = TaskCompileContext(
            definition_id=f"preview-{definition_index}",
            definition_name=definition.name,
            definition=recipe,
            runner_template=definition.runner.execution_document(),
        )
        compilation_table = TaskCompiler(compile_context).compile()
        instance_count = len(compilation_table.instances)
        node_count = len(compilation_table.nodes)
        remaining_instance_count = max(instance_limit - previewed_instances, 0)
        visible_instances = compilation_table.instances[:remaining_instance_count]
        instance_previews: List[Dict[str, Any]] = []
        for instance_row in visible_instances:
            node_previews: List[Dict[str, Any]] = []
            instance_nodes = compilation_table.nodes_for(instance_row.instance_key)
            execution_groups = {
                member.node_id: (group[0].node_id, len(group))
                for group in parallel_request_groups(instance_nodes)
                for member in group
            }
            for node_row in instance_nodes[:node_limit]:
                template_benchmark = definition.runner.execution_benchmark()
                node_preview: Dict[str, Any] = {
                    "node_id": node_row.node_id,
                    "dependencies": list(node_row.dependencies),
                    "after_seconds": node_row.after_seconds,
                    "role": node_row.role,
                    "execution_kind": node_row.execution_kind,
                    "planned_workers": (
                        1
                        if node_row.execution_kind == "atomic"
                        else (planned_worker_count(template_benchmark))
                    ),
                    "concurrent_workers": (
                        1
                        if node_row.execution_kind == "atomic"
                        else (definition.runner.concurrent_workers)
                    ),
                    "payload_id": node_row.payload_id,
                    "message_sources": list(node_row.message_sources),
                    "payload_kind": (
                        "literal" if node_row.prompt_text is not None else "generated"
                    ),
                    "execution_group": execution_groups[node_row.node_id][0],
                    "group_size": execution_groups[node_row.node_id][1],
                }
                if include_debug:
                    seed_field = (
                        "dataset_seed"
                        if node_row.execution_kind == "batch"
                        else "payload_seed"
                    )
                    node_preview[seed_field] = node_row.payload_seed
                    if node_row.prefix_seed is not None:
                        node_preview["prefix_seed"] = node_row.prefix_seed
                node_previews.append(node_preview)
            instance_previews.append(
                {
                    "instance_key": instance_row.instance_key,
                    "dimensions": instance_row.dimensions,
                    "trial_index": instance_row.trial_index,
                    "node_count": len(instance_nodes),
                    "shown_node_count": min(len(instance_nodes), node_limit),
                    "truncated_node_count": max(len(instance_nodes) - node_limit, 0),
                    "nodes": node_previews,
                }
            )
        shown_instance_count = len(instance_previews)
        definition_previews.append(
            {
                "name": definition.name,
                "instance_count": instance_count,
                "node_count": node_count,
                "shown_instance_count": shown_instance_count,
                "truncated_instance_count": instance_count - shown_instance_count,
                "instances": instance_previews,
            }
        )
        total_instances += instance_count
        total_nodes += node_count
        previewed_instances += shown_instance_count
    return {
        "version": PROTOCOL_VERSION,
        "valid": True,
        "campaign": payload.campaign.name,
        "debug": include_debug,
        "summary": {
            "immediate_runners": payload.immediate_runner_count
            * (payload.campaign.rounds.count if payload.campaign.rounds else 1),
            "runner_plans": len(payload.runner_plans),
            "task_definitions": len(payload.task_definitions),
            "task_instances": total_instances,
            "task_nodes": total_nodes,
            "previewed_instances": previewed_instances,
        },
        "task_definitions": definition_previews,
        "rounds": (
            dump_model(payload.campaign.rounds) if payload.campaign.rounds else None
        ),
        "runner_groups": [
            {
                "name": group.name,
                "phase": group.phase
                or (
                    payload.campaign.rounds.phases[0]
                    if payload.campaign.rounds
                    else "default"
                ),
                "matrix": group.matrix,
                "families": group.count,
                "cycles": (
                    payload.campaign.rounds.count if payload.campaign.rounds else 1
                ),
                "workers_per_runner": planned_worker_count(
                    group.runner.execution_benchmark()
                ),
            }
            for group in payload.runner_groups
        ],
    }


def _redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    redacted = copy.deepcopy(config)
    database = redacted.get("database")
    if isinstance(database, dict) and database.get("url"):
        try:
            database["url"] = make_url(database["url"]).render_as_string(
                hide_password=True
            )
        except Exception:
            database["url"] = "<redacted>"
    return redacted


def _snapshot_response(snapshot: ConfigSnapshot) -> Dict[str, Any]:
    return {
        "source": snapshot.source,
        "loaded_at": snapshot.loaded_at,
        "generation": snapshot.generation,
        "config": _redact_config(snapshot.config),
    }


def create_app(
    config_store: Optional[ConfigStore] = None,
    database: Optional[Database] = None,
    scheduler: Optional[Scheduler] = None,
    provider_registry: Optional[ProviderRegistry] = None,
    model_discovery: Optional[ProviderModelDiscovery] = None,
    tokenizer_cache: Optional[TokenizerResolver] = None,
    dataset_cache: Optional[DatasetResolver] = None,
    planner: Optional[Planner] = None,
) -> FastAPI:
    store = config_store or ConfigStore()
    validated_config = store.current()
    db = database or Database(validated_config.database)
    repository = RunnerRepository(
        db,
        log_directory=validated_config.scheduler.log_directory,
        log_bytes_limit=validated_config.scheduler.log_bytes_limit,
    )
    providers = provider_registry or ProviderRegistry.from_loader(
        load_provider_profiles
    )
    discovery = model_discovery or ProviderModelDiscovery(providers)
    if tokenizer_cache is None and dataset_cache is None:
        artifact_caches = ArtifactCaches.from_environment()
        tokenizers: TokenizerResolver = artifact_caches.tokenizer
        datasets: DatasetResolver = artifact_caches.dataset
    else:
        tokenizers = tokenizer_cache or TokenizerCache()
        datasets = dataset_cache or DatasetCache()
    token_verifier = TokenVerifier(validated_config.auth, repository)
    active_scheduler = scheduler or Scheduler(
        repository,
        validated_config.scheduler,
        validated_config.database,
        providers,
        tokenizers,
        datasets,
        validated_config.performance_guard,
    )
    active_planner = planner or Planner(repository, validated_config.planner)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if validated_config.database.auto_create_schema:
            await db.create_schema()
        try:
            schema_state = await db.schema_status()
            application.state.schema_state = schema_state
            if schema_state == "ready":
                await active_scheduler.start()
                await active_planner.start()
            else:
                LOGGER.error("Database %s; scheduling disabled", schema_state)
            yield
        finally:
            await active_planner.stop()
            await active_scheduler.stop()
            await db.dispose()

    application = FastAPI(
        title="LLMPerf Backend",
        description="Backend runtime configuration for LLMPerf experiments.",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    application.state.config_store = store
    application.state.database = db
    application.state.runner_repository = repository
    application.state.scheduler = active_scheduler
    application.state.planner = active_planner
    application.state.provider_registry = providers
    application.state.model_discovery = discovery
    application.state.tokenizer_cache = tokenizers
    application.state.dataset_cache = datasets

    async def require_schema(request: Request) -> None:
        path = request.url.path
        persistent_route = any(
            path.startswith(f"/api/v1/{name}")
            for name in (
                "runners",
                "runner-plans",
                "campaigns",
                "users",
                "trusted-clients",
            )
        )
        preflight = any(part in path for part in ("validate", "preview"))
        if (
            persistent_route
            and not preflight
            and getattr(application.state, "schema_state", "unchecked") != "ready"
        ):
            raise HTTPException(
                status_code=503,
                detail="schema_mismatch: required database tables or columns are missing",
            )

    api = APIRouter(
        prefix="/api/v1",
        dependencies=[Depends(token_verifier), Depends(require_schema)],
    )

    role_level = {"viewer": 10, "operator": 20, "superuser": 30}

    def require_role(request: Request, required: str) -> str:
        principal = getattr(request.state, "principal", {})
        current_role = str(principal.get("role", ""))
        if role_level.get(current_role, 0) < role_level[required]:
            raise HTTPException(
                status_code=403,
                detail=f"{required} access required",
            )
        return str(principal["sub"])

    async def resolve_benchmark(
        request: Request,
        benchmark: Dict[str, Any],
        *,
        tokenizer_selection: str,
        artifact_resolutions: Optional[
            Dict[Tuple[str, ...], ArtifactResolution]
        ] = None,
    ) -> Dict[str, Any]:
        try:
            resolved = request.app.state.provider_registry.resolve_benchmark(benchmark)
            tokenizer = resolved.get("tokenizer")
            if tokenizer is not None:
                resolution = await request.app.state.tokenizer_cache.resolve(tokenizer)
                if artifact_resolutions is not None:
                    tokenizer_key = (
                        "tokenizer",
                        resolution.tokenizer_id,
                        resolution.revision,
                        str(resolution.use_fast),
                    )
                    artifact_resolutions.setdefault(tokenizer_key, resolution)
                tokenizer_spec = resolution.benchmark_spec(
                    selection=tokenizer_selection,
                    accuracy=(
                        "compatible"
                        if tokenizer_selection == "explicit"
                        else "approximate"
                    ),
                    requested_revision=tokenizer.get("revision") or "main",
                )
                cache_probe = resolved.get("cache_probe")
                if (
                    cache_probe is not None
                    and tokenizer_spec["accuracy"] == "approximate"
                    and not cache_probe.get("allow_approximate_tokenizer", False)
                ):
                    raise TokenizerResolutionError(
                        "cache_probe requires an explicit tokenizer; set "
                        "allow_approximate_tokenizer=true to acknowledge the "
                        "global default"
                    )
                if cache_probe is not None and not tokenizer_spec["immutable_revision"]:
                    LOGGER.warning(
                        "Rejected cache_probe tokenizer: id=%s requested=%s "
                        "resolved=%s immutable=false",
                        tokenizer_spec["id"],
                        tokenizer_spec.get("requested_revision"),
                        tokenizer_spec["revision"],
                    )
                    raise TokenizerResolutionError(
                        "cache_probe tokenizer did not resolve to an immutable "
                        "Hugging Face commit revision: "
                        f"{tokenizer_spec['id']} requested "
                        f"{tokenizer_spec.get('requested_revision')!r}, resolved "
                        f"{tokenizer_spec['revision']!r}"
                    )
                LOGGER.info(
                    "Tokenizer requirement validated: id=%s requested=%s "
                    "resolved=%s immutable=%s cache_probe=%s",
                    tokenizer_spec["id"],
                    tokenizer_spec.get("requested_revision"),
                    tokenizer_spec["revision"],
                    tokenizer_spec["immutable_revision"],
                    resolved.get("cache_probe") is not None,
                )
                resolved["tokenizer"] = tokenizer_spec
            dataset = resolved.get("dataset")
            if dataset is not None:
                dataset_resolution = await request.app.state.dataset_cache.resolve(
                    dataset
                )
                if artifact_resolutions is not None:
                    dataset_key = (
                        "dataset",
                        dataset_resolution.dataset_id,
                        dataset_resolution.filename,
                        dataset_resolution.revision,
                    )
                    artifact_resolutions.setdefault(dataset_key, dataset_resolution)
                resolved["dataset"] = dataset_resolution.benchmark_spec()
            for input_definition in resolved.get("inputs", {}).values():
                input_dataset = input_definition.get("dataset")
                if input_dataset is not None:
                    input_resolution = await request.app.state.dataset_cache.resolve(
                        input_dataset
                    )
                    input_definition["dataset"] = input_resolution.benchmark_spec()
                    if artifact_resolutions is not None:
                        input_key = (
                            "dataset",
                            input_resolution.dataset_id,
                            input_resolution.filename,
                            input_resolution.revision,
                        )
                        artifact_resolutions.setdefault(input_key, input_resolution)
            return dump_model(ResolvedBenchmarkConfig.model_validate(resolved))
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_input=False, include_context=False),
            ) from exc
        except (
            DatasetResolutionError,
            ProviderConfigError,
            TokenizerResolutionError,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def resolve_runner(
        request: Request,
        runner: Any,
        default_benchmark: Dict[str, Any],
        artifact_resolutions: Optional[
            Dict[Tuple[str, ...], ArtifactResolution]
        ] = None,
    ) -> Dict[str, Any]:
        validated_runner = BenchmarkRunnerSpec.model_validate(runner)
        try:
            benchmark = dump_model(
                BenchmarkConfig.model_validate(
                    validated_runner.execution_benchmark(default_benchmark)
                )
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_input=False, include_context=False),
            ) from exc
        tokenizer_selection = (
            "explicit" if validated_runner.tokenizer is not None else "global_default"
        )
        return {
            "label": validated_runner.label,
            "metadata": validated_runner.metadata,
            "benchmark": await resolve_benchmark(
                request,
                benchmark,
                tokenizer_selection=tokenizer_selection,
                artifact_resolutions=artifact_resolutions,
            ),
        }

    async def resolve_plan(
        request: Request,
        runner_plan: RunnerPlanCreate,
        default_benchmark: Dict[str, Any],
        artifact_resolutions: Optional[
            Dict[Tuple[str, ...], ArtifactResolution]
        ] = None,
    ) -> Dict[str, Any]:
        return {
            "runtime": dump_model(runner_plan.runtime),
            "plan": {
                "name": runner_plan.name,
                "timezone": runner_plan.timezone,
                "starts_at": runner_plan.starts_at,
                "ends_at": runner_plan.ends_at,
                "max_occurrences": runner_plan.max_occurrences,
                "recurrence": dump_model(runner_plan.recurrence),
                "overlap_policy": runner_plan.overlap_policy,
                "misfire_grace_seconds": runner_plan.misfire_grace_seconds,
            },
            "runner_template": await resolve_runner(
                request,
                runner_plan.runner,
                default_benchmark,
                artifact_resolutions,
            ),
        }

    async def enforce_performance_guard(
        request: Request,
        runners: Any = (),
        runner_plans: Any = (),
        task_definitions: Any = (),
        runtime_concurrent_workers: Optional[int] = None,
        campaign_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        config = request.app.state.config_store.current()
        limits = (
            [runtime_concurrent_workers]
            if runtime_concurrent_workers is not None
            else []
        )
        if campaign_id is not None:
            repository = request.app.state.runner_repository
            campaign_document = await repository.get_campaign(campaign_id)
            if campaign_document is None:
                raise HTTPException(status_code=404, detail="Campaign not found")
            parent_runtime_id = campaign_document.get("runtime_id")
            if parent_runtime_id is not None:
                parent_runtime = await repository.get_submission_runtime(
                    parent_runtime_id
                )
                if parent_runtime is not None:
                    limits.extend(parent_runtime["limits"].values())
        effective_runtime_limit = min(limits) if limits else None
        try:
            assessment = await asyncio.to_thread(
                assess_workload,
                runners,
                runner_plans,
                task_definitions,
                config.performance_guard,
                request.app.state.scheduler.config.max_concurrent_runners,
                request.app.state.scheduler.config.request_actor_capacity,
                ray_actor_max_concurrency=(
                    request.app.state.scheduler.config.ray_actor_max_concurrency
                ),
                runtime_concurrent_workers=effective_runtime_limit,
            )
        except WorkloadSafetyError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "performance_safety": exc.assessment,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        LOGGER.info("Performance safety assessment: %s", assessment)
        return assessment

    async def resolve_campaign_payload(
        request: Request,
        payload: BenchmarkCampaignStart,
        artifact_resolutions: Optional[
            Dict[Tuple[str, ...], ArtifactResolution]
        ] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Resolve and validate one Campaign workload without persisting it."""

        default_benchmark = request.app.state.config_store.snapshot().config[
            "benchmark"
        ]
        resolved_runner_templates = [
            await resolve_runner(
                request,
                runner,
                default_benchmark,
                artifact_resolutions,
            )
            for runner in payload.runners
        ]
        from llmperf_backend.compiler.rounds import expand_rounds, expand_runner_group

        for group in payload.runner_groups:
            resolved_group_template = await resolve_runner(
                request, group.runner, default_benchmark, artifact_resolutions
            )
            resolved_runner_templates.extend(
                expand_runner_group(
                    resolved_group_template, group.name, group.matrix, group.phase
                )
            )

        resolved_runners = (
            expand_rounds(
                resolved_runner_templates,
                payload.campaign.rounds.count,
                payload.campaign.rounds.seed,
                payload.campaign.rounds.prefix_seed_scope,
                payload.campaign.rounds.phases,
            )
            if payload.campaign.rounds is not None
            else resolved_runner_templates
        )
        resolved_runner_plans = [
            await resolve_plan(
                request,
                runner_plan,
                default_benchmark,
                artifact_resolutions,
            )
            for runner_plan in payload.runner_plans
        ]
        resolved_task_definitions: List[Dict[str, Any]] = []
        for definition in payload.task_definitions:
            resolved_runner = await resolve_runner(
                request,
                definition.runner,
                default_benchmark,
                artifact_resolutions,
            )
            resolved_benchmark = resolved_runner["benchmark"]
            if resolved_benchmark.get("cache_probe") is not None:
                raise HTTPException(
                    status_code=422,
                    detail="compiled task runner cannot define cache_probe",
                )
            resolved_dataset = resolved_benchmark.get("dataset")
            if resolved_dataset is not None and not is_immutable_huggingface_revision(
                resolved_dataset.get("revision")
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "compiled task datasets must resolve to an immutable "
                        "Hugging Face commit revision"
                    ),
                )
            if int(resolved_benchmark.get("stddev_input_tokens", 0)) != 0:
                raise HTTPException(
                    status_code=422,
                    detail="compiled tasks require stddev_input_tokens=0",
                )
            if int(resolved_benchmark.get("stddev_output_tokens", 0)) != 0:
                raise HTTPException(
                    status_code=422,
                    detail="compiled tasks require stddev_output_tokens=0",
                )
            resolved_tokenizer = resolved_benchmark.get("tokenizer") or {}
            if not resolved_tokenizer.get("immutable_revision"):
                raise HTTPException(
                    status_code=422,
                    detail=("compiled tasks require an immutable tokenizer revision"),
                )
            resolved_task_definitions.append(
                {
                    "definition": dump_model(definition),
                    "runner_template": resolved_runner,
                }
            )
        await enforce_performance_guard(
            request,
            runners=resolved_runners,
            runner_plans=resolved_runner_plans,
            task_definitions=resolved_task_definitions,
            runtime_concurrent_workers=payload.runtime.concurrent_workers,
        )
        return (
            resolved_runners,
            resolved_runner_plans,
            resolved_task_definitions,
        )

    @application.get("/health", tags=["system"])
    async def health(request: Request) -> Dict[str, Any]:
        snapshot = request.app.state.config_store.snapshot()
        database_ok = await request.app.state.database.ping()
        schema_state = await request.app.state.database.schema_status()
        token_verifier.refresh()
        auth_status = token_verifier.status()
        return {
            "status": (
                "ok"
                if database_ok
                and schema_state == "ready"
                and not auth_status["reload_error"]
                else "degraded"
            ),
            "database": "connected" if database_ok else "unavailable",
            "database_schema": schema_state,
            "auth": auth_status,
            "providers": len(request.app.state.provider_registry.list_public()),
            "planner": request.app.state.planner.status()["status"],
            "config_source": snapshot.source,
            "config_generation": snapshot.generation,
        }

    @api.get("/scheduler/status", tags=["scheduler"])
    async def scheduler_status(request: Request) -> Dict[str, Any]:
        require_role(request, "viewer")
        return request.app.state.scheduler.status()

    @api.get("/planner/status", tags=["planner"])
    async def planner_status(request: Request) -> Dict[str, Any]:
        require_role(request, "viewer")
        return request.app.state.planner.status()

    @api.get("/config", tags=["configuration"])
    def get_config(request: Request) -> Dict[str, Any]:
        return _snapshot_response(request.app.state.config_store.snapshot())

    @api.get("/config/schema", tags=["configuration"])
    def get_config_schema() -> Dict[str, Any]:
        return app_config_schema()

    @api.post("/config/validate", tags=["configuration"])
    def validate_config(payload: YAMLValidationRequest) -> Dict[str, Any]:
        from llmperf_backend.configuration.loader import load_config_text

        try:
            config = load_config_text(payload.yaml_content)
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"valid": True, "config": _redact_config(dump_model(config))}

    @api.post("/config/reload", tags=["configuration"])
    def reload_config(request: Request) -> Dict[str, Any]:
        require_role(request, "superuser")
        try:
            snapshot = request.app.state.config_store.reload()
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response = _snapshot_response(snapshot)
        response["note"] = (
            "Benchmark defaults apply to new Runners immediately; database, Scheduler, and "
            "server changes require a backend restart."
        )
        return response

    @api.post(
        "/runners",
        tags=["runners"],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_runner(
        request: Request, payload: BenchmarkRunnerCreate
    ) -> Dict[str, Any]:
        actor = require_role(request, "operator")
        resolved_runner = await resolve_runner(
            request,
            payload,
            request.app.state.config_store.snapshot().config["benchmark"],
        )
        resolved_benchmark = resolved_runner["benchmark"]
        await enforce_performance_guard(
            request,
            runners=[resolved_runner],
            runtime_concurrent_workers=payload.runtime.concurrent_workers,
            campaign_id=payload.campaign_id,
        )
        runner = await request.app.state.runner_repository.create_runner(
            resolved_benchmark,
            payload.metadata,
            actor,
            campaign_id=payload.campaign_id,
            label=payload.label,
            runtime=payload.runtime,
        )
        if runner is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return runner

    @api.post(
        "/campaigns",
        tags=["campaigns"],
        status_code=status.HTTP_201_CREATED,
    )
    async def create_campaign(
        request: Request, payload: BenchmarkCampaignCreate
    ) -> Dict[str, Any]:
        actor = require_role(request, "operator")
        if payload.rounds is not None:
            raise HTTPException(
                status_code=422, detail="Use /campaigns/start to submit Runner rounds"
            )
        return await request.app.state.runner_repository.create_campaign(
            payload.name,
            payload.description,
            payload.tags,
            actor,
            runtime=payload.runtime,
        )

    async def validate_campaign_artifact_document(
        request: Request,
        payload: BenchmarkCampaignStart,
        progress_callback: Optional[ArtifactProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Materialize and verify Campaign artifacts without persistence."""

        artifact_resolutions: Dict[Tuple[str, ...], ArtifactResolution] = {}
        if progress_callback is None:
            resolved_payload = await resolve_campaign_payload(
                request, payload, artifact_resolutions
            )
        else:
            with artifact_progress_scope(progress_callback):
                resolved_payload = await resolve_campaign_payload(
                    request, payload, artifact_resolutions
                )
        resolved_runners, resolved_runner_plans, resolved_task_definitions = (
            resolved_payload
        )
        ordered_resolutions = tuple(
            resolution for _, resolution in sorted(artifact_resolutions.items())
        )
        event_loop = asyncio.get_running_loop()
        try:
            artifact_evidence = await event_loop.run_in_executor(
                None, validate_resolved_artifacts, ordered_resolutions
            )
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        artifacts = [evidence.public_dict() for evidence in artifact_evidence]
        task_estimates = []
        for task_definition in resolved_task_definitions:
            definition_document = task_definition["definition"]
            recipe_document = {
                key: value
                for key, value in definition_document.items()
                if key != "runner"
            }
            task_estimates.append(
                TaskCompiler.estimate(
                    recipe_document,
                    benchmark=task_definition["runner_template"]["benchmark"],
                )
            )
        LOGGER.info(
            "Campaign artifact validation completed: campaign=%s artifacts=%d "
            "bytes=%d",
            payload.campaign.name,
            len(artifacts),
            sum(int(artifact["size_bytes"]) for artifact in artifacts),
        )
        return {
            "version": PROTOCOL_VERSION,
            "valid": True,
            "campaign": payload.campaign.name,
            "workload": {
                "immediate_runners": len(resolved_runners),
                "runner_plans": len(resolved_runner_plans),
                "task_definitions": len(resolved_task_definitions),
                "task_instances": sum(
                    estimate["instances"] for estimate in task_estimates
                ),
                "task_nodes": sum(estimate["nodes"] for estimate in task_estimates),
            },
            "artifacts": artifacts,
        }

    @api.post(
        "/campaigns/validate-artifacts",
        tags=["campaigns", "artifacts"],
    )
    async def validate_campaign_artifacts(
        request: Request, payload: BenchmarkCampaignStart
    ) -> Dict[str, Any]:
        """Materialize and verify Campaign artifacts without persistence."""

        require_role(request, "operator")
        return await validate_campaign_artifact_document(request, payload)

    @api.post(
        "/campaigns/validate-artifacts/stream",
        tags=["campaigns", "artifacts"],
    )
    async def stream_campaign_artifacts(
        request: Request, payload: BenchmarkCampaignStart
    ) -> StreamingResponse:
        """Stream path-free transfer progress and final validation evidence."""

        require_role(request, "operator")
        event_loop = asyncio.get_running_loop()
        progress_events: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        def publish_progress(progress: ArtifactDownloadProgress) -> None:
            event_loop.call_soon_threadsafe(
                progress_events.put_nowait,
                {"event": "progress", "progress": progress.public_dict()},
            )

        async def run_validation() -> None:
            try:
                validation = await validate_campaign_artifact_document(
                    request,
                    payload,
                    publish_progress,
                )
                await progress_events.put({"event": "result", "result": validation})
            except HTTPException as exc:
                await progress_events.put(
                    {
                        "event": "error",
                        "status_code": exc.status_code,
                        "detail": exc.detail,
                    }
                )
            except Exception:
                LOGGER.exception("Streaming Campaign artifact validation failed")
                await progress_events.put(
                    {
                        "event": "error",
                        "status_code": 500,
                        "detail": "Internal Server Error",
                    }
                )

        async def event_stream() -> AsyncIterator[str]:
            validation_task = asyncio.create_task(run_validation())
            while True:
                try:
                    event = await asyncio.wait_for(progress_events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    event = {"event": "heartbeat"}
                yield json.dumps(event, separators=(",", ":")) + "\n"
                if event["event"] in {"result", "error"}:
                    await validation_task
                    return

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
        )

    @api.post("/campaigns/preview", tags=["campaigns", "tasks"])
    async def preview_campaign_tasks(
        request: Request,
        payload: BenchmarkCampaignStart,
        limit: int = Query(default=20, ge=1, le=200),
        node_limit: int = Query(default=100, ge=1, le=1_000),
        debug: bool = Query(default=False),
    ) -> Dict[str, Any]:
        """Validate and expand TaskDefinitions without persistence or artifacts."""

        require_role(request, "viewer")
        try:
            return _compile_campaign_task_preview(payload, limit, node_limit, debug)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.post(
        "/campaigns/start",
        tags=["campaigns", "runners", "planner"],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_campaign(
        request: Request, payload: BenchmarkCampaignStart
    ) -> Dict[str, Any]:
        """Validate first, then atomically distribute one Campaign workload."""

        actor = require_role(request, "operator")
        runners, runner_plans, task_definitions = await resolve_campaign_payload(
            request,
            payload,
        )
        workload = await request.app.state.runner_repository.create_campaign_workload(
            payload.campaign.name,
            payload.campaign.description,
            payload.campaign.tags,
            runners,
            runner_plans,
            task_definitions,
            actor,
            runtime=payload.runtime,
        )
        campaign_id = workload["campaign"]["campaign_id"]
        workload_summary = workload["summary"]
        LOGGER.info(
            "Campaign %s workload accepted: immediate_runners=%d runner_plans=%d "
            "task_definitions=%d task_instances=%d task_nodes=%d",
            campaign_id,
            workload_summary["immediate_runners"],
            workload_summary["runner_plans"],
            workload_summary["task_definitions"],
            workload_summary["task_instances"],
            workload_summary["task_nodes"],
        )
        for runner_plan in workload["runner_plans"]:
            LOGGER.info(
                "RunnerPlan %s registered in Campaign %s: next_fire_at=%s",
                runner_plan["runner_plan_id"],
                campaign_id,
                runner_plan["next_fire_at"],
            )
        return workload

    @api.post(
        "/campaigns/{campaign_id}/runners",
        tags=["campaigns", "runners"],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_campaign_runners(
        request: Request, campaign_id: str, payload: BenchmarkRunnerBatchCreate
    ) -> Dict[str, Any]:
        actor = require_role(request, "operator")
        default_benchmark = request.app.state.config_store.snapshot().config[
            "benchmark"
        ]
        runners = [
            await resolve_runner(request, runner, default_benchmark)
            for runner in payload.runners
        ]
        await enforce_performance_guard(
            request,
            runners=runners,
            runtime_concurrent_workers=payload.runtime.concurrent_workers,
            campaign_id=campaign_id,
        )
        created = await request.app.state.runner_repository.create_runners(
            campaign_id, runners, actor, runtime=payload.runtime
        )
        if created is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return {
            "campaign_id": campaign_id,
            "runtime_id": created[0]["runtime_id"],
            "items": created,
        }

    @api.post("/runner-plans/preview", tags=["planner"])
    async def preview_runner_plan(
        request: Request, payload: RunnerPlanPreview
    ) -> Dict[str, Any]:
        require_role(request, "viewer")
        timing = {
            "timezone": payload.timezone,
            "starts_at": payload.starts_at,
            "ends_at": payload.ends_at,
            "max_occurrences": payload.max_occurrences,
            "recurrence": dump_model(payload.recurrence),
        }
        effective_starts_at = payload.starts_at or datetime.now(timezone.utc)
        return {
            "start_mode": "explicit" if payload.starts_at else "immediate",
            "effective_starts_at": effective_starts_at,
            "items": preview_fires(
                timing,
                payload.count,
                default_starts_at=effective_starts_at,
            ),
        }

    @api.post(
        "/campaigns/{campaign_id}/runner-plans",
        tags=["planner"],
        status_code=status.HTTP_201_CREATED,
    )
    async def create_runner_plan(
        request: Request, campaign_id: str, payload: RunnerPlanCreate
    ) -> Dict[str, Any]:
        actor = require_role(request, "operator")
        default_benchmark = request.app.state.config_store.snapshot().config[
            "benchmark"
        ]
        resolved_plan = await resolve_plan(request, payload, default_benchmark)
        await enforce_performance_guard(
            request,
            runner_plans=[resolved_plan],
            runtime_concurrent_workers=payload.runtime.concurrent_workers,
            campaign_id=campaign_id,
        )
        plan = await request.app.state.runner_repository.create_runner_plan(
            campaign_id,
            resolved_plan["plan"],
            resolved_plan["runner_template"],
            actor,
            runtime=payload.runtime,
        )
        if plan is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        LOGGER.info(
            "RunnerPlan %s registered in Campaign %s: next_fire_at=%s",
            plan["runner_plan_id"],
            campaign_id,
            plan["next_fire_at"],
        )
        return plan

    @api.get("/runner-plans", tags=["planner"])
    async def list_runner_plans(
        request: Request,
        plan_status: Optional[str] = Query(default=None, alias="status"),
        campaign_id: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        require_role(request, "viewer")
        if plan_status is not None and plan_status not in PLAN_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"status must be one of {sorted(PLAN_STATUSES)}",
            )
        items = await request.app.state.runner_repository.list_runner_plans(
            plan_status, campaign_id, limit, offset
        )
        return {"items": items, "limit": limit, "offset": offset}

    @api.get("/runner-plans/{runner_plan_id}", tags=["planner"])
    async def get_runner_plan(request: Request, runner_plan_id: str) -> Dict[str, Any]:
        require_role(request, "viewer")
        plan = await request.app.state.runner_repository.get_runner_plan(runner_plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="RunnerPlan not found")
        return plan

    @api.get("/runner-plans/{runner_plan_id}/events", tags=["planner"])
    async def get_plan_events(request: Request, runner_plan_id: str) -> Dict[str, Any]:
        require_role(request, "viewer")
        events = await request.app.state.runner_repository.get_plan_events(
            runner_plan_id
        )
        if events is None:
            raise HTTPException(status_code=404, detail="RunnerPlan not found")
        return {"runner_plan_id": runner_plan_id, "items": events}

    async def change_plan(
        request: Request, runner_plan_id: str, action: str
    ) -> Dict[str, Any]:
        require_role(request, "operator")
        plan = await request.app.state.runner_repository.change_runner_plan(
            runner_plan_id, action
        )
        if plan is None:
            raise HTTPException(status_code=404, detail="RunnerPlan not found")
        error = plan.pop("transition_error", None)
        if error:
            raise HTTPException(status_code=409, detail=error)
        return plan

    @api.post("/runner-plans/{runner_plan_id}/pause", tags=["planner"])
    async def pause_runner_plan(
        request: Request, runner_plan_id: str
    ) -> Dict[str, Any]:
        return await change_plan(request, runner_plan_id, "pause")

    @api.post("/runner-plans/{runner_plan_id}/resume", tags=["planner"])
    async def resume_runner_plan(
        request: Request, runner_plan_id: str
    ) -> Dict[str, Any]:
        return await change_plan(request, runner_plan_id, "resume")

    @api.post("/runner-plans/{runner_plan_id}/cancel", tags=["planner"])
    async def cancel_runner_plan(
        request: Request, runner_plan_id: str
    ) -> Dict[str, Any]:
        return await change_plan(request, runner_plan_id, "cancel")

    @api.get("/providers", tags=["providers"])
    async def list_providers(request: Request) -> Dict[str, Any]:
        require_role(request, "viewer")
        return request.app.state.provider_registry.public_document()

    @api.post("/providers/reload", tags=["providers"])
    async def reload_providers(request: Request) -> Dict[str, Any]:
        """Atomically reload only Provider Profiles from the Backend env file."""

        require_role(request, "operator")
        try:
            document = request.app.state.provider_registry.reload()
            request.app.state.model_discovery.invalidate()
            return document
        except ProviderConfigError as exc:
            # Candidate validation happens before the active registry is changed.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.get("/providers/{provider_id}/models", tags=["providers"])
    async def list_provider_models(
        request: Request,
        provider_id: str,
        refresh: bool = Query(default=False),
    ) -> Dict[str, Any]:
        require_role(request, "operator" if refresh else "viewer")
        try:
            return await request.app.state.model_discovery.models(
                provider_id, refresh=refresh
            )
        except ProviderConfigError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ProviderDiscoveryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @api.get("/runtimes/{runtime_id}", tags=["runtime"])
    async def get_submission_runtime(
        request: Request, runtime_id: str
    ) -> Dict[str, Any]:
        require_role(request, "viewer")
        runtime_document = (
            await request.app.state.runner_repository.get_submission_runtime(runtime_id)
        )
        if runtime_document is None:
            raise HTTPException(status_code=404, detail="Submission runtime not found")
        return runtime_document

    @api.get("/campaigns", tags=["campaigns"])
    async def list_campaigns(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        items = await request.app.state.runner_repository.list_campaigns(limit, offset)
        await attach_progress(
            items,
            request.app.state.runner_repository,
            request.app.state.scheduler,
            campaigns=True,
        )
        return {"items": items, "limit": limit, "offset": offset}

    @api.get("/campaigns/{campaign_id}", tags=["campaigns"])
    async def get_campaign(request: Request, campaign_id: str) -> Dict[str, Any]:
        campaign = await request.app.state.runner_repository.get_campaign_status(
            campaign_id
        )
        if campaign is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        await attach_progress(
            [campaign],
            request.app.state.runner_repository,
            request.app.state.scheduler,
            campaigns=True,
        )
        return campaign

    @api.post("/campaigns/{campaign_id}/cancel", tags=["campaigns"])
    async def cancel_campaign(request: Request, campaign_id: str) -> Dict[str, Any]:
        require_role(request, "operator")
        campaign = await request.app.state.runner_repository.request_cancel_campaign(
            campaign_id
        )
        if campaign is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign

    @api.get("/campaigns/{campaign_id}/export", tags=["results"])
    async def export_campaign(
        request: Request,
        campaign_id: str,
        include_requests: bool = Query(default=False),
    ) -> JSONResponse:
        document = await request.app.state.runner_repository.export_campaign(
            campaign_id, include_requests
        )
        if document is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return JSONResponse(
            content=json_safe(document),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="llmperf-campaign-{campaign_id}.json"'
                )
            },
        )

    @api.get("/runners", tags=["runners"])
    async def list_runners(
        request: Request,
        runner_status: Optional[str] = Query(default=None, alias="status"),
        campaign_id: Optional[str] = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        full: bool = Query(default=False),
    ) -> Dict[str, Any]:
        if runner_status is not None and runner_status not in RUNNER_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"status must be one of {sorted(RUNNER_STATUSES)}",
            )
        runners = await request.app.state.runner_repository.list_runners(
            runner_status,
            limit,
            offset,
            full=full,
            campaign_id=campaign_id,
        )
        await attach_progress(
            runners,
            request.app.state.runner_repository,
            request.app.state.scheduler,
            campaigns=False,
        )
        return {
            "items": runners,
            "limit": limit,
            "offset": offset,
            "full": full,
            "campaign_id": campaign_id,
        }

    @api.get("/runners/{runner_id}", tags=["runners"])
    async def get_runner(request: Request, runner_id: str) -> Dict[str, Any]:
        runner = await request.app.state.runner_repository.get_runner(runner_id)
        if runner is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        await attach_progress(
            [runner],
            request.app.state.runner_repository,
            request.app.state.scheduler,
            campaigns=False,
        )
        return runner

    @api.get("/runners/{runner_id}/logs", tags=["runners"])
    async def get_runner_logs(request: Request, runner_id: str) -> Dict[str, Any]:
        """Return only the persisted Worker identity and captured output."""

        runner = await request.app.state.runner_repository.get_runner(runner_id)
        if runner is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        try:
            logs = await request.app.state.runner_repository.read_logs(
                runner.get("log_key")
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail="Runner log file unavailable"
            ) from exc
        return {
            "runner_id": runner["runner_id"],
            "status": runner["status"],
            "scheduler_id": runner.get("scheduler_id"),
            "worker": runner.get("worker"),
            **logs,
        }

    @api.post("/runners/{runner_id}/cancel", tags=["runners"])
    async def cancel_runner(request: Request, runner_id: str) -> Dict[str, Any]:
        require_role(request, "operator")
        runner = await request.app.state.runner_repository.request_cancel(runner_id)
        if runner is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        return runner

    @api.get("/runners/{runner_id}/results", tags=["results"])
    async def get_runner_results(
        request: Request,
        runner_id: str,
        include_requests: bool = Query(default=False),
    ) -> Dict[str, Any]:
        results = await request.app.state.runner_repository.get_results(
            runner_id, include_requests
        )
        if results is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        return results

    @api.get("/runners/{runner_id}/export", tags=["results"])
    async def export_runner(request: Request, runner_id: str) -> JSONResponse:
        runner = await request.app.state.runner_repository.get_runner(runner_id)
        if runner is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        if runner["status"] not in {SUCCEEDED, FAILED} or runner["summary"] is None:
            raise HTTPException(
                status_code=409,
                detail="Only Runners with persisted benchmark results can be exported",
            )
        results = await request.app.state.runner_repository.get_results(runner_id, True)
        document = {
            "version": PROTOCOL_VERSION,
            "runner": {
                "runner_id": runner["runner_id"],
                "campaign_id": runner["campaign_id"],
                "label": runner["label"],
                "status": runner["status"],
                "error_message": runner["error_message"],
                "benchmark": runner["benchmark"],
                "metadata": runner["metadata"],
                "created_at": runner["created_at"],
                "started_at": runner["started_at"],
                "finished_at": runner["finished_at"],
            },
            "results": results,
        }
        return JSONResponse(
            content=json_safe(document),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="llmperf-runner-{runner_id}.json"'
                )
            },
        )

    @api.get("/runners/{runner_id}/events", tags=["runners"])
    async def get_runner_events(request: Request, runner_id: str) -> Dict[str, Any]:
        events = await request.app.state.runner_repository.get_events(runner_id)
        if events is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        return {"runner_id": runner_id, "items": events}

    @api.get("/admin/trusted-clients", tags=["administration"])
    async def list_trusted_clients(request: Request) -> Dict[str, Any]:
        require_role(request, "superuser")
        items = await request.app.state.runner_repository.list_trusted_clients()
        return {"items": items}

    @api.put("/admin/trusted-clients/{username}", tags=["administration"])
    async def write_trusted_client(
        request: Request, username: str, payload: TrustedClientWrite
    ) -> Dict[str, Any]:
        actor = require_role(request, "superuser")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", username) or username in {
            ".",
            "..",
        }:
            raise HTTPException(status_code=422, detail="Invalid trusted username")
        if username == validated_config.auth.bootstrap_subject:
            raise HTTPException(
                status_code=409,
                detail="Rotate the bootstrap key through its configured PEM file",
            )
        try:
            key_id, normalized_pem = normalize_public_key(payload.public_key)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        client = await request.app.state.runner_repository.upsert_trusted_client(
            username,
            key_id,
            normalized_pem,
            payload.role,
            payload.display_name,
            payload.email,
            actor,
            validated_config.auth.previous_key_grace_seconds,
        )
        if client is None:
            raise HTTPException(
                status_code=409,
                detail="This public key is already assigned to another user",
            )
        return client

    @api.delete("/admin/trusted-clients/{username}", tags=["administration"])
    async def revoke_trusted_client(request: Request, username: str) -> Dict[str, Any]:
        actor = require_role(request, "superuser")
        if username == validated_config.auth.bootstrap_subject:
            raise HTTPException(
                status_code=409, detail="Bootstrap user cannot be revoked"
            )
        client = await request.app.state.runner_repository.revoke_trusted_client(
            username, actor
        )
        if client is None:
            raise HTTPException(status_code=404, detail="Trusted client not found")
        return client

    @api.get("/admin/trusted-client-events", tags=["administration"])
    async def list_trusted_client_events(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        require_role(request, "superuser")
        items = await request.app.state.runner_repository.list_trusted_client_events(
            limit
        )
        return {"items": items}

    application.include_router(api)
    return application

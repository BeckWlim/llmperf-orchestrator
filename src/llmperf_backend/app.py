"""FastAPI application for configuration and durable benchmark orchestration."""

from contextlib import asynccontextmanager
import copy
from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.engine import make_url

from llmperf_backend.auth import TokenVerifier, normalize_public_key
from llmperf_backend.config import ConfigError, ConfigStore, ConfigSnapshot
from llmperf_backend.environment import load_provider_environment
from llmperf_backend.models import (
    BenchmarkCampaignCreate,
    BenchmarkCampaignStart,
    BenchmarkRunnerBatchCreate,
    BenchmarkRunnerCreate,
    RunnerPlanCreate,
    RunnerPlanPreview,
    TrustedClientWrite,
    YAMLValidationRequest,
    app_config_schema,
    dump_model,
)
from llmperf_backend.persistence import (
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
from llmperf_backend.planner import Planner, preview_fires
from llmperf_backend.providers import (
    ProviderConfigError,
    ProviderDiscoveryError,
    ProviderModelDiscovery,
    ProviderRegistry,
)
from llmperf_backend.scheduler import Scheduler
from llmperf_backend.safety import WorkloadSafetyError, assess_workload
from llmperf_backend.tokenizers import TokenizerCache, TokenizerResolutionError
from llmperf_backend.datasets import DatasetCache, DatasetResolutionError


RUNNER_STATUSES = {QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED}
LOGGER = logging.getLogger(__name__)


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
    tokenizer_cache: Optional[TokenizerCache] = None,
    dataset_cache: Optional[DatasetCache] = None,
    planner: Optional[Planner] = None,
) -> FastAPI:
    store = config_store or ConfigStore()
    validated_config = store.current()
    db = database or Database(validated_config.database)
    repository = RunnerRepository(db)
    providers = provider_registry or ProviderRegistry.from_environment(
        load_provider_environment(), reload_loader=load_provider_environment
    )
    discovery = model_discovery or ProviderModelDiscovery(providers)
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
            await active_planner.start()
            await active_scheduler.start()
            yield
        finally:
            await active_planner.stop()
            await active_scheduler.stop()
            await db.dispose()

    application = FastAPI(
        title="LLMPerf Backend",
        description="Backend runtime configuration for LLMPerf experiments.",
        version="0.1.0",
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
    api = APIRouter(
        prefix="/api/v1",
        dependencies=[Depends(token_verifier)],
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
        request: Request, benchmark: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            resolved = request.app.state.provider_registry.resolve_benchmark(benchmark)
            tokenizer = resolved.get("tokenizer")
            if tokenizer is not None:
                resolution = await request.app.state.tokenizer_cache.resolve(tokenizer)
                tokenizer_spec = resolution.benchmark_spec(
                    selection=tokenizer.get("selection", "global_default"),
                    accuracy=tokenizer.get("accuracy", "approximate"),
                    requested_revision=(
                        tokenizer.get("requested_revision")
                        or tokenizer.get("revision")
                        or "main"
                    ),
                )
                if (
                    resolved.get("cache_probe") is not None
                    and not tokenizer_spec["immutable_revision"]
                ):
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
                resolved["dataset"] = dataset_resolution.benchmark_spec()
            return resolved
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
    ) -> Dict[str, Any]:
        benchmark = (
            default_benchmark
            if runner.benchmark is None
            else dump_model(runner.benchmark)
        )
        return {
            "label": runner.label,
            "metadata": runner.metadata,
            "benchmark": await resolve_benchmark(request, benchmark),
        }

    async def resolve_plan(
        request: Request,
        runner_plan: RunnerPlanCreate,
        default_benchmark: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
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
                request, runner_plan.runner, default_benchmark
            ),
        }

    def enforce_performance_guard(
        request: Request,
        runners: Any = (),
        runner_plans: Any = (),
        protocol_definitions: Any = (),
    ) -> Dict[str, Any]:
        config = request.app.state.config_store.current()
        try:
            assessment = assess_workload(
                runners,
                runner_plans,
                protocol_definitions,
                config.performance_guard,
                request.app.state.scheduler.config.max_concurrent_runners,
                int(
                    request.app.state.scheduler.config.ray_num_cpus
                    / request.app.state.scheduler.config.ray_actor_num_cpus
                ),
            )
        except WorkloadSafetyError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "performance_safety": exc.assessment,
                },
            ) from exc
        LOGGER.info("Performance safety assessment: %s", assessment)
        return assessment

    @application.get("/health", tags=["system"])
    async def health(request: Request) -> Dict[str, Any]:
        snapshot = request.app.state.config_store.snapshot()
        database_ok = await request.app.state.database.ping()
        token_verifier.refresh()
        auth_status = token_verifier.status()
        return {
            "status": (
                "ok" if database_ok and not auth_status["reload_error"] else "degraded"
            ),
            "database": "connected" if database_ok else "unavailable",
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
        from llmperf_backend.config import load_config_text

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
        if payload.benchmark is None:
            benchmark = request.app.state.config_store.snapshot().config["benchmark"]
        else:
            benchmark = dump_model(payload.benchmark)
        benchmark = await resolve_benchmark(request, benchmark)
        enforce_performance_guard(request, runners=[{"benchmark": benchmark}])
        runner = await request.app.state.runner_repository.create_runner(
            benchmark,
            payload.metadata,
            actor,
            campaign_id=payload.campaign_id,
            label=payload.label,
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
        return await request.app.state.runner_repository.create_campaign(
            payload.name, payload.description, payload.tags, actor
        )

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
        default_benchmark = request.app.state.config_store.snapshot().config[
            "benchmark"
        ]
        runners = [
            await resolve_runner(request, runner, default_benchmark)
            for runner in payload.runners
        ]
        runner_plans = [
            await resolve_plan(request, runner_plan, default_benchmark)
            for runner_plan in payload.runner_plans
        ]
        protocol_definitions = []
        for definition in payload.protocol_definitions:
            runner = await resolve_runner(request, definition.runner, default_benchmark)
            benchmark = runner["benchmark"]
            if benchmark.get("cache_probe") is not None:
                raise HTTPException(
                    status_code=422,
                    detail="cache protocol runner cannot define cache_probe",
                )
            if benchmark.get("protocol_request") is not None:
                raise HTTPException(
                    status_code=422,
                    detail="protocol_request is backend-owned and cannot be submitted",
                )
            if benchmark.get("dataset") is not None:
                raise HTTPException(
                    status_code=422,
                    detail="cache protocols require generated deterministic prompts",
                )
            if benchmark.get("llm_api") != "openai":
                raise HTTPException(
                    status_code=422,
                    detail="cache protocols currently require llm_api=openai",
                )
            if int(benchmark.get("stddev_input_tokens", 0)) != 0:
                raise HTTPException(
                    status_code=422,
                    detail="cache protocols require stddev_input_tokens=0",
                )
            if int(benchmark.get("stddev_output_tokens", 0)) != 0:
                raise HTTPException(
                    status_code=422,
                    detail="cache protocols require stddev_output_tokens=0",
                )
            tokenizer = benchmark.get("tokenizer") or {}
            if (
                not tokenizer.get("immutable_revision")
                or tokenizer.get("accuracy") == "approximate"
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "cache protocol requires an explicit/model-bound "
                        "tokenizer at an immutable Hugging Face revision"
                    ),
                )
            protocol_definitions.append(
                {
                    "definition": dump_model(definition),
                    "runner_template": runner,
                }
            )
        enforce_performance_guard(
            request,
            runners=runners,
            runner_plans=runner_plans,
            protocol_definitions=protocol_definitions,
        )
        workload = await request.app.state.runner_repository.create_campaign_workload(
            payload.campaign.name,
            payload.campaign.description,
            payload.campaign.tags,
            runners,
            runner_plans,
            protocol_definitions,
            actor,
        )
        campaign_id = workload["campaign"]["campaign_id"]
        LOGGER.info(
            "Campaign %s workload accepted: runners=%d runner_plans=%d protocol_definitions=%d",
            campaign_id,
            len(workload["items"]),
            len(workload["runner_plans"]),
            len(workload["protocol_definitions"]),
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
        enforce_performance_guard(request, runners=runners)
        created = await request.app.state.runner_repository.create_runners(
            campaign_id, runners, actor
        )
        if created is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return {"campaign_id": campaign_id, "items": created}

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
        enforce_performance_guard(request, runner_plans=[resolved_plan])
        plan = await request.app.state.runner_repository.create_runner_plan(
            campaign_id,
            resolved_plan["plan"],
            resolved_plan["runner_template"],
            actor,
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

    @api.get("/campaigns", tags=["campaigns"])
    async def list_campaigns(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        items = await request.app.state.runner_repository.list_campaigns(limit, offset)
        return {"items": items, "limit": limit, "offset": offset}

    @api.get("/campaigns/{campaign_id}", tags=["campaigns"])
    async def get_campaign(request: Request, campaign_id: str) -> Dict[str, Any]:
        campaign = await request.app.state.runner_repository.get_campaign_status(
            campaign_id
        )
        if campaign is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
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
        return runner

    @api.get("/runners/{runner_id}/logs", tags=["runners"])
    async def get_runner_logs(request: Request, runner_id: str) -> Dict[str, Any]:
        """Return only the persisted Worker identity and captured output."""

        runner = await request.app.state.runner_repository.get_runner(runner_id)
        if runner is None:
            raise HTTPException(status_code=404, detail="Runner not found")
        return {
            "runner_id": runner["runner_id"],
            "status": runner["status"],
            "scheduler_id": runner.get("scheduler_id"),
            "worker": runner.get("worker"),
            "stdout": runner.get("stdout"),
            "stderr": runner.get("stderr"),
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
            "version": 1,
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

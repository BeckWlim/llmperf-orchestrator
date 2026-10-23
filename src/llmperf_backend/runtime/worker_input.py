"""Lightweight Worker recipes and slot-bound, actor-local prompt generation."""

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import random
import threading
import time
from typing import Any, Dict, Mapping, Sequence

from llmperf.benchmark.execution import RequestStep
from llmperf.clients.models import RequestConfig, WorkerInput
from llmperf.configuration.duration import duration_seconds
from llmperf.datasets.preparation import (
    CorpusCalibration,
    PromptPlan,
    calibrate,
    calibrate_corpus,
    render_chunk,
)
from llmperf.datasets.prompts import PromptDatasetSource, prepare_prompt_requests
from llmperf.datasets.probes import build_cache_probe_plan
from llmperf.datasets.tokenizer import TOKENIZER_PATH, TOKENIZER_FAST
from llmperf_backend.compiler.requests import request_plan
from llmperf_backend.runtime.worker import (
    INPUT_DATASET_PATHS,
    WORKER_DATASET_PATH,
    _prepared_tokenizer,
)


def indexed_seed(seed: int, namespace: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{namespace}".encode()).digest()[:8], "big"
    )


def worker_steps(
    benchmark: Mapping[str, Any], calibrated: bool = True
) -> Sequence[RequestStep]:
    """Expand descriptions only: no tokenizer, corpus reads, or prompt allocation."""
    probe = benchmark.get("cache_probe")
    plan = request_plan(benchmark)
    specifications = plan.expand() if not probe else []
    probe_coordinates: list[tuple[int, int]] = []
    if probe:
        families = list(range(int(probe["trials"])))
        if (
            probe.get("schedule", "randomized_family_blocks")
            == "randomized_family_blocks"
        ):
            random.Random(benchmark.get("dataset_seed", 11111)).shuffle(families)
        probe_coordinates = [
            (family, occurrence)
            for family in families
            for occurrence in range(int(probe.get("repeats_after_prime", 1)) + 1)
        ]
    contexts = benchmark.get("task_contexts") or []
    named_counts: dict[str, int] = {}
    steps: list[RequestStep] = []
    count = len(probe_coordinates) if probe else len(specifications)
    for index in range(count):
        specification = specifications[index] if not probe else None
        input_name = (specification.input if specification else None) or plan.input
        definition = (benchmark.get("inputs") or {}).get(input_name, {})
        tokens = definition.get("tokens", {})
        mean_tokens = (
            int(tokens.get("mean", benchmark["mean_input_tokens"]))
            if isinstance(tokens, dict)
            else int(tokens)
        )
        token_stddev = (
            int(tokens.get("stddev", benchmark.get("stddev_input_tokens", 0)))
            if isinstance(tokens, dict)
            else 0
        )
        context = contexts[index] if contexts else benchmark.get("task_context") or {}
        atomic = bool(context) and context.get("execution_kind", "atomic") == "atomic"
        base_seed = (
            int(context["payload_seed"])
            if atomic
            else int(benchmark.get("dataset_seed", 11111))
        )
        base_prefix_seed = benchmark.get("shared_prefix_seed")
        # Ordinary/default inputs retain the calibrated family seed contract.
        if specification and input_name != plan.input:
            seed_material = (
                f"{base_seed}:{definition.get('dataset_seed', 11111)}:{input_name}"
            )
            seed = int.from_bytes(
                hashlib.sha256(seed_material.encode()).digest()[:8], "big"
            )
            prefix_seed = (
                indexed_seed(base_prefix_seed, input_name)
                if base_prefix_seed is not None
                and context.get("prefix_seed_version") != "explicit-prefix-v1"
                else base_prefix_seed
            )
        else:
            seed = base_seed
            prefix_seed = base_prefix_seed
        named_index = named_counts.get(input_name, 0)
        named_counts[input_name] = named_index + 1
        dataset = definition.get("dataset") or benchmark.get("dataset")
        family, occurrence = probe_coordinates[index] if probe else (0, 0)
        input_index = family if probe else named_index
        recipe = WorkerInput(
            input_name=input_name,
            index=input_index,
            seed=seed,
            prefix_seed=prefix_seed,
            tokens=mean_tokens,
            token_stddev=token_stddev,
            prefix_tokens=definition.get(
                "shared_prefix_tokens", benchmark.get("shared_prefix_tokens", 0)
            ),
            mode=definition.get(
                "dataset_prompt_mode", benchmark.get("dataset_prompt_mode", "sample")
            ),
            repeat_count=definition.get(
                "dataset_repeat_count", benchmark.get("dataset_repeat_count", 1)
            ),
            text=context.get(
                "prompt_text", definition.get("text", benchmark.get("prompt"))
            ),
            dataset_adapter=dataset["adapter"] if dataset else None,
            dataset_filename=dataset["filename"] if dataset else None,
            calibrated=calibrated
            and benchmark.get("input_token_count", "estimate") == "estimate"
            and not probe,
            atomic=atomic,
            output_token_count=benchmark.get("output_token_count", "provider"),
            output_tokens=int(benchmark.get("mean_output_tokens", 150)),
            probe=probe,
            probe_family=family,
            probe_occurrence=occurrence,
        )
        parameters = dict(benchmark.get("additional_sampling_params") or {})
        output_rng = random.Random(indexed_seed(seed, f"output:{index}"))
        parameters.setdefault(
            "max_tokens",
            max(
                1,
                round(
                    output_rng.gauss(
                        benchmark.get("mean_output_tokens", 150),
                        benchmark.get("stddev_output_tokens", 0),
                    )
                ),
            ),
        )
        if benchmark.get("input_messages"):
            parameters["messages"] = list(benchmark["input_messages"])
        metadata = dict(context)
        metadata["input_generation"] = "worker-index-v1"
        if probe:
            metadata["plan_index"] = index
        if benchmark.get("input_messages"):
            metadata["history_version"] = len(benchmark["input_messages"])
        request = RequestConfig(
            model=benchmark["model"],
            prompt=("", mean_tokens),
            sampling_params=parameters,
            llm_api=benchmark["adapter"],
            metadata=metadata,
            timeout_seconds=duration_seconds(benchmark["timeout"]),
            preparation=recipe,
        )
        if specification:
            steps.append(
                RequestStep(
                    specification.name,
                    request,
                    specification.dependencies,
                    specification.after.seconds,
                    specification.at,
                    specification.message_sources,
                )
            )
        else:
            identity = f"probe-{family}-{occurrence}"
            steps.append(
                RequestStep(
                    identity,
                    request,
                    (f"probe-{family}-{occurrence - 1}",) if occurrence else (),
                    (
                        duration_seconds((probe or {}).get("delay", 0))
                        if occurrence
                        else 0
                    ),
                )
            )
    return steps


class WorkerInputPreparer:
    """One bounded cache and one CPU preparation lane per request actor."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.plans: OrderedDict[str, PromptPlan] = OrderedDict()
        self.prefix_bytes = 0
        self.calibrations: OrderedDict[str, CorpusCalibration] = OrderedDict()

    def prepare(
        self, request: RequestConfig, environment: Mapping[str, str]
    ) -> RequestConfig:
        recipe = request.preparation
        if recipe is None:
            return request
        started_at = time.monotonic()
        with self.lock:
            if (
                request.deadline_at is not None
                and datetime.now(timezone.utc) >= request.deadline_at
            ):
                raise TimeoutError("Worker deadline elapsed before input preparation")
            tokenizer = _prepared_tokenizer(
                environment[TOKENIZER_PATH],
                environment.get(TOKENIZER_FAST, "true").lower() == "true",
            )
            paths = json.loads(environment.get(INPUT_DATASET_PATHS, "{}"))
            dataset_path = paths.get(recipe.input_name) or environment.get(
                WORKER_DATASET_PATH
            )
            source = (
                PromptDatasetSource.external(
                    recipe.dataset_adapter,
                    dataset_path,
                    filename=recipe.dataset_filename,
                )
                if recipe.dataset_adapter and dataset_path
                else PromptDatasetSource.builtin_sonnet()
            )
            if recipe.dataset_adapter and not dataset_path:
                raise ValueError("Worker dataset path is missing")
            calibration_key = json.dumps(
                [
                    recipe.dataset_adapter,
                    recipe.dataset_filename,
                    dataset_path,
                    environment[TOKENIZER_PATH],
                    environment.get(TOKENIZER_FAST, "true"),
                ]
            )
            if recipe.calibrated:
                if calibration_key not in self.calibrations:
                    self.calibrations[calibration_key] = calibrate_corpus(
                        tokenizer, source
                    )
                self.calibrations.move_to_end(calibration_key)
                if len(self.calibrations) > 4:
                    self.calibrations.popitem(last=False)
            corpus_calibration = (
                self.calibrations[calibration_key] if recipe.calibrated else None
            )

            def get_length(text: str) -> int:
                if corpus_calibration is not None:
                    return max(
                        0, round(len(text) / corpus_calibration.characters_per_token)
                    )
                return len(tokenizer.encode(text, add_special_tokens=False))

            if recipe.text is not None:
                prompt = (recipe.text, get_length(recipe.text))
                evidence: Dict[str, Any] = {"source": "literal"}
            elif (
                recipe.calibrated
                and recipe.mode == "concatenate"
                and recipe.repeat_count == 1
            ):
                plan_document = recipe.model_dump(
                    exclude={"index", "probe_family", "probe_occurrence"}
                )
                if recipe.prefix_seed is not None:
                    plan_document["seed"] = recipe.prefix_seed
                cache_key = json.dumps(
                    [
                        plan_document,
                        dataset_path,
                        environment[TOKENIZER_PATH],
                        environment.get(TOKENIZER_FAST, "true"),
                    ],
                    sort_keys=True,
                )
                cached_plan = self.plans.get(cache_key)
                if cached_plan is None:
                    cached_plan = calibrate(
                        tokenizer,
                        recipe.seed,
                        recipe.tokens,
                        recipe.token_stddev,
                        recipe.prefix_tokens,
                        recipe.output_tokens,
                        source=source,
                        shared_prefix_seed=recipe.prefix_seed,
                        calibration=corpus_calibration,
                    )
                    self.plans[cache_key] = cached_plan
                    self.prefix_bytes += 4 * len(cached_plan.prefix)
                    while len(self.plans) > 1 and (
                        len(self.plans) > 128 or self.prefix_bytes > 64 * 1024 * 1024
                    ):
                        _, evicted_plan = self.plans.popitem(last=False)
                        self.prefix_bytes -= 4 * len(evicted_plan.prefix)
                self.plans.move_to_end(cache_key)
                rendering_plan = replace(cached_plan, seed=recipe.seed)
                prompt, evidence = render_chunk(rendering_plan, recipe.index, 1)[0]
            else:
                request_seed = (
                    recipe.seed
                    if recipe.atomic
                    else indexed_seed(
                        recipe.seed, f"input:{recipe.index // recipe.repeat_count}"
                    )
                )
                prompts, evidences = prepare_prompt_requests(
                    source=source,
                    prompt_mode=recipe.mode,
                    num_requests=1,
                    repeat_count=1,
                    mean_input_tokens=recipe.tokens,
                    stddev_input_tokens=recipe.token_stddev,
                    mean_output_tokens=recipe.output_tokens,
                    get_token_length=get_length,
                    seed=request_seed,
                    shared_prefix_tokens=recipe.prefix_tokens,
                    shared_prefix_seed=(
                        recipe.prefix_seed
                        if recipe.prefix_seed is not None
                        else recipe.seed
                    ),
                )
                prompt, evidence = prompts[0], evidences[0]
            evidence["input_count_source"] = (
                "calibrated_estimate" if recipe.calibrated else "tokenizer"
            )
            if corpus_calibration is not None:
                estimated_count = evidence.pop("actual_input_tokens", None)
                if estimated_count is not None:
                    evidence["estimated_input_tokens"] = estimated_count
                evidence["characters_per_token"] = (
                    corpus_calibration.characters_per_token
                )
                evidence["calibration_records"] = corpus_calibration.record_count
            metadata = dict(request.metadata or {})
            if recipe.probe:
                probe_config = {**recipe.probe, "trials": 1, "repeats_after_prime": 1}
                probe_config.setdefault("shared_prefix_tokens", recipe.prefix_tokens)
                probe_plan = build_cache_probe_plan(
                    [prompt], probe_config, tokenizer, recipe.seed
                )
                probe_request = probe_plan[1 if recipe.probe_occurrence else 0]
                prompt = (probe_request.prompt, probe_request.local_input_tokens)
                metadata.update(probe_request.metadata())
                metadata.update(
                    family_id=f"family-{recipe.probe_family:05d}",
                    occurrence=recipe.probe_occurrence,
                    dispatch_index=(request.metadata or {}).get("plan_index", 0),
                    request_id=f"probe-{recipe.probe_family}-{recipe.probe_occurrence}",
                )
            metadata.update(
                prompt_evidence=evidence,
                payload_evidence=evidence,
                prompt_hash="sha256:" + hashlib.sha256(prompt[0].encode()).hexdigest(),
                local_input_tokens=prompt[1],
                preparation_seconds=time.monotonic() - started_at,
                prepared_at=datetime.now(timezone.utc).isoformat(),
            )
            parameters = dict(request.sampling_params or {})
            if metadata.get("history_version") is not None:
                messages = [
                    *parameters.get("messages", []),
                    {"role": "user", "content": prompt[0]},
                ]
                parameters["messages"] = messages
                metadata["messages_hash"] = (
                    "sha256:"
                    + hashlib.sha256(
                        json.dumps(
                            messages, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                )
            return request.model_copy(
                update={
                    "prompt": prompt,
                    "sampling_params": parameters,
                    "metadata": metadata,
                    "preparation": None,
                }
            )

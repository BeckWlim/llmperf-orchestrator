"""Compile finite YAML task recipes into protocol-agnostic invoke DAGs."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
from itertools import product
from typing import Any, Dict, List, Mapping, Sequence
from uuid import uuid4

from llmperf_backend.models import (
    DimensionReference,
    InvokeStep,
    ParallelStep,
    RepeatStep,
)


@dataclass(frozen=True)
class TaskNodeBlueprint:
    node_id: str
    dispatch_id: str
    dependencies: List[str]
    after_seconds: int
    runner_template: Dict[str, Any]
    role: str
    payload_id: str


@dataclass(frozen=True)
class TaskInstanceBlueprint:
    instance_id: str
    instance_key: str
    dimensions: Dict[str, int]
    trial_index: int
    nodes: List[TaskNodeBlueprint]


@dataclass(frozen=True)
class TaskCompileContext:
    definition_id: str
    definition_name: str
    definition: Dict[str, Any]
    runner_template: Dict[str, Any]
    database_now: datetime
    created_by: str


def _seed(
    seed: int, dimensions: Mapping[str, int], trial: int, seed_namespace: str
) -> int:
    coordinates = ";".join(f"{key}={value}" for key, value in sorted(dimensions.items()))
    value = f"{seed}\0{coordinates}\0{trial}\0{seed_namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFFFFFF


def _value(expression: Any, dimensions: Mapping[str, int]) -> int:
    if isinstance(expression, Mapping):
        return int(dimensions[str(expression["dimension"])])
    if isinstance(expression, DimensionReference):
        return int(dimensions[expression.dimension])
    return int(expression)


def _runner(
    context: TaskCompileContext,
    instance_id: str,
    node_id: str,
    role: str,
    payload_id: str,
    payload_seed: int,
    trial_index: int,
    dimensions: Mapping[str, int],
) -> Dict[str, Any]:
    task_context = {
        "definition_id": context.definition_id,
        "instance_id": instance_id,
        "node_id": node_id,
        "role": role,
        "payload_id": payload_id,
        "payload_seed": payload_seed,
        "trial_index": trial_index,
        "dimensions": dict(dimensions),
    }
    benchmark = dict(context.runner_template["benchmark"])
    benchmark.update(
        {
            "max_completed_requests": 1,
            "concurrent_requests": 1,
            "dataset_seed": payload_seed,
            "task_request": task_context,
        }
    )
    metadata = dict(context.runner_template.get("metadata") or {})
    metadata["task"] = task_context
    prefix = context.runner_template.get("label") or context.definition_name
    return {
        "label": f"{prefix}:{node_id}"[:200],
        "metadata": metadata,
        "benchmark": benchmark,
    }


def estimate_task_definition(definition: Mapping[str, Any]) -> Dict[str, int]:
    """Count expanded atomic nodes without allocating runtime identities."""

    matrix = definition.get("matrix") or {}
    dimension_names = sorted(matrix)
    combinations = (
        product(*(matrix[name] for name in dimension_names))
        if dimension_names
        else [()]
    )
    instances = 0
    nodes = 0
    for values in combinations:
        dimensions = dict(zip(dimension_names, (int(value) for value in values)))
        per_instance = 0
        for step in definition["sequence"]:
            kind = str(step["kind"])
            if kind == "invoke":
                per_instance += 1
            elif kind == "repeat":
                count = _value(step["count"], dimensions)
                if count < 0 or count > 100:
                    raise ValueError("expanded repeat count must be between 0 and 100")
                per_instance += count
            elif kind == "parallel":
                per_instance += len(step["invokes"])
            else:
                raise ValueError(f"unsupported task syntax: {kind}")
        trials = int(definition.get("trials", 1))
        instances += trials
        nodes += per_instance * trials
    return {"instances": instances, "nodes": nodes}


def compile_task_definition(context: TaskCompileContext) -> List[TaskInstanceBlueprint]:
    """Expand matrix/repeat/parallel syntax into a bounded atomic invoke graph."""

    definition = context.definition
    matrix = definition.get("matrix") or {}
    dimension_names = sorted(matrix)
    combinations: Sequence[Sequence[int]] = (
        product(*(matrix[name] for name in dimension_names))
        if dimension_names
        else [()]
    )
    instances = []
    for values in combinations:
        dimensions = dict(zip(dimension_names, (int(value) for value in values)))
        for trial_index in range(int(definition.get("trials", 1))):
            instance_id = str(uuid4())
            nodes: List[TaskNodeBlueprint] = []
            frontier: List[str] = []

            def append(invoke: Mapping[str, Any], node_id: str, delay: int) -> str:
                if delay < 0:
                    raise ValueError("task delays cannot be negative")
                dispatch_id = str(uuid4())
                payload_id = str(invoke["payload"])
                payload_spec = definition["payloads"][payload_id]
                payload_seed = _seed(
                    int(definition["seed"]),
                    dimensions,
                    trial_index,
                    str(payload_spec["seed_namespace"]),
                )
                nodes.append(
                    TaskNodeBlueprint(
                        node_id=node_id,
                        dispatch_id=dispatch_id,
                        dependencies=list(frontier),
                        after_seconds=delay,
                        runner_template=_runner(
                            context,
                            instance_id,
                            node_id,
                            str(invoke["role"]),
                            payload_id,
                            payload_seed,
                            trial_index,
                            dimensions,
                        ),
                        role=str(invoke["role"]),
                        payload_id=payload_id,
                    )
                )
                return dispatch_id

            for raw_step in definition["sequence"]:
                kind = str(raw_step["kind"])
                if kind == "invoke":
                    dispatch_id = append(
                        raw_step,
                        str(raw_step["id"]),
                        _value(raw_step.get("after_seconds", 0), dimensions),
                    )
                    frontier = [dispatch_id]
                elif kind == "repeat":
                    count = _value(raw_step["count"], dimensions)
                    interval = _value(raw_step.get("interval_seconds", 0), dimensions)
                    if count < 0 or count > 100:
                        raise ValueError("expanded repeat count must be between 0 and 100")
                    invoke = raw_step["invoke"]
                    invoke_delay = _value(invoke.get("after_seconds", 0), dimensions)
                    for index in range(1, count + 1):
                        node_id = f"{raw_step['id']}:{index}"
                        dispatch_id = append(invoke, node_id, interval + invoke_delay)
                        frontier = [dispatch_id]
                elif kind == "parallel":
                    parents = list(frontier)
                    dispatch_ids = []
                    for invoke in raw_step["invokes"]:
                        frontier = parents
                        dispatch_ids.append(
                            append(
                                invoke,
                                str(invoke["id"]),
                                _value(raw_step.get("after_seconds", 0), dimensions)
                                + _value(invoke.get("after_seconds", 0), dimensions),
                            )
                        )
                    frontier = dispatch_ids
                else:
                    raise ValueError(f"unsupported task syntax: {kind}")
            coordinates = ";".join(
                f"{key}={value}" for key, value in sorted(dimensions.items())
            )
            instances.append(
                TaskInstanceBlueprint(
                    instance_id=instance_id,
                    instance_key=f"{coordinates};trial={trial_index}",
                    dimensions=dimensions,
                    trial_index=trial_index,
                    nodes=nodes,
                )
            )
    return instances

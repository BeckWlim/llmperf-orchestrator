"""Generic evidence projection for compiled task graphs."""

from typing import Any, Dict, List, Mapping, Sequence


def build_task_analyses(
    instances: Sequence[Mapping[str, Any]],
    dispatches: Sequence[Mapping[str, Any]],
    runners: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Join task topology and Runner evidence without interpreting experiment roles."""

    by_instance: Dict[str, List[Mapping[str, Any]]] = {}
    for dispatch in dispatches:
        instance_id = dispatch.get("task_instance_id")
        if isinstance(instance_id, str):
            by_instance.setdefault(instance_id, []).append(dispatch)

    analyses: List[Dict[str, Any]] = []
    for instance in instances:
        instance_id = instance.get("task_instance_id")
        if not isinstance(instance_id, str):
            continue
        nodes = []
        for dispatch in by_instance.get(instance_id, []):
            runner_id = dispatch.get("runner_id")
            runner = runners.get(str(runner_id)) if runner_id else None
            nodes.append(
                {
                    **dict(dispatch),
                    "runner": dict(runner) if runner is not None else None,
                }
            )
        analyses.append({**dict(instance), "nodes": nodes})
    return analyses

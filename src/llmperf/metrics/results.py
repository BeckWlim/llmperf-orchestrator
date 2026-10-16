"""Versioned benchmark result serialization and export helpers."""

import json
import subprocess
import time
from typing import Any, Dict, Optional, List, Mapping, Sequence

from llmperf.version import PROTOCOL_VERSION

RESULTS_VERSION = PROTOCOL_VERSION


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
                    "outcome": (instance.get("outcome") or {})
                    .get("nodes", {})
                    .get(dispatch.get("node_id")),
                }
            )
        analyses.append({**dict(instance), "nodes": nodes})
    return analyses

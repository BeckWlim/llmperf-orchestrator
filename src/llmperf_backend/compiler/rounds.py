"""Deterministic expansion and pure runtime admission for Runner rounds."""

from copy import deepcopy
import hashlib
from itertools import product
import json
from typing import Any, Dict, List, Literal, Mapping, Sequence


def _seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def expand_runner_group(
    template: Dict[str, Any],
    name: str,
    matrix: Mapping[str, Sequence[int]],
    phase: str | None = None,
) -> List[Dict[str, Any]]:
    """Expand validated coordinates only; never resolve artifacts per coordinate."""
    dimensions = sorted(matrix)
    expanded: List[Dict[str, Any]] = []
    for values in product(*(matrix[dimension] for dimension in dimensions)):
        coordinates = dict(zip(dimensions, values))
        identity = json.dumps(
            [name, coordinates], sort_keys=True, separators=(",", ":")
        )
        runner = deepcopy(template)
        runner["benchmark"]["dataset_seed"] = _seed(
            f"runner-matrix-v1:{template['benchmark'].get('dataset_seed', 11111)}:{identity}"
        )
        runner["metadata"] = {
            **template.get("metadata", {}),
            "runner_group": {"name": name, "dimensions": coordinates, "phase": phase},
        }
        coordinate_label = ",".join(
            f"{key}={value}" for key, value in coordinates.items()
        )
        runner["label"] = f"{name}/{coordinate_label}"[:200]
        expanded.append(runner)
    return expanded


def expand_rounds(
    runners: Sequence[Dict[str, Any]],
    count: int,
    seed: int,
    prefix_seed_scope: Literal["round", "family"] = "round",
    phases: Sequence[str] = ("default",),
) -> List[Dict[str, Any]]:
    """Family scope fixes each template's prefix across completion-gated rounds.

    Named inputs derive independent prefixes from this template seed. Round and
    request indices change suffixes; no fairness or randomized submission lives here.
    The default retains the original round seed derivation.
    """
    if prefix_seed_scope not in {"round", "family"}:
        raise ValueError("prefix_seed_scope must be round or family")
    expanded: List[Dict[str, Any]] = []
    for cycle in range(1, count + 1):
        for phase_index, phase in enumerate(phases):
            stage_index = (cycle - 1) * len(phases) + phase_index + 1
            for family_index, template in enumerate(runners, 1):
                group = template.get("metadata", {}).get("runner_group")
                assigned_phase = (group or {}).get("phase") or phases[0]
                if assigned_phase != phase:
                    continue
                runner = deepcopy(template)
                if prefix_seed_scope == "family":
                    identity = (
                        json.dumps(
                            [group["name"], group["dimensions"]],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if group
                        else str(family_index)
                    )
                    family_prefix_seed = _seed(
                        f"round-family-prefix-v1:{seed}:{identity}"
                    )
                    runner["benchmark"]["shared_prefix_seed"] = family_prefix_seed
                    runner["benchmark"]["dataset_seed"] = _seed(
                        f"round-family-suffix-v1:{family_prefix_seed}:{cycle}:{phase}"
                    )
                else:
                    runner["benchmark"]["dataset_seed"] = _seed(
                        f"{seed}:{stage_index}:{family_index}"
                    )
                runner["label"] = (
                    f"round-{stage_index:03d}/{template.get('label') or family_index}"[
                        :200
                    ]
                )
                runner["metadata"] = {
                    **runner.get("metadata", {}),
                    "round": stage_index,
                    "runner_round": {
                        "index": stage_index,
                        "count": count * len(phases),
                        "family": family_index,
                        "seed": seed,
                        "cycle": cycle,
                        "phase": phase,
                    },
                }
                if prefix_seed_scope == "family":
                    runner["metadata"]["runner_round"].update(
                        prefix_seed_scope="family",
                        prefix_seed_version="round-family-prefix-v1",
                        suffix_seed_version="round-family-suffix-v1",
                    )
                expanded.append(runner)
    return expanded


def ready_round_runners(
    runners: Sequence[Dict[str, Any]], states: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Only the earliest unfinished round in each campaign may execute."""
    earliest: Dict[str, int] = {}
    for state in states:
        if state["status"] not in {"queued", "running"}:
            continue
        campaign_id = str(state["campaign_id"])
        round_index = int(state["round_index"])
        earliest[campaign_id] = min(earliest.get(campaign_id, round_index), round_index)
    return [
        runner
        for runner in runners
        if not runner.get("metadata", {}).get("runner_round")
        or runner["metadata"]["runner_round"]["index"]
        == earliest.get(str(runner["campaign_id"]))
    ]

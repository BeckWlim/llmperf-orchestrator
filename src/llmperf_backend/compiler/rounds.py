"""Deterministic expansion and pure runtime admission for Runner rounds."""

from copy import deepcopy
import hashlib
from typing import Any, Dict, List, Mapping, Sequence


def expand_rounds(
    runners: Sequence[Dict[str, Any]], count: int, seed: int
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for round_index in range(1, count + 1):
        for family_index, template in enumerate(runners, 1):
            runner = deepcopy(template)
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{round_index}:{family_index}".encode()
                ).digest()[:8],
                "big",
            )
            runner["benchmark"]["dataset_seed"] = derived_seed
            runner["label"] = (
                f"round-{round_index:03d}/{template.get('label') or family_index}"[:200]
            )
            runner["metadata"] = {
                **runner.get("metadata", {}),
                "round": round_index,
                "runner_round": {
                    "index": round_index,
                    "count": count,
                    "family": family_index,
                    "seed": seed,
                },
            }
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

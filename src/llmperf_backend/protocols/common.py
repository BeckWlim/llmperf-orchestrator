"""Shared deterministic prompt and one-Runner phase construction."""

import hashlib
from typing import Any, Dict, List, Optional


def prompt_seed(protocol: str, seed: int, *parts: object) -> int:
    payload = "\0".join([protocol, str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def phase_template(
    definition_name: str,
    protocol: str,
    definition_id: str,
    instance_id: str,
    runner_template: Dict[str, Any],
    role: str,
    delay_seconds: int,
    identity: Dict[str, Any],
    prompt_seeds: List[int],
    mapping_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    template = runner_template
    benchmark = dict(template["benchmark"])
    context = {
        "protocol": protocol,
        "definition_id": definition_id,
        "instance_id": instance_id,
        "role": role,
        "delay_seconds": delay_seconds,
        **identity,
    }
    if mapping_keys is None:
        context["prompt_seed"] = prompt_seeds[0]
    elif role == "prime":
        context["prompt_seeds"] = prompt_seeds
        context["mapping_keys"] = mapping_keys
    else:
        context["prompt_seed"] = prompt_seeds[0]
        context["mapping_key"] = mapping_keys[0]
    benchmark.update(
        {
            "max_completed_requests": len(prompt_seeds),
            "concurrent_requests": 1,
            "dataset_seed": prompt_seeds[0],
            "protocol_request": context,
        }
    )
    metadata = dict(template.get("metadata") or {})
    metadata["protocol"] = context
    label_prefix = template.get("label") or definition_name
    phase_label = identity.get("observation_index")
    phase = role if phase_label is None else f"{role}-{phase_label}"
    chain_label = identity.get("chain_index", identity.get("trial_index", 0))
    return {
        "label": f"{label_prefix}:{phase}:d{delay_seconds}:i{chain_label}"[:200],
        "metadata": metadata,
        "benchmark": benchmark,
    }

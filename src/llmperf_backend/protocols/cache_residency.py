"""Bundled-Prime access-conditioned residency protocol compiler."""

from datetime import datetime
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from llmperf_backend.planner import as_utc
from llmperf_backend.protocol_schedules import expand_geographic_schedule
from llmperf_backend.protocols.base import (
    DispatchBlueprint,
    InstanceBlueprint,
    ProtocolCompileContext,
)
from llmperf_backend.protocols.common import phase_template, prompt_seed


class CacheResidencyPlugin:
    protocol = "cache-residency/v1"

    @staticmethod
    def _schedule(
        config: Dict[str, Any], database_now: datetime
    ) -> Tuple[datetime, List[Tuple[int, Optional[datetime]]]]:
        schedule = config["schedule"]
        if schedule["kind"] == "relative":
            return database_now, [
                (int(offset), None) for offset in schedule["offsets_seconds"]
            ]
        starts_at = as_utc(datetime.fromisoformat(schedule["starts_at"]))
        observations = expand_geographic_schedule(
            schedule["timezone"],
            starts_at,
            int(schedule["every_seconds"]),
            int(schedule["duration_days"]),
        )
        return starts_at, [
            (int((instant - starts_at).total_seconds()), instant)
            for instant in observations
        ]

    def compile(self, context: ProtocolCompileContext) -> List[InstanceBlueprint]:
        config = context.config
        prime_due_at, scheduled_observations = self._schedule(
            config, context.database_now
        )
        chain_indices = list(range(int(config["chains"])))
        random.Random(int(config["seed"])).shuffle(chain_indices)
        instances = []
        for chain_index in chain_indices:
            instance_id = str(uuid4())
            mapping_keys = [
                f"chain-{chain_index}:observation-{observation_index}"
                for observation_index in range(len(scheduled_observations))
            ]
            prime_seeds = [
                prompt_seed(
                    self.protocol,
                    int(config["seed"]),
                    chain_index,
                    "target",
                    mapping_key,
                )
                for mapping_key in mapping_keys
            ]
            observations = []
            for observation_index, (offset_seconds, scheduled_at) in enumerate(
                scheduled_observations
            ):
                mapping_key = mapping_keys[observation_index]
                control_seed = (
                    prompt_seed(
                        self.protocol,
                        int(config["seed"]),
                        chain_index,
                        "cold_control",
                        mapping_key,
                    )
                    if config.get("cold_control", True)
                    else None
                )
                comparison_order = (
                    "control_first"
                    if control_seed is not None
                    and prompt_seed(
                        self.protocol,
                        int(config["seed"]),
                        chain_index,
                        "order",
                        observation_index,
                    )
                    % 2
                    else "warm_first" if control_seed is not None else "warm_only"
                )
                observations.append(
                    {
                        "observation_index": observation_index,
                        "offset_seconds": offset_seconds,
                        "scheduled_at": (
                            scheduled_at.isoformat()
                            if scheduled_at is not None
                            else None
                        ),
                        "mapping_key": mapping_key,
                        "control_prompt_seed": control_seed,
                        "comparison_order": comparison_order,
                    }
                )
            spec = {
                "chain_index": chain_index,
                "schedule": config["schedule"],
                "mapping": config["mapping"],
                "mapping_keys": mapping_keys,
                "prime_prompt_seeds": prime_seeds,
                "observations": observations,
            }
            identity = {"chain_index": chain_index}
            prime_id = str(uuid4())
            dispatches = [
                DispatchBlueprint(
                    dispatch_id=prime_id,
                    role="prime",
                    state="pending",
                    due_at=prime_due_at,
                    parent_dispatch_id=None,
                    runner_template=phase_template(
                        context.definition_name,
                        self.protocol,
                        context.definition_id,
                        instance_id,
                        context.runner_template,
                        "prime",
                        0,
                        identity,
                        prime_seeds,
                        mapping_keys,
                    ),
                    lineage={"created_by": context.created_by},
                )
            ]
            previous_id = prime_id
            for observation in observations:
                phases = ["warm"]
                if observation["control_prompt_seed"] is not None:
                    phases.append("cold_control")
                    if observation["comparison_order"] == "control_first":
                        phases.reverse()
                for role in phases:
                    observation_index = int(observation["observation_index"])
                    mapping_key = str(observation["mapping_key"])
                    seed = (
                        int(observation["control_prompt_seed"])
                        if role == "cold_control"
                        else prime_seeds[observation_index]
                    )
                    dispatch_id = str(uuid4())
                    dispatches.append(
                        DispatchBlueprint(
                            dispatch_id=dispatch_id,
                            role=f"{role}:{observation_index}",
                            state="blocked",
                            due_at=(
                                datetime.fromisoformat(observation["scheduled_at"])
                                if observation["scheduled_at"] is not None
                                else None
                            ),
                            parent_dispatch_id=previous_id,
                            runner_template=phase_template(
                                context.definition_name,
                                self.protocol,
                                context.definition_id,
                                instance_id,
                                context.runner_template,
                                role,
                                int(observation["offset_seconds"]),
                                {
                                    **identity,
                                    "observation_index": observation_index,
                                },
                                [seed],
                                [
                                    (
                                        f"control:{mapping_key}"
                                        if role == "cold_control"
                                        else mapping_key
                                    )
                                ],
                            ),
                            lineage={
                                "created_by": context.created_by,
                                "observation_index": observation_index,
                                "offset_seconds": observation["offset_seconds"],
                                "scheduled_at": observation["scheduled_at"],
                                "phase": role,
                                "mapping_key": mapping_key,
                            },
                        )
                    )
                    previous_id = dispatch_id
            instances.append(
                InstanceBlueprint(
                    instance_id=instance_id,
                    instance_key=f"chain={chain_index}",
                    spec=spec,
                    checkpoint={},
                    outcome={"observations": {}},
                    dispatches=dispatches,
                )
            )
        return instances

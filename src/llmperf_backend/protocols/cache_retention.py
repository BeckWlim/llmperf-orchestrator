"""Independent-family passive cache-retention protocol compiler."""

from datetime import datetime
import random
from typing import List
from uuid import uuid4

from llmperf_backend.protocols.base import (
    DispatchBlueprint,
    InstanceBlueprint,
    ProtocolCompileContext,
)
from llmperf_backend.protocols.common import phase_template, prompt_seed


class CacheRetentionPlugin:
    protocol = "cache-retention/v1"

    def compile(self, context: ProtocolCompileContext) -> List[InstanceBlueprint]:
        config = context.config
        assignments = [
            (int(delay), trial)
            for delay in config["delay_seconds"]
            for trial in range(int(config["trials_per_delay"]))
        ]
        random.Random(int(config["seed"])).shuffle(assignments)
        instances = []
        for delay, trial in assignments:
            instance_id = str(uuid4())
            target_seed = prompt_seed(
                self.protocol, int(config["seed"]), delay, trial, "target"
            )
            control_seed = (
                prompt_seed(
                    self.protocol,
                    int(config["seed"]),
                    delay,
                    trial,
                    "cold_control",
                )
                if config.get("cold_control", True)
                else None
            )
            comparison_order = (
                "control_first"
                if control_seed is not None
                and prompt_seed(
                    self.protocol, int(config["seed"]), delay, trial, "order"
                )
                % 2
                else "warm_first" if control_seed is not None else "warm_only"
            )
            spec = {
                "delay_seconds": delay,
                "trial_index": trial,
                "prompt_seed": target_seed,
                "control_prompt_seed": control_seed,
                "comparison_order": comparison_order,
            }
            identity = {"trial_index": trial}
            prime_id = str(uuid4())
            dispatches = [
                DispatchBlueprint(
                    dispatch_id=prime_id,
                    role="prime",
                    state="pending",
                    due_at=context.database_now,
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
                        [target_seed],
                    ),
                    lineage={"created_by": context.created_by},
                )
            ]
            phases = ["warm"]
            if control_seed is not None:
                phases.append("cold_control")
                if comparison_order == "control_first":
                    phases.reverse()
            for role in phases:
                seed = control_seed if role == "cold_control" else target_seed
                dispatches.append(
                    DispatchBlueprint(
                        dispatch_id=str(uuid4()),
                        role=role,
                        state="blocked",
                        due_at=None,
                        parent_dispatch_id=prime_id,
                        runner_template=phase_template(
                            context.definition_name,
                            self.protocol,
                            context.definition_id,
                            instance_id,
                            context.runner_template,
                            role,
                            delay,
                            identity,
                            [int(seed)],
                        ),
                        lineage={"created_by": context.created_by},
                    )
                )
            instances.append(
                InstanceBlueprint(
                    instance_id=instance_id,
                    instance_key=f"delay={delay};trial={trial}",
                    spec=spec,
                    checkpoint={},
                    outcome={},
                    dispatches=dispatches,
                )
            )
        return instances

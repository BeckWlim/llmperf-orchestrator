"""Typed task DSL, extensible node hierarchy, and deterministic DAG compiler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
from itertools import product
import re
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from uuid import uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated
from llmperf.configuration.duration import DurationValue
from llmperf_backend.compiler.requests import (
    RequestDependency,
    planned_worker_count,
    planned_token_budget,
    validate_dependencies,
)


MAX_REPEAT_COUNT = 100
MAX_GRAPH_DEPTH = 20
MAX_NODES_PER_INSTANCE = 10_000
MAX_INSTANCES_PER_DEFINITION = 10_000
MAX_PLANNED_SPAN_SECONDS = 86_400
DIMENSION_EXPRESSION = re.compile(r"\A\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)\Z")


# External Pydantic AST.


class CompilerModel(BaseModel):
    """Strict external syntax owned by the task compiler."""

    model_config = ConfigDict(extra="forbid")


class DimensionReference(CompilerModel):
    dimension: str = Field(min_length=1, max_length=64)


def _parse_task_value(raw_value: object) -> object:
    if not isinstance(raw_value, str):
        return raw_value
    expression_match = DIMENSION_EXPRESSION.fullmatch(raw_value.strip())
    if expression_match is None:
        return raw_value
    return {"dimension": expression_match.group("name")}


TaskValue = Annotated[
    Union[int, DimensionReference], BeforeValidator(_parse_task_value)
]


class TaskDuration(DurationValue[Union[float, DimensionReference]]):
    @field_validator("value", mode="before")
    @classmethod
    def parse_dimension(cls, raw_value: object) -> object:
        return _parse_task_value(raw_value)

    @model_validator(mode="after")
    def validate_literal(self) -> TaskDuration:
        if isinstance(self.value, float) and (
            not math.isfinite(self.value) or self.value < 0
        ):
            raise ValueError("duration must be finite and nonnegative")
        return self

    def resolve(self, dimensions: Mapping[str, int]) -> float:
        scalar = (
            float(dimensions[self.value.dimension])
            if isinstance(self.value, DimensionReference)
            and self.value.dimension in dimensions
            else self.value
        )
        if isinstance(scalar, DimensionReference):
            raise ValueError(
                f"task expression references unknown dimension {scalar.dimension}"
            )
        seconds = scalar * self.unit_seconds
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("task delays must be finite and nonnegative")
        return seconds


class TaskPayloadSpec(CompilerModel):
    seed: str = Field(min_length=1, max_length=64)
    text: Optional[str] = Field(default=None, min_length=1)


class DependentNodeConfig(CompilerModel):
    depends_on: List[RequestDependency] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dependencies(self) -> DependentNodeConfig:
        validate_dependencies(self.depends_on)
        return self


class InvokeConfig(DependentNodeConfig):
    """One atomic Provider invocation."""

    name: str = Field(min_length=1, max_length=64)
    payload: str = Field(min_length=1, max_length=64)
    role: Optional[str] = Field(default=None, min_length=1, max_length=64)
    after: TaskDuration = Field(default_factory=lambda: TaskDuration(value=0))

    @property
    def effective_role(self) -> str:
        return self.role or self.name


class RunConfig(DependentNodeConfig):
    """One bounded batch Runner using the definition's Runner template."""

    name: str = Field(min_length=1, max_length=64)
    role: Optional[str] = Field(default=None, min_length=1, max_length=64)
    after: TaskDuration = Field(default_factory=lambda: TaskDuration(value=0))

    @model_validator(mode="after")
    def validate_messages(self) -> RunConfig:
        if any(dependency.reuse_messages for dependency in self.depends_on):
            raise ValueError(
                "run dependencies only wait; define message reuse in request.steps or invoke"
            )
        return self


class RepeatConfig(CompilerModel):
    """Bounded serial expansion of one nested node."""

    name: str = Field(min_length=1, max_length=64)
    count: TaskValue
    every: TaskDuration = Field(default_factory=lambda: TaskDuration(value=0))
    node: TaskNodeSpec


class ParallelConfig(CompilerModel):
    """Concurrent branches sharing one incoming dependency frontier."""

    name: str = Field(min_length=1, max_length=64)
    after: TaskDuration = Field(default_factory=lambda: TaskDuration(value=0))

    branches: List[TaskNodeSpec] = Field(min_length=1, max_length=100)


class SequenceConfig(CompilerModel):
    """Nested serial composition with an optional initial delay."""

    name: str = Field(min_length=1, max_length=64)
    after: TaskDuration = Field(default_factory=lambda: TaskDuration(value=0))

    steps: List[TaskNodeSpec] = Field(min_length=1, max_length=100)


NodeConfig = Union[
    InvokeConfig, RunConfig, RepeatConfig, ParallelConfig, SequenceConfig
]


class TaskNodeSpec(CompilerModel):
    """Readable one-key mapping from a YAML primitive to its configuration."""

    invoke: Optional[InvokeConfig] = None
    run: Optional[RunConfig] = None
    repeat: Optional[RepeatConfig] = None
    parallel: Optional[ParallelConfig] = None
    sequence: Optional[SequenceConfig] = None

    @model_validator(mode="after")
    def validate_primitive(self) -> TaskNodeSpec:
        configured_primitives = [
            name
            for name in ("invoke", "run", "repeat", "parallel", "sequence")
            if getattr(self, name) is not None
        ]
        if len(configured_primitives) != 1:
            raise ValueError("task node must define exactly one primitive")
        return self

    def selected(self) -> Tuple[str, NodeConfig]:
        for primitive_name in ("invoke", "run", "repeat", "parallel", "sequence"):
            primitive_config = getattr(self, primitive_name)
            if primitive_config is not None:
                return primitive_name, primitive_config
        raise ValueError("task node has no configured primitive")


RepeatConfig.model_rebuild()
ParallelConfig.model_rebuild()
SequenceConfig.model_rebuild()


class TaskInstancesConfig(CompilerModel):
    """Independent TaskInstance expansion applied around one workflow graph."""

    matrix: Dict[str, List[int]] = Field(default_factory=dict)
    trials: int = Field(default=1, ge=1, le=1_000)
    seed: int = Field(default=11111, ge=0, le=18_446_744_073_709_551_615)
    prefix_seed_scope: Literal["trial", "family"] = "trial"

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, matrix: Dict[str, List[int]]) -> Dict[str, List[int]]:
        for dimension_name, dimension_values in matrix.items():
            if not dimension_name or len(dimension_name) > 64:
                raise ValueError("task matrix dimension names must contain 1-64 chars")
            if not dimension_values or len(dimension_values) > 100:
                raise ValueError("task matrix dimensions require 1-100 values")
            if len(set(dimension_values)) != len(dimension_values):
                raise ValueError("task matrix dimension values must be unique")
        return matrix

    @property
    def count(self) -> int:
        combinations = 1
        for dimension_values in self.matrix.values():
            combinations *= len(dimension_values)
        return combinations * self.trials


class TaskRecipe(CompilerModel):
    """Validated compiler input independent from one submitted Runner template."""

    name: str = Field(min_length=1, max_length=200)
    instances: TaskInstancesConfig = Field(default_factory=TaskInstancesConfig)
    payloads: Dict[str, TaskPayloadSpec] = Field(default_factory=dict, max_length=100)
    workflow: List[TaskNodeSpec] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_compiler_semantics(self) -> TaskRecipe:
        TaskCompiler.validate_recipe(self)
        return self

    @property
    def instance_count(self) -> int:
        return self.instances.count


TaskRunner = TypeVar("TaskRunner")


class TaskDefinitionCreate(TaskRecipe, Generic[TaskRunner]):
    """External task AST parameterized by its owning Runner input model."""

    runner: TaskRunner

    def recipe(self) -> TaskRecipe:
        recipe_document = self.model_dump(mode="json", exclude={"runner"})
        return TaskRecipe.model_validate(recipe_document)


# UUID-free compilation-table intermediate representation.


@dataclass(frozen=True)
class TaskExpression:
    """Resolved-at-compilation scalar with one stable representation."""

    literal_value: Optional[int]
    dimension_name: Optional[str]

    @classmethod
    def from_value(cls, value: TaskValue) -> TaskExpression:
        if isinstance(value, DimensionReference):
            return cls(literal_value=None, dimension_name=value.dimension)
        return cls(literal_value=int(value), dimension_name=None)

    def resolve(self, dimensions: Mapping[str, int]) -> int:
        if self.dimension_name is not None:
            if self.dimension_name not in dimensions:
                raise ValueError(
                    f"task expression references unknown dimension {self.dimension_name}"
                )
            return int(dimensions[self.dimension_name])
        if self.literal_value is None:
            raise ValueError("task expression has no literal or dimension")
        return self.literal_value


@dataclass(frozen=True)
class CompiledInstanceRow:
    """One stable matrix/trial row in the compiler intermediate table."""

    instance_key: str
    dimensions: Dict[str, int]
    trial_index: int


@dataclass(frozen=True)
class CompiledNodeRow:
    """One expanded logical node row before runtime identity assignment."""

    instance_key: str
    node_id: str
    dependencies: Tuple[str, ...]
    after_seconds: float
    role: str
    payload_id: Optional[str]
    payload_seed: int
    prompt_text: Optional[str] = None
    execution_kind: Literal["atomic", "batch"] = "atomic"
    message_sources: Tuple[str, ...] = ()


class CompilationTable:
    """Indexed in-memory IR table cached between expansion and assembly."""

    def __init__(
        self,
        instances: Iterable[CompiledInstanceRow],
        nodes: Iterable[CompiledNodeRow],
    ):
        instance_rows = tuple(instances)
        node_rows = tuple(nodes)
        instance_keys = {instance.instance_key for instance in instance_rows}
        if len(instance_keys) != len(instance_rows):
            raise ValueError("compiled task instance keys must be unique")
        nodes_by_instance: Dict[str, List[CompiledNodeRow]] = {
            instance_key: [] for instance_key in instance_keys
        }
        for node_row in node_rows:
            if node_row.instance_key not in nodes_by_instance:
                raise ValueError(
                    f"compiled node references unknown instance {node_row.instance_key}"
                )
            nodes_by_instance[node_row.instance_key].append(node_row)
        self.instances = instance_rows
        self.nodes = node_rows
        self._nodes_by_instance = {
            instance_key: tuple(instance_nodes)
            for instance_key, instance_nodes in nodes_by_instance.items()
        }

    def nodes_for(self, instance_key: str) -> Tuple[CompiledNodeRow, ...]:
        try:
            return self._nodes_by_instance[instance_key]
        except KeyError as exc:
            raise ValueError(f"unknown compiled instance {instance_key}") from exc


def parallel_request_groups(
    nodes: Sequence[CompiledNodeRow],
) -> Tuple[Tuple[CompiledNodeRow, ...], ...]:
    """Keep atomic completion boundaries; runtime pools all eligible requests."""
    return tuple((node,) for node in nodes)


@dataclass(frozen=True)
class TaskNodeAssembly:
    """Runtime-ready node with UUID dependencies and serialized Runner input."""

    node_id: str
    dispatch_id: str
    dependencies: Tuple[str, ...]
    after_seconds: float
    runner_template: Dict[str, Any]
    role: str
    payload_id: Optional[str]
    parallel_leader: Optional[str] = None
    parallel_size: int = 1


@dataclass(frozen=True)
class TaskInstanceAssembly:
    """Runtime-ready instance produced from the stable compilation table."""

    instance_id: str
    instance_key: str
    dimensions: Dict[str, int]
    trial_index: int
    nodes: Tuple[TaskNodeAssembly, ...]


@dataclass(frozen=True)
class TaskCompileContext:
    definition_id: str
    definition_name: str
    definition: TaskRecipe
    runner_template: Dict[str, Any]


@dataclass(frozen=True)
class InstanceContext:
    seed: int
    dimensions: Mapping[str, int]
    trial_index: int
    payloads: Mapping[str, TaskPayloadSpec]
    prefix_seed_scope: Literal["trial", "family"] = "trial"

    def family_prefix_seed(self) -> int:
        coordinates = ";".join(
            f"{key}={value}" for key, value in sorted(self.dimensions.items())
        )
        material = f"family-prefix-v1\0{self.seed}\0{coordinates}"
        return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")

    def batch_seed(self, node_id: str) -> int:
        if self.prefix_seed_scope == "family":
            suffix_material = (
                f"family-suffix-v1\0{self.family_prefix_seed()}"
                f"\0{self.trial_index}\0{node_id}"
            )
            return int.from_bytes(
                hashlib.sha256(suffix_material.encode()).digest()[:8], "big"
            )
        coordinates = ";".join(
            f"{key}={value}" for key, value in sorted(self.dimensions.items())
        )
        material = (
            f"batch-v1\0{self.seed}\0{coordinates}\0{self.trial_index}\0{node_id}"
        )
        return int.from_bytes(
            hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
        )

    def payload_seed(self, payload_id: str) -> int:
        payload_spec = self.payloads.get(payload_id)
        if payload_spec is None:
            raise ValueError(f"task invoke references unknown payload {payload_id}")
        coordinates = ";".join(
            f"{key}={value}" for key, value in sorted(self.dimensions.items())
        )
        seed_material = (
            f"{self.seed}\0{coordinates}\0{self.trial_index}\0" f"{payload_spec.seed}"
        ).encode("utf-8")
        return (
            int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
            & 0x7FFFFFFF
        )


class GraphBuilder:
    """Mutable builder scoped to one matrix coordinate and trial."""

    def __init__(self, instance: InstanceContext, instance_key: str):
        self.instance = instance
        self.instance_key = instance_key
        self._nodes: List[CompiledNodeRow] = []
        self._node_ids: set[str] = set()

    def emit(
        self,
        *,
        node_id: str,
        dependencies: Tuple[str, ...],
        after_seconds: float,
        role: str,
        payload_id: Optional[str],
        message_sources: Tuple[str, ...] = (),
        execution_kind: Literal["atomic", "batch"] = "atomic",
    ) -> str:
        if not node_id or len(node_id) > 120:
            raise ValueError("expanded task node IDs must contain 1-120 chars")
        if node_id in self._node_ids:
            raise ValueError(f"expanded task node ID is not unique: {node_id}")
        if after_seconds < 0:
            raise ValueError("task delays cannot be negative")
        if len(self._nodes) >= MAX_NODES_PER_INSTANCE:
            raise ValueError(
                f"task instance cannot exceed {MAX_NODES_PER_INSTANCE} nodes"
            )
        if execution_kind == "atomic" and payload_id is None:
            raise ValueError("atomic invocation requires a payload")
        compiled_row = CompiledNodeRow(
            instance_key=self.instance_key,
            node_id=node_id,
            dependencies=dependencies,
            after_seconds=after_seconds,
            role=role,
            payload_id=payload_id,
            payload_seed=(
                self.instance.payload_seed(payload_id)
                if payload_id is not None
                else self.instance.batch_seed(node_id)
            ),
            message_sources=message_sources,
            prompt_text=(
                self.instance.payloads[payload_id].text
                if payload_id is not None
                else None
            ),
            execution_kind=execution_kind,
        )
        self._nodes.append(compiled_row)
        self._node_ids.add(node_id)
        return node_id

    def rows(self) -> Tuple[CompiledNodeRow, ...]:
        return tuple(self._nodes)


def _qualified_name(path: Tuple[str, ...], name: str) -> str:
    return ".".join((*path, name))


# Immutable semantic node hierarchy.


@dataclass(frozen=True)
class BaseNode(ABC):
    """Base class for every composable static-DAG task primitive."""

    name: str

    @classmethod
    @abstractmethod
    def from_config(cls, config: CompilerModel, depth: int) -> BaseNode:
        """Build one immutable compiler node from its validated Pydantic AST."""

    @abstractmethod
    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: float,
    ) -> Tuple[str, ...]:
        """Append logical invoke nodes and return the outgoing frontier."""


@dataclass(frozen=True)
class InvokeNode(BaseNode):
    payload_id: str
    role: str
    after: TaskDuration
    depends_on: Tuple[RequestDependency, ...] = ()

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> InvokeNode:
        if not isinstance(config, InvokeConfig):
            raise TypeError("invoke node requires InvokeConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            payload_id=config.payload,
            role=config.effective_role,
            after=config.after,
            depends_on=tuple(config.depends_on),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: float,
    ) -> Tuple[str, ...]:
        node_id = _qualified_name(path, self.name)
        own_delay_seconds = self.after.resolve(builder.instance.dimensions)
        dependencies = tuple(
            _qualified_reference(path, dependency.step)
            for dependency in self.depends_on
        )
        message_sources = tuple(
            _qualified_reference(path, dependency.step)
            for dependency in self.depends_on
            if dependency.reuse_messages
        )
        emitted_node_id = builder.emit(
            node_id=node_id,
            dependencies=tuple(dict.fromkeys((*incoming_frontier, *dependencies))),
            after_seconds=inherited_delay_seconds + own_delay_seconds,
            role=self.role,
            payload_id=self.payload_id,
            message_sources=message_sources,
        )
        return (emitted_node_id,)


@dataclass(frozen=True)
class RunNode(BaseNode):
    role: str
    after: TaskDuration
    depends_on: Tuple[RequestDependency, ...] = ()

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> RunNode:
        if not isinstance(config, RunConfig):
            raise TypeError("run node requires RunConfig")
        _validate_depth(depth)
        return cls(
            config.name,
            config.role or config.name,
            config.after,
            tuple(config.depends_on),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: float,
    ) -> Tuple[str, ...]:
        node_id = builder.emit(
            node_id=_qualified_name(path, self.name),
            dependencies=tuple(
                dict.fromkeys(
                    (
                        *incoming_frontier,
                        *(
                            _qualified_reference(path, dependency.step)
                            for dependency in self.depends_on
                        ),
                    )
                )
            ),
            after_seconds=inherited_delay_seconds
            + self.after.resolve(builder.instance.dimensions),
            role=self.role,
            payload_id=None,
            execution_kind="batch",
        )
        return (node_id,)


@dataclass(frozen=True)
class RepeatNode(BaseNode):
    count: TaskExpression
    every: TaskDuration
    node: BaseNode

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> RepeatNode:
        if not isinstance(config, RepeatConfig):
            raise TypeError("repeat node requires RepeatConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            count=TaskExpression.from_value(config.count),
            every=config.every,
            node=_node_from_spec(config.node, depth + 1),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: float,
    ) -> Tuple[str, ...]:
        repeat_count = self.count.resolve(builder.instance.dimensions)
        if not 0 <= repeat_count <= MAX_REPEAT_COUNT:
            raise ValueError(
                f"expanded repeat count must be between 0 and {MAX_REPEAT_COUNT}"
            )
        interval_seconds = self.every.resolve(builder.instance.dimensions)
        if interval_seconds < 0:
            raise ValueError("task delays cannot be negative")
        outgoing_frontier = incoming_frontier
        repeat_path = (*path, self.name)
        for iteration_index in range(1, repeat_count + 1):
            iteration_delay_seconds = interval_seconds
            if iteration_index == 1:
                iteration_delay_seconds += inherited_delay_seconds
            outgoing_frontier = self.node.expand(
                builder,
                outgoing_frontier,
                (*repeat_path, str(iteration_index)),
                iteration_delay_seconds,
            )
        return outgoing_frontier


@dataclass(frozen=True)
class ParallelNode(BaseNode):
    after: TaskDuration
    branches: Tuple[BaseNode, ...]

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> ParallelNode:
        if not isinstance(config, ParallelConfig):
            raise TypeError("parallel node requires ParallelConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            after=config.after,
            branches=tuple(
                _node_from_spec(branch, depth + 1) for branch in config.branches
            ),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: float,
    ) -> Tuple[str, ...]:
        own_delay_seconds = self.after.resolve(builder.instance.dimensions)
        branch_delay_seconds = inherited_delay_seconds + own_delay_seconds
        parallel_path = (*path, self.name)
        outgoing_frontier: List[str] = []
        for branch in self.branches:
            branch_frontier = branch.expand(
                builder,
                incoming_frontier,
                parallel_path,
                branch_delay_seconds,
            )
            outgoing_frontier.extend(branch_frontier)
        return tuple(outgoing_frontier)


@dataclass(frozen=True)
class SequenceNode(BaseNode):
    after: TaskDuration
    steps: Tuple[BaseNode, ...]

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> SequenceNode:
        if not isinstance(config, SequenceConfig):
            raise TypeError("sequence node requires SequenceConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            after=config.after,
            steps=tuple(_node_from_spec(step, depth + 1) for step in config.steps),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: float,
    ) -> Tuple[str, ...]:
        own_delay_seconds = self.after.resolve(builder.instance.dimensions)
        outgoing_frontier = incoming_frontier
        sequence_path = (*path, self.name)
        for step_index, step in enumerate(self.steps):
            step_delay_seconds = (
                inherited_delay_seconds + own_delay_seconds if step_index == 0 else 0
            )
            outgoing_frontier = step.expand(
                builder,
                outgoing_frontier,
                sequence_path,
                step_delay_seconds,
            )
        return outgoing_frontier


NODE_TYPES = {
    "invoke": InvokeNode,
    "run": RunNode,
    "repeat": RepeatNode,
    "parallel": ParallelNode,
    "sequence": SequenceNode,
}


def _qualified_reference(path: Tuple[str, ...], name: str) -> str:
    return _qualified_name(path, name) if "." not in name else name


def _validate_depth(depth: int) -> None:
    if depth > MAX_GRAPH_DEPTH:
        raise ValueError(f"task graph cannot exceed {MAX_GRAPH_DEPTH} levels")


def _node_from_spec(spec: TaskNodeSpec, depth: int = 1) -> BaseNode:
    primitive_name, primitive_config = spec.selected()
    node_type = NODE_TYPES[primitive_name]
    return node_type.from_config(primitive_config, depth)


@dataclass(frozen=True)
class TaskGraph:
    """Immutable, recursively composable static task graph."""

    roots: Tuple[BaseNode, ...]

    @classmethod
    def from_recipe(cls, recipe: TaskRecipe) -> TaskGraph:
        return cls(roots=tuple(_node_from_spec(spec) for spec in recipe.workflow))

    def expand(
        self, instance: InstanceContext, instance_key: str
    ) -> Tuple[CompiledNodeRow, ...]:
        builder = GraphBuilder(instance, instance_key)
        frontier: Tuple[str, ...] = ()
        for root in self.roots:
            frontier = root.expand(builder, frontier, (), 0)
        compiled_rows = builder.rows()
        ancestors: Dict[str, set[str]] = {}
        prior_nodes: Dict[str, CompiledNodeRow] = {}
        for row in compiled_rows:
            predecessors = set(row.dependencies)
            for dependency in row.dependencies:
                if dependency not in ancestors:
                    raise ValueError("depends_on must reference an earlier node")
                predecessors.update(ancestors[dependency])
            ancestors[row.node_id] = predecessors
            for message_source in row.message_sources:
                if message_source not in predecessors:
                    raise ValueError(
                        "message sources must reference an earlier dependency node"
                    )
                if prior_nodes[message_source].execution_kind != "atomic":
                    raise ValueError(
                        "message sources must reference one atomic request; use request.steps for history within run"
                    )
            prior_nodes[row.node_id] = row
        return compiled_rows


def _matrix_coordinates(recipe: TaskRecipe) -> Iterable[Dict[str, int]]:
    matrix = recipe.instances.matrix
    dimension_names = sorted(matrix)
    if not dimension_names:
        yield {}
        return
    for coordinate_values in product(*(matrix[name] for name in dimension_names)):
        yield dict(zip(dimension_names, (int(value) for value in coordinate_values)))


def _validate_worker_template(
    nodes: Sequence[CompiledNodeRow], benchmark: Mapping[str, Any]
) -> None:
    if any(node.execution_kind == "atomic" for node in nodes):
        if benchmark.get("steps") is not None or benchmark.get("repeat", 1) != 1:
            raise ValueError(
                "invoke requires a single Worker template; use run for request steps or repetition"
            )
    if any(node.message_sources for node in nodes):
        if benchmark.get("adapter") == "vertexai":
            raise ValueError(
                "VertexAI prediction adapter does not support structured message reuse"
            )
        if "messages" in benchmark.get("additional_sampling_params", {}):
            raise ValueError(
                "Reused messages cannot be overridden by sampling messages"
            )


def _planned_span(nodes: Tuple[CompiledNodeRow, ...]) -> float:
    completion_offsets: Dict[str, float] = {}
    for node in nodes:
        dependency_offset = max(
            (completion_offsets[dependency] for dependency in node.dependencies),
            default=0.0,
        )
        completion_offsets[node.node_id] = dependency_offset + node.after_seconds
    return max(completion_offsets.values(), default=0.0)


# Graph compilation and final runtime assembly.


class TaskCompiler:
    """Compile a validated recipe into a stable, UUID-free intermediate table."""

    def __init__(self, context: TaskCompileContext):
        self.context = context
        self.graph = TaskGraph.from_recipe(context.definition)
        self._compilation_table: Optional[CompilationTable] = None

    @classmethod
    def validate_recipe(cls, recipe: TaskRecipe) -> None:
        if recipe.instance_count > MAX_INSTANCES_PER_DEFINITION:
            raise ValueError(
                f"task definition cannot exceed {MAX_INSTANCES_PER_DEFINITION} instances"
            )
        graph = TaskGraph.from_recipe(recipe)
        for coordinate_index, dimensions in enumerate(_matrix_coordinates(recipe)):
            validation_key = f"validation-{coordinate_index}"
            instance_context = InstanceContext(
                seed=recipe.instances.seed,
                dimensions=dimensions,
                trial_index=0,
                payloads=recipe.payloads,
                prefix_seed_scope=recipe.instances.prefix_seed_scope,
            )
            compiled_nodes = graph.expand(instance_context, validation_key)
            if recipe.instances.prefix_seed_scope == "family" and any(
                node.execution_kind != "batch" for node in compiled_nodes
            ):
                raise ValueError("family prefix seeds require run-only workflows")
            planned_span_seconds = _planned_span(compiled_nodes)
            if planned_span_seconds > MAX_PLANNED_SPAN_SECONDS:
                raise ValueError("task planned span cannot exceed 24 hours")

    @classmethod
    def estimate(
        cls,
        definition: Union[TaskRecipe, Mapping[str, Any]],
        requests_per_actor: int = 1,
        benchmark: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, int]:
        if requests_per_actor < 1:
            raise ValueError("requests_per_actor must be positive")
        recipe = (
            definition
            if isinstance(definition, TaskRecipe)
            else TaskRecipe.model_validate(definition)
        )
        graph = TaskGraph.from_recipe(recipe)
        node_count = 0
        runner_count = 0
        request_count = 0
        peak_runners = 0
        max_concurrency = 0
        peak_requests = 0
        coordinate_count = 0
        for coordinate_index, dimensions in enumerate(_matrix_coordinates(recipe)):
            coordinate_count += 1
            estimate_key = f"estimate-{coordinate_index}"
            instance_context = InstanceContext(
                seed=recipe.instances.seed,
                dimensions=dimensions,
                trial_index=0,
                payloads=recipe.payloads,
                prefix_seed_scope=recipe.instances.prefix_seed_scope,
            )
            compiled_nodes = graph.expand(instance_context, estimate_key)
            _validate_worker_template(compiled_nodes, benchmark or {})
            node_count += len(compiled_nodes) * recipe.instances.trials
            groups = parallel_request_groups(compiled_nodes)
            runner_count += len(groups) * recipe.instances.trials
            batch_limit = planned_worker_count(benchmark or {})
            batch_concurrency = min(
                batch_limit,
                int((benchmark or {}).get("concurrent_requests") or batch_limit),
            )
            request_count += (
                sum(
                    batch_limit if node.execution_kind == "batch" else 1
                    for node in compiled_nodes
                )
                * recipe.instances.trials
            )
            runner_weights = {node.node_id: 1 for node in compiled_nodes}
            peak_runners += (
                max(
                    (
                        _weighted_peak(root, dimensions, runner_weights)
                        for root in graph.roots
                    ),
                    default=0,
                )
                * recipe.instances.trials
            )
            max_concurrency = max(
                max_concurrency,
                max(
                    (
                        batch_concurrency if node.execution_kind == "batch" else 1
                        for node in compiled_nodes
                    ),
                    default=0,
                ),
            )
            request_weights = {
                node.node_id: batch_concurrency if node.execution_kind == "batch" else 1
                for node in compiled_nodes
            }
            peak_requests += (
                max(
                    (
                        _weighted_peak(root, dimensions, request_weights)
                        for root in graph.roots
                    ),
                    default=0,
                )
                * recipe.instances.trials
            )
        return {
            "instances": coordinate_count * recipe.instances.trials,
            "nodes": node_count,
            "runners": runner_count,
            "requests": request_count,
            "peak_runners": peak_runners,
            "max_concurrency": max_concurrency,
            "peak_requests": peak_requests,
            "peak_actors": (peak_requests + requests_per_actor - 1)
            // requests_per_actor,
        }

    @classmethod
    def estimate_token_budget(
        cls,
        definition: Mapping[str, Any],
        input_tokens: int,
        output_tokens: int,
        benchmark: Optional[Mapping[str, Any]] = None,
    ) -> int:
        recipe = TaskRecipe.model_validate(definition)
        graph = TaskGraph.from_recipe(recipe)
        total = 0
        for index, dimensions in enumerate(_matrix_coordinates(recipe)):
            context = InstanceContext(
                recipe.instances.seed,
                dimensions,
                0,
                recipe.payloads,
                recipe.instances.prefix_seed_scope,
            )
            histories: Dict[str, int] = {}
            for node in graph.expand(context, f"budget-{index}"):
                if node.execution_kind == "batch":
                    total += (
                        planned_token_budget(
                            benchmark
                            or {
                                "mean_input_tokens": input_tokens,
                                "mean_output_tokens": output_tokens,
                            }
                        )
                        * recipe.instances.trials
                    )
                    continue
                # Literal payloads need not match the configured generated-prompt size.
                user_tokens = (
                    max(input_tokens, len(node.prompt_text.encode("utf-8")))
                    if node.prompt_text
                    else input_tokens
                )
                history_tokens = sum(
                    histories[source] for source in node.message_sources
                )
                total += (
                    history_tokens + user_tokens + output_tokens
                ) * recipe.instances.trials
                histories[node.node_id] = history_tokens + user_tokens + output_tokens
        return total

    def compile(self) -> CompilationTable:
        cached_table = self._compilation_table
        if cached_table is not None:
            return cached_table
        instance_rows: List[CompiledInstanceRow] = []
        node_rows: List[CompiledNodeRow] = []
        recipe = self.context.definition
        for dimensions in _matrix_coordinates(recipe):
            coordinates = ";".join(
                f"{key}={value}" for key, value in sorted(dimensions.items())
            )
            for trial_index in range(recipe.instances.trials):
                instance_key = (
                    f"{coordinates};trial={trial_index}"
                    if coordinates
                    else f"trial={trial_index}"
                )
                compiled_instance = CompiledInstanceRow(
                    instance_key=instance_key,
                    dimensions=dict(dimensions),
                    trial_index=trial_index,
                )
                instance_context = InstanceContext(
                    seed=recipe.instances.seed,
                    dimensions=dimensions,
                    trial_index=trial_index,
                    payloads=recipe.payloads,
                    prefix_seed_scope=recipe.instances.prefix_seed_scope,
                )
                compiled_nodes = self.graph.expand(instance_context, instance_key)
                _validate_worker_template(
                    compiled_nodes, self.context.runner_template.get("benchmark") or {}
                )
                instance_rows.append(compiled_instance)
                node_rows.extend(compiled_nodes)
        compilation_table = CompilationTable(instance_rows, node_rows)
        self._compilation_table = compilation_table
        return compilation_table


def _uuid_identifier() -> str:
    return str(uuid4())


def _weighted_peak(
    node: BaseNode,
    dimensions: Mapping[str, int],
    weights: Mapping[str, int],
    path: Tuple[str, ...] = (),
) -> int:
    """Conservative overlap bound over the compiler's serial/parallel structure."""
    if isinstance(node, (InvokeNode, RunNode)):
        return weights.get(_qualified_name(path, node.name), 0)
    nested_path = (*path, node.name)
    if isinstance(node, ParallelNode):
        return sum(
            _weighted_peak(child, dimensions, weights, nested_path)
            for child in node.branches
        )
    if isinstance(node, SequenceNode):
        return max(
            (
                _weighted_peak(child, dimensions, weights, nested_path)
                for child in node.steps
            ),
            default=0,
        )
    if isinstance(node, RepeatNode):
        return max(
            (
                _weighted_peak(
                    node.node, dimensions, weights, (*nested_path, str(index))
                )
                for index in range(1, node.count.resolve(dimensions) + 1)
            ),
            default=0,
        )
    raise TypeError("Unsupported task node in resource estimate")


class TaskAssembler:
    """Assign runtime identities and serialize one compiled table for persistence."""

    def __init__(
        self,
        context: TaskCompileContext,
        table: CompilationTable,
        identifier_factory: Callable[[], str] = _uuid_identifier,
    ):
        self.context = context
        self.table = table
        self.identifier_factory = identifier_factory

    def assemble(self) -> Tuple[TaskInstanceAssembly, ...]:
        assemblies: List[TaskInstanceAssembly] = []
        for instance_row in self.table.instances:
            instance_id = self.identifier_factory()
            compiled_nodes = self.table.nodes_for(instance_row.instance_key)
            dispatch_ids = {
                node.node_id: self.identifier_factory() for node in compiled_nodes
            }
            memberships = {
                node.node_id: (dispatch_ids[group[0].node_id], len(group))
                for group in parallel_request_groups(compiled_nodes)
                for node in group
                if len(group) > 1
            }
            node_assemblies = tuple(
                self._assemble_node(
                    instance_row,
                    node,
                    instance_id,
                    dispatch_ids,
                    memberships.get(node.node_id),
                )
                for node in compiled_nodes
            )
            assemblies.append(
                TaskInstanceAssembly(
                    instance_id=instance_id,
                    instance_key=instance_row.instance_key,
                    dimensions=dict(instance_row.dimensions),
                    trial_index=instance_row.trial_index,
                    nodes=node_assemblies,
                )
            )
        return tuple(assemblies)

    def _assemble_node(
        self,
        instance: CompiledInstanceRow,
        node: CompiledNodeRow,
        instance_id: str,
        dispatch_ids: Mapping[str, str],
        membership: Optional[Tuple[str, int]] = None,
    ) -> TaskNodeAssembly:
        dependency_dispatch_ids = tuple(
            dispatch_ids[dependency_node_id] for dependency_node_id in node.dependencies
        )
        return TaskNodeAssembly(
            node_id=node.node_id,
            dispatch_id=dispatch_ids[node.node_id],
            dependencies=dependency_dispatch_ids,
            after_seconds=node.after_seconds,
            runner_template=self._runner_template(instance, node, instance_id),
            role=node.role,
            payload_id=node.payload_id,
            parallel_leader=membership[0] if membership else None,
            parallel_size=membership[1] if membership else 1,
        )

    def _runner_template(
        self,
        instance: CompiledInstanceRow,
        node: CompiledNodeRow,
        instance_id: str,
    ) -> Dict[str, Any]:
        from llmperf_backend.configuration.models import (
            BatchBenchmarkConfig,
            CompiledBenchmarkConfig,
        )

        task_context: Dict[str, Any] = {
            "definition_id": self.context.definition_id,
            "instance_id": instance_id,
            "node_id": node.node_id,
            "role": node.role,
            "trial_index": instance.trial_index,
            "dimensions": dict(instance.dimensions),
            "execution_kind": node.execution_kind,
        }
        benchmark = dict(self.context.runner_template["benchmark"])
        if node.message_sources and benchmark.get("adapter") == "vertexai":
            raise ValueError(
                "VertexAI prediction adapter does not support structured message reuse"
            )
        if node.message_sources and "messages" in benchmark.get(
            "additional_sampling_params", {}
        ):
            raise ValueError(
                "Reused messages cannot be overridden by sampling messages"
            )
        benchmark["dataset_seed"] = node.payload_seed
        benchmark["task_context"] = task_context
        if node.execution_kind == "batch":
            task_context.update(dataset_seed=node.payload_seed, seed_version="batch-v1")
            if self.context.definition.instances.prefix_seed_scope == "family":
                family_context = InstanceContext(
                    self.context.definition.instances.seed,
                    instance.dimensions,
                    instance.trial_index,
                    self.context.definition.payloads,
                    "family",
                )
                prefix_seed = family_context.family_prefix_seed()
                benchmark["shared_prefix_seed"] = prefix_seed
                task_context.update(
                    seed_version="family-suffix-v1",
                    shared_prefix_seed=prefix_seed,
                    prefix_seed_version="family-prefix-v1",
                )
            batch_benchmark = BatchBenchmarkConfig.model_validate(benchmark)
            benchmark_document = batch_benchmark.model_dump(mode="json")
        else:
            task_context.update(
                payload_id=node.payload_id,
                payload_seed=node.payload_seed,
                message_sources=list(node.message_sources),
                prompt_text=node.prompt_text,
            )
            if benchmark.get("steps") is not None or benchmark.get("repeat", 1) != 1:
                raise ValueError(
                    "invoke requires a single Worker template; use run for request steps or repetition"
                )
            benchmark.update(repeat=1, concurrent_requests=1)
            atomic_benchmark = CompiledBenchmarkConfig.model_validate(benchmark)
            benchmark_document = atomic_benchmark.model_dump(mode="json")
        prefix = (
            self.context.runner_template.get("label") or self.context.definition_name
        )
        return {
            "label": f"{prefix}:{node.node_id}"[:200],
            "metadata": dict(self.context.runner_template.get("metadata") or {}),
            "benchmark": benchmark_document,
        }

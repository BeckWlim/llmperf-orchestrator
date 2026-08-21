"""Typed task DSL, extensible node hierarchy, and deterministic DAG compiler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from itertools import product
import re
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
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


class TaskPayloadSpec(CompilerModel):
    seed_namespace: str = Field(min_length=1, max_length=64)


class InvokeConfig(CompilerModel):
    """One atomic Provider invocation."""

    name: str = Field(min_length=1, max_length=64)
    payload: str = Field(min_length=1, max_length=64)
    role: Optional[str] = Field(default=None, min_length=1, max_length=64)
    after_seconds: TaskValue = 0

    @property
    def effective_role(self) -> str:
        return self.role or self.name


class RepeatConfig(CompilerModel):
    """Bounded serial expansion of one nested node."""

    name: str = Field(min_length=1, max_length=64)
    count: TaskValue
    every_seconds: TaskValue = 0
    node: TaskNodeSpec


class ParallelConfig(CompilerModel):
    """Concurrent branches sharing one incoming dependency frontier."""

    name: str = Field(min_length=1, max_length=64)
    after_seconds: TaskValue = 0
    branches: List[TaskNodeSpec] = Field(min_length=1, max_length=100)


class SequenceConfig(CompilerModel):
    """Nested serial composition with an optional initial delay."""

    name: str = Field(min_length=1, max_length=64)
    after_seconds: TaskValue = 0
    steps: List[TaskNodeSpec] = Field(min_length=1, max_length=100)


NodeConfig = Union[InvokeConfig, RepeatConfig, ParallelConfig, SequenceConfig]


class TaskNodeSpec(CompilerModel):
    """Readable one-key mapping from a YAML primitive to its configuration."""

    invoke: Optional[InvokeConfig] = None
    repeat: Optional[RepeatConfig] = None
    parallel: Optional[ParallelConfig] = None
    sequence: Optional[SequenceConfig] = None

    @model_validator(mode="after")
    def validate_primitive(self) -> TaskNodeSpec:
        configured_primitives = [
            name
            for name in ("invoke", "repeat", "parallel", "sequence")
            if getattr(self, name) is not None
        ]
        if len(configured_primitives) != 1:
            raise ValueError("task node must define exactly one primitive")
        return self

    def selected(self) -> Tuple[str, NodeConfig]:
        for primitive_name in ("invoke", "repeat", "parallel", "sequence"):
            primitive_config = getattr(self, primitive_name)
            if primitive_config is not None:
                return primitive_name, primitive_config
        raise ValueError("task node has no configured primitive")


RepeatConfig.model_rebuild()
ParallelConfig.model_rebuild()
SequenceConfig.model_rebuild()


# One-way normalization of the original flat task syntax at the input boundary.


def _legacy_node(raw_step: object, position: int) -> Dict[str, Any]:
    if not isinstance(raw_step, Mapping):
        raise ValueError("legacy task sequence entries must be mappings")
    step_kind = str(raw_step.get("kind", ""))
    if step_kind == "invoke":
        invoke_config: Dict[str, Any] = {
            "name": raw_step.get("id"),
            "role": raw_step.get("role"),
            "payload": raw_step.get("payload"),
        }
        if "after_seconds" in raw_step:
            invoke_config["after_seconds"] = raw_step["after_seconds"]
        return {"invoke": invoke_config}
    if step_kind == "repeat":
        repeat_config: Dict[str, Any] = {
            "name": raw_step.get("id"),
            "count": raw_step.get("count"),
            "every_seconds": raw_step.get("interval_seconds", 0),
            "node": _legacy_node(raw_step.get("invoke"), 0),
        }
        return {"repeat": repeat_config}
    if step_kind == "parallel":
        parallel_config: Dict[str, Any] = {
            "name": f"parallel-{position + 1}",
            "after_seconds": raw_step.get("after_seconds", 0),
            "branches": [
                _legacy_node(branch, branch_position)
                for branch_position, branch in enumerate(raw_step.get("invokes", []))
            ],
        }
        return {"parallel": parallel_config}
    raise ValueError(f"unsupported legacy task syntax: {step_kind}")


def _normalize_legacy_definition(raw_value: object) -> object:
    if not isinstance(raw_value, Mapping) or "sequence" not in raw_value:
        return raw_value
    if "workflow" in raw_value:
        raise ValueError("task definition cannot contain both workflow and sequence")
    raw_sequence = raw_value.get("sequence")
    if not isinstance(raw_sequence, list):
        raise ValueError("legacy task sequence must be a list")
    normalized_definition = dict(raw_value)
    normalized_definition.pop("sequence", None)
    normalized_definition["workflow"] = [
        _legacy_node(raw_step, position)
        for position, raw_step in enumerate(raw_sequence)
    ]
    return normalized_definition


class TaskInstancesConfig(CompilerModel):
    """Independent TaskInstance expansion applied around one workflow graph."""

    matrix: Dict[str, List[int]] = Field(default_factory=dict)
    trials: int = Field(default=1, ge=1, le=1_000)
    seed: int = Field(default=11111, ge=0, le=2_147_483_647)

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
    payloads: Dict[str, TaskPayloadSpec] = Field(min_length=1, max_length=100)
    workflow: List[TaskNodeSpec] = Field(min_length=1, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_syntax(cls, raw_value: object) -> object:
        return _normalize_legacy_definition(raw_value)

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
    after_seconds: int
    role: str
    payload_id: str
    payload_seed: int


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


@dataclass(frozen=True)
class TaskNodeAssembly:
    """Runtime-ready node with UUID dependencies and serialized Runner input."""

    node_id: str
    dispatch_id: str
    dependencies: Tuple[str, ...]
    after_seconds: int
    runner_template: Dict[str, Any]
    role: str
    payload_id: str


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

    def payload_seed(self, payload_id: str) -> int:
        payload_spec = self.payloads.get(payload_id)
        if payload_spec is None:
            raise ValueError(f"task invoke references unknown payload {payload_id}")
        coordinates = ";".join(
            f"{key}={value}" for key, value in sorted(self.dimensions.items())
        )
        seed_material = (
            f"{self.seed}\0{coordinates}\0{self.trial_index}\0"
            f"{payload_spec.seed_namespace}"
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
        after_seconds: int,
        role: str,
        payload_id: str,
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
        compiled_row = CompiledNodeRow(
            instance_key=self.instance_key,
            node_id=node_id,
            dependencies=dependencies,
            after_seconds=after_seconds,
            role=role,
            payload_id=payload_id,
            payload_seed=self.instance.payload_seed(payload_id),
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
        inherited_delay_seconds: int,
    ) -> Tuple[str, ...]:
        """Append logical invoke nodes and return the outgoing frontier."""


@dataclass(frozen=True)
class InvokeNode(BaseNode):
    payload_id: str
    role: str
    after_seconds: TaskExpression

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> InvokeNode:
        if not isinstance(config, InvokeConfig):
            raise TypeError("invoke node requires InvokeConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            payload_id=config.payload,
            role=config.effective_role,
            after_seconds=TaskExpression.from_value(config.after_seconds),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: int,
    ) -> Tuple[str, ...]:
        node_id = _qualified_name(path, self.name)
        own_delay_seconds = self.after_seconds.resolve(builder.instance.dimensions)
        emitted_node_id = builder.emit(
            node_id=node_id,
            dependencies=incoming_frontier,
            after_seconds=inherited_delay_seconds + own_delay_seconds,
            role=self.role,
            payload_id=self.payload_id,
        )
        return (emitted_node_id,)


@dataclass(frozen=True)
class RepeatNode(BaseNode):
    count: TaskExpression
    every_seconds: TaskExpression
    node: BaseNode

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> RepeatNode:
        if not isinstance(config, RepeatConfig):
            raise TypeError("repeat node requires RepeatConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            count=TaskExpression.from_value(config.count),
            every_seconds=TaskExpression.from_value(config.every_seconds),
            node=_node_from_spec(config.node, depth + 1),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: int,
    ) -> Tuple[str, ...]:
        repeat_count = self.count.resolve(builder.instance.dimensions)
        if not 0 <= repeat_count <= MAX_REPEAT_COUNT:
            raise ValueError(
                f"expanded repeat count must be between 0 and {MAX_REPEAT_COUNT}"
            )
        interval_seconds = self.every_seconds.resolve(builder.instance.dimensions)
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
    after_seconds: TaskExpression
    branches: Tuple[BaseNode, ...]

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> ParallelNode:
        if not isinstance(config, ParallelConfig):
            raise TypeError("parallel node requires ParallelConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            after_seconds=TaskExpression.from_value(config.after_seconds),
            branches=tuple(
                _node_from_spec(branch, depth + 1) for branch in config.branches
            ),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: int,
    ) -> Tuple[str, ...]:
        own_delay_seconds = self.after_seconds.resolve(builder.instance.dimensions)
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
    after_seconds: TaskExpression
    steps: Tuple[BaseNode, ...]

    @classmethod
    def from_config(cls, config: CompilerModel, depth: int) -> SequenceNode:
        if not isinstance(config, SequenceConfig):
            raise TypeError("sequence node requires SequenceConfig")
        _validate_depth(depth)
        return cls(
            name=config.name,
            after_seconds=TaskExpression.from_value(config.after_seconds),
            steps=tuple(_node_from_spec(step, depth + 1) for step in config.steps),
        )

    def expand(
        self,
        builder: GraphBuilder,
        incoming_frontier: Tuple[str, ...],
        path: Tuple[str, ...],
        inherited_delay_seconds: int,
    ) -> Tuple[str, ...]:
        own_delay_seconds = self.after_seconds.resolve(builder.instance.dimensions)
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
    "repeat": RepeatNode,
    "parallel": ParallelNode,
    "sequence": SequenceNode,
}


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
        return builder.rows()


def _matrix_coordinates(recipe: TaskRecipe) -> Iterable[Dict[str, int]]:
    matrix = recipe.instances.matrix
    dimension_names = sorted(matrix)
    if not dimension_names:
        yield {}
        return
    for coordinate_values in product(*(matrix[name] for name in dimension_names)):
        yield dict(zip(dimension_names, (int(value) for value in coordinate_values)))


def _planned_span(nodes: Tuple[CompiledNodeRow, ...]) -> int:
    completion_offsets: Dict[str, int] = {}
    for node in nodes:
        dependency_offset = max(
            (completion_offsets[dependency] for dependency in node.dependencies),
            default=0,
        )
        completion_offsets[node.node_id] = dependency_offset + node.after_seconds
    return max(completion_offsets.values(), default=0)


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
            )
            compiled_nodes = graph.expand(instance_context, validation_key)
            planned_span_seconds = _planned_span(compiled_nodes)
            if planned_span_seconds > MAX_PLANNED_SPAN_SECONDS:
                raise ValueError("task planned span cannot exceed 24 hours")

    @classmethod
    def estimate(
        cls, definition: Union[TaskRecipe, Mapping[str, Any]]
    ) -> Dict[str, int]:
        recipe = (
            definition
            if isinstance(definition, TaskRecipe)
            else TaskRecipe.model_validate(definition)
        )
        graph = TaskGraph.from_recipe(recipe)
        node_count = 0
        coordinate_count = 0
        for coordinate_index, dimensions in enumerate(_matrix_coordinates(recipe)):
            coordinate_count += 1
            estimate_key = f"estimate-{coordinate_index}"
            instance_context = InstanceContext(
                seed=recipe.instances.seed,
                dimensions=dimensions,
                trial_index=0,
                payloads=recipe.payloads,
            )
            compiled_nodes = graph.expand(instance_context, estimate_key)
            node_count += len(compiled_nodes) * recipe.instances.trials
        return {
            "instances": coordinate_count * recipe.instances.trials,
            "nodes": node_count,
        }

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
                )
                compiled_nodes = self.graph.expand(instance_context, instance_key)
                instance_rows.append(compiled_instance)
                node_rows.extend(compiled_nodes)
        compilation_table = CompilationTable(instance_rows, node_rows)
        self._compilation_table = compilation_table
        return compilation_table


def _uuid_identifier() -> str:
    return str(uuid4())


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
            node_assemblies = tuple(
                self._assemble_node(
                    instance_row,
                    node,
                    instance_id,
                    dispatch_ids,
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
        )

    def _runner_template(
        self,
        instance: CompiledInstanceRow,
        node: CompiledNodeRow,
        instance_id: str,
    ) -> Dict[str, Any]:
        from llmperf_backend.models import CompiledBenchmarkConfig

        task_context = {
            "definition_id": self.context.definition_id,
            "instance_id": instance_id,
            "node_id": node.node_id,
            "role": node.role,
            "payload_id": node.payload_id,
            "payload_seed": node.payload_seed,
            "trial_index": instance.trial_index,
            "dimensions": dict(instance.dimensions),
        }
        benchmark = dict(self.context.runner_template["benchmark"])
        benchmark.update(
            {
                "max_completed_requests": 1,
                "concurrent_requests": 1,
                "dataset_seed": node.payload_seed,
                "task_context": task_context,
            }
        )
        validated_benchmark = CompiledBenchmarkConfig.model_validate(benchmark)
        prefix = (
            self.context.runner_template.get("label") or self.context.definition_name
        )
        return {
            "label": f"{prefix}:{node.node_id}"[:200],
            "metadata": dict(self.context.runner_template.get("metadata") or {}),
            "benchmark": validated_benchmark.model_dump(mode="json"),
        }

"""Initialization-only calibration and bounded parallel prompt preparation."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import random
import threading
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    runtime_checkable,
)

from llmperf.datasets.prompts import (
    PromptDataset,
    PromptDatasetSource,
    load_prompt_dataset,
)

if TYPE_CHECKING:
    from ray import ObjectRef

PreparedPrompt = Tuple[Tuple[str, int], Dict[str, Any]]


class Encoder(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]: ...


@dataclass(frozen=True)
class PromptPlan:
    records: Tuple[str, ...]
    lengths: Tuple[int, ...]
    prefix: str
    prefix_tokens: int
    seed: int
    mean_tokens: int
    stddev_tokens: int
    dataset_source: Optional[PromptDatasetSource] = None
    characters_per_token: float = 0.0
    prefix_record_indices: Tuple[int, ...] = ()
    calibration_records: int = 0


@lru_cache(maxsize=4)
def _indexed_dataset(source: PromptDatasetSource) -> PromptDataset:
    """Retain a bounded number of memory-mapped indexes in each process."""
    return load_prompt_dataset(source)


def _assemble_dataset(
    dataset: PromptDataset, characters_per_token: float, budget: int, seed: int
) -> Tuple[str, List[int]]:
    """Sample without replacement per cycle, without shuffling the entire corpus."""
    rng = random.Random(seed)
    remaining_characters = max(0, round(budget * characters_per_token))
    record_count = len(dataset.records)
    if record_count == 0:
        raise ValueError("Prompt dataset has no usable records")
    pieces: List[str] = []
    selected: List[int] = []
    while remaining_characters > 0:
        swaps: Dict[int, int] = {}
        for remaining_records in range(record_count, 0, -1):
            chosen = rng.randrange(remaining_records)
            position = swaps.get(chosen, chosen)
            swaps[chosen] = swaps.get(remaining_records - 1, remaining_records - 1)
            record_index, record_text = dataset.records[position]
            segment = (record_text + dataset.separator)[:remaining_characters]
            pieces.append(segment)
            selected.append(record_index)
            remaining_characters -= len(segment)
            if remaining_characters == 0:
                break
    return "".join(pieces), selected


def _assemble(
    records: Sequence[str], lengths: Sequence[int], budget: int, seed: int
) -> Tuple[str, List[int]]:
    rng = random.Random(seed)
    pieces: List[str] = []
    selected: List[int] = []
    remaining = budget
    order = list(range(len(records)))
    while remaining > 0:
        rng.shuffle(order)
        for index in order:
            record = records[index]
            token_count = lengths[index]
            selected.append(index)
            if token_count >= remaining:
                characters = max(1, int(len(record) * remaining / token_count))
                pieces.append(record[:characters])
                remaining = 0
                break
            pieces.append(record)
            remaining -= token_count
    return "".join(pieces), selected


def calibrate(
    tokenizer: Encoder,
    seed: int,
    mean_tokens: int,
    stddev_tokens: int,
    shared_prefix_tokens: int,
    output_tokens: int,
    source: Optional[PromptDatasetSource] = None,
) -> PromptPlan:
    if source is not None and source.is_external:
        dataset = _indexed_dataset(source)
        sample_positions = random.Random(0).sample(
            range(len(dataset.records)), min(128, len(dataset.records))
        )
        sampled_characters = 0
        sampled_tokens = 0
        for position in sample_positions:
            _, record_text = dataset.records[position]
            sample_text = (record_text + dataset.separator)[:4096]
            sampled_characters += len(sample_text)
            sampled_tokens += len(
                tokenizer.encode(sample_text, add_special_tokens=False)
            )
        if sampled_characters == 0 or sampled_tokens == 0:
            raise ValueError("Prompt dataset cannot calibrate token estimates")
        characters_per_token = sampled_characters / sampled_tokens
        prefix, prefix_indices = _assemble_dataset(
            dataset, characters_per_token, shared_prefix_tokens, seed
        )
        return PromptPlan(
            records=(),
            lengths=(),
            prefix=prefix,
            prefix_tokens=shared_prefix_tokens,
            seed=seed,
            mean_tokens=mean_tokens,
            stddev_tokens=stddev_tokens,
            dataset_source=source,
            characters_per_token=characters_per_token,
            prefix_record_indices=tuple(prefix_indices),
            calibration_records=len(sample_positions),
        )
    dataset = load_prompt_dataset(PromptDatasetSource.builtin_sonnet())
    records = tuple(text + dataset.separator for _, text in dataset.records)
    lengths = tuple(
        max(1, len(tokenizer.encode(text, add_special_tokens=False)))
        for text in records
    )
    instruction = dataset.instruction(output_tokens)
    instruction_tokens = len(tokenizer.encode(instruction, add_special_tokens=False))
    prefix_budget = max(instruction_tokens, shared_prefix_tokens)
    prefix_body, _ = _assemble(
        records, lengths, prefix_budget - instruction_tokens, seed
    )
    return PromptPlan(
        records,
        lengths,
        instruction + prefix_body,
        prefix_budget,
        seed,
        mean_tokens,
        stddev_tokens,
    )


def render_chunk(plan: PromptPlan, start: int, count: int) -> List[PreparedPrompt]:
    prepared: List[PreparedPrompt] = []
    external_dataset = (
        _indexed_dataset(plan.dataset_source)
        if plan.dataset_source is not None
        else None
    )
    for request_index in range(start, start + count):
        request_seed = int.from_bytes(
            hashlib.sha256(f"{plan.seed}:{request_index}".encode()).digest()[:8], "big"
        )
        rng = random.Random(request_seed)
        target_tokens = max(
            plan.prefix_tokens + 1, int(rng.gauss(plan.mean_tokens, plan.stddev_tokens))
        )
        suffix, indices = (
            _assemble_dataset(
                external_dataset,
                plan.characters_per_token,
                target_tokens - plan.prefix_tokens,
                request_seed,
            )
            if external_dataset is not None
            else _assemble(
                plan.records,
                plan.lengths,
                target_tokens - plan.prefix_tokens,
                request_seed,
            )
        )
        prompt_text = plan.prefix + suffix
        evidence = {
            "source": (
                external_dataset.source
                if external_dataset is not None
                else "builtin-sonnet"
            ),
            "mode": "calibrated-concatenate",
            "seed": plan.seed,
            "request_index": request_index,
            "record_indices": indices,
            "target_input_tokens": target_tokens,
            "input_count_source": "calibrated_estimate",
            "manifest_hash": "sha256:"
            + hashlib.sha256(prompt_text.encode()).hexdigest(),
        }
        if external_dataset is not None:
            evidence.update(
                prefix_record_indices=list(plan.prefix_record_indices),
                calibration_records=plan.calibration_records,
                characters_per_token=plan.characters_per_token,
            )
        prepared.append(((prompt_text, target_tokens), evidence))
    return prepared


class RenderMethod(Protocol):
    def remote(
        self, plan: PromptPlan, start: int, count: int
    ) -> "ObjectRef[List[PreparedPrompt]]": ...


@runtime_checkable
class PreparationHandle(Protocol):
    @property
    def render(self) -> RenderMethod: ...


class PreparationActor:
    def render(self, plan: PromptPlan, start: int, count: int) -> List[PreparedPrompt]:
        return render_chunk(plan, start, count)


def create_preparation_actor() -> PreparationHandle:
    import ray

    handle = (
        ray.remote(PreparationActor)
        .options(num_cpus=1, max_concurrency=1, max_restarts=0, max_task_retries=0)
        .remote()
    )
    if not isinstance(handle, PreparationHandle):
        raise TypeError("Preparation actor does not expose render")
    return handle


class PreparedStream:
    """Ordered, bounded ObjectRef prefetch shared by a Runner's request threads."""

    def __init__(
        self,
        plan: PromptPlan,
        actors: Sequence[PreparationHandle],
        total: int,
        chunk_size: int,
        buffer_chunks: int,
    ):
        import ray

        if not actors or chunk_size < 1 or buffer_chunks < 1 or total < 0:
            raise ValueError("Preparation requires actors and positive chunk bounds")
        self.plan = plan
        self.actors = actors
        self.total = total
        self.chunk_size = chunk_size
        self.next_chunk = 0
        self.next_index = 0
        self.lock = threading.Lock()
        self.pending: deque["ObjectRef[List[PreparedPrompt]]"] = deque()
        self.ready: deque[PreparedPrompt] = deque()
        self.closed = False
        self.ray = ray
        for _ in range(buffer_chunks):
            self._submit()

    def _submit(self) -> None:
        if self.next_chunk >= self.total or self.closed:
            return
        start = self.next_chunk
        count = min(self.chunk_size, self.total - start)
        actor = self.actors[
            (start // self.chunk_size + self.plan.seed) % len(self.actors)
        ]
        self.pending.append(actor.render.remote(self.plan, start, count))
        self.next_chunk += count

    def take(self) -> Tuple[int, PreparedPrompt]:
        with self.lock:
            if self.closed or self.next_index >= self.total:
                raise StopIteration
            if not self.ready:
                self.ready.extend(self.ray.get(self.pending.popleft()))
                self._submit()
            index = self.next_index
            self.next_index += 1
            return index, self.ready.popleft()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for reference in self.pending:
                self.ray.cancel(reference, force=False)
            self.pending.clear()
            self.ready.clear()

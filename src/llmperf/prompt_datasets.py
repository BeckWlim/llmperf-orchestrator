"""Unified prompt dataset loading and deterministic request construction."""

from dataclasses import dataclass
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

Prompt = Tuple[str, int]
PromptEvidence = Dict[str, Any]
PromptRecord = Tuple[int, str]
DatasetPath = Union[str, PathLike[str]]

BUILTIN_SONNET_SOURCE = "builtin-sonnet"
SHAREGPT_ADAPTER = "sharegpt"
TEXT_ADAPTER = "text"
_BUILTIN_SONNET_INSTRUCTION = (
    "Randomly stream lines from the following text with "
    "{output_tokens} output tokens. Don't generate eos tokens:\n\n"
)


@dataclass(frozen=True)
class PromptDataset:
    """Normalized text records from either the bundled or an external source."""

    source: str
    records: Tuple[PromptRecord, ...]
    separator: str
    instruction_template: Optional[str] = None

    def instruction(self, output_tokens: int) -> str:
        if self.instruction_template is None:
            return ""
        return self.instruction_template.format(output_tokens=output_tokens)


@dataclass(frozen=True)
class PromptDatasetAdapter:
    """One explicit conversion from a source artifact to prompt records."""

    name: str
    requires_path: bool
    load: Callable[[Optional[Path]], PromptDataset]
    fixed_prompt_mode: Optional[str] = None


@dataclass(frozen=True)
class PromptDatasetSource:
    """Explicit adapter selection and optional external artifact location."""

    adapter: str
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        adapter = get_prompt_dataset_adapter(self.adapter)
        normalized_path = (
            Path(self.path).expanduser() if self.path is not None else None
        )
        if adapter.requires_path and normalized_path is None:
            raise ValueError(f"Dataset adapter {self.adapter!r} requires a path")
        if not adapter.requires_path and normalized_path is not None:
            raise ValueError(f"Dataset adapter {self.adapter!r} does not accept a path")
        object.__setattr__(self, "path", normalized_path)

    @classmethod
    def builtin_sonnet(cls) -> "PromptDatasetSource":
        return cls(adapter=BUILTIN_SONNET_SOURCE)

    @classmethod
    def external(cls, adapter: str, path: DatasetPath) -> "PromptDatasetSource":
        validate_external_dataset_adapter(adapter)
        return cls(adapter=adapter, path=Path(path))

    @property
    def is_external(self) -> bool:
        return get_prompt_dataset_adapter(self.adapter).requires_path


def _load_builtin_sonnet() -> PromptDataset:
    path = Path(__file__).with_name("sonnet.txt")
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Unable to read bundled sonnet dataset {path}: {exc}"
        ) from exc

    records = tuple((index, line) for index, line in enumerate(lines) if line)
    if not records:
        raise ValueError("Bundled sonnet dataset has no usable text records")
    return PromptDataset(
        source=BUILTIN_SONNET_SOURCE,
        records=records,
        separator="",
        instruction_template=_BUILTIN_SONNET_INSTRUCTION,
    )


def _load_sharegpt(path: Optional[Path]) -> PromptDataset:
    if path is None:
        raise ValueError("ShareGPT dataset adapter requires a dataset path")
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            document = json.load(stream)
    except OSError as exc:
        raise ValueError(f"Unable to read ShareGPT dataset {path}: {exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"ShareGPT dataset is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(document, list):
        raise ValueError("ShareGPT dataset must be a JSON array")

    records = []
    for record_index, record in enumerate(document):
        if not isinstance(record, dict):
            continue
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        first_turn = conversations[0]
        if not isinstance(first_turn, dict):
            continue
        prompt = first_turn.get("value")
        if isinstance(prompt, str) and prompt.strip():
            records.append((record_index, prompt))
    if not records:
        raise ValueError("ShareGPT dataset has no usable first-turn prompts")
    return PromptDataset(
        source=SHAREGPT_ADAPTER,
        records=tuple(records),
        separator="\n\n",
    )


def _load_text(path: Optional[Path]) -> PromptDataset:
    if path is None:
        raise ValueError("Text dataset adapter requires a dataset path")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read text dataset {path}: {exc}") from exc

    records = tuple(
        (record_index, line) for record_index, line in enumerate(lines) if line.strip()
    )
    if not records:
        raise ValueError("Text dataset has no usable non-empty line prompts")
    return PromptDataset(source=TEXT_ADAPTER, records=records, separator="\n\n")


def _load_sonnet_adapter(path: Optional[Path]) -> PromptDataset:
    if path is not None:
        raise ValueError("Bundled sonnet dataset adapter does not accept a path")
    return _load_builtin_sonnet()


_DATASET_ADAPTERS = {
    BUILTIN_SONNET_SOURCE: PromptDatasetAdapter(
        name=BUILTIN_SONNET_SOURCE,
        requires_path=False,
        load=_load_sonnet_adapter,
        fixed_prompt_mode="concatenate",
    ),
    SHAREGPT_ADAPTER: PromptDatasetAdapter(
        name=SHAREGPT_ADAPTER,
        requires_path=True,
        load=_load_sharegpt,
    ),
    TEXT_ADAPTER: PromptDatasetAdapter(
        name=TEXT_ADAPTER,
        requires_path=True,
        load=_load_text,
    ),
}


def external_dataset_adapters() -> Tuple[str, ...]:
    """Return stable adapter identifiers accepted for external artifacts."""

    return tuple(
        name for name, adapter in _DATASET_ADAPTERS.items() if adapter.requires_path
    )


def get_prompt_dataset_adapter(name: str) -> PromptDatasetAdapter:
    """Resolve an adapter without leaking format-specific checks to callers."""

    adapter = _DATASET_ADAPTERS.get(name)
    if adapter is None:
        supported = ", ".join(sorted(_DATASET_ADAPTERS))
        raise ValueError(
            f"Unsupported prompt dataset adapter: {name}. Supported: {supported}"
        )
    return adapter


def validate_external_dataset_adapter(name: str) -> str:
    """Validate and return an external artifact adapter identifier."""

    adapter = get_prompt_dataset_adapter(name)
    if not adapter.requires_path:
        raise ValueError(
            f"Prompt dataset adapter {name!r} does not accept external artifacts"
        )
    return adapter.name


def load_prompt_dataset(source: PromptDatasetSource) -> PromptDataset:
    """Load records through the source's explicitly selected adapter."""

    return get_prompt_dataset_adapter(source.adapter).load(source.path)


def _manifest(document: Dict[str, Any]) -> PromptEvidence:
    result = dict(document)
    result["manifest_hash"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return result


def _positive_int(mean: int, stddev: int, rng: random.Random) -> int:
    value = -1
    while value <= 0:
        value = int(rng.gauss(mean, stddev))
    return value


class _RecordCursor:
    def __init__(self, dataset: PromptDataset, rng: random.Random):
        self.dataset = dataset
        self.rng = rng
        self.order = list(dataset.records)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.cycle = 0

    def next(self) -> Tuple[int, str, int]:
        if self.cursor == len(self.order):
            self.cycle += 1
            self.order = list(self.dataset.records)
            self.rng.shuffle(self.order)
            self.cursor = 0
        record_index, text = self.order[self.cursor]
        self.cursor += 1
        return record_index, text, self.cycle


def _truncate_to_budget(
    initial: str,
    piece: str,
    target_tokens: int,
    get_token_length: Callable[[str], int],
) -> Tuple[str, int]:
    low = 0
    high = len(piece)
    while low < high:
        middle = (low + high + 1) // 2
        if get_token_length(initial + piece[:middle]) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    return piece[:low], low


def concatenate_prompt_requests(
    dataset: PromptDataset,
    num_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    get_token_length: Callable[[str], int],
    seed: int = 11111,
    shared_prefix_tokens: int = 0,
) -> Tuple[List[Prompt], List[PromptEvidence]]:
    """Fill token budgets from one seeded, no-replacement record cursor."""

    rng = random.Random(seed)
    cursor = _RecordCursor(dataset, rng)

    def construct(
        target_tokens: int, initial_text: str, include_instruction: bool
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        text = initial_text
        if include_instruction:
            text += dataset.instruction(mean_output_tokens)
        if get_token_length(text) > target_tokens:
            text, _ = _truncate_to_budget("", text, target_tokens, get_token_length)
        actual_tokens = get_token_length(text)
        segments = []
        while actual_tokens < target_tokens:
            record_index, record, cycle = cursor.next()
            separator = dataset.separator if text and not text.endswith("\n\n") else ""
            piece = separator + record
            complete_tokens = get_token_length(text + piece)
            previous_tokens = actual_tokens
            if complete_tokens <= target_tokens:
                text += piece
                actual_tokens = complete_tokens
                used_characters = len(record)
                truncated = False
            else:
                selected, selected_characters = _truncate_to_budget(
                    text, piece, target_tokens, get_token_length
                )
                text += selected
                actual_tokens = get_token_length(text)
                used_characters = max(0, selected_characters - len(separator))
                truncated = True
            segments.append(
                {
                    "record_index": record_index,
                    "corpus_cycle": cycle,
                    "characters": used_characters,
                    "tokens_added": actual_tokens - previous_tokens,
                    "truncated": truncated,
                }
            )
            if truncated:
                break
        if not text:
            raise ValueError(
                f"{dataset.source} prompt construction could not fit text in the token budget"
            )
        return text, actual_tokens, segments

    shared_prefix = ""
    shared_segments: List[Dict[str, Any]] = []
    if shared_prefix_tokens:
        shared_prefix, _, shared_segments = construct(
            shared_prefix_tokens, "", include_instruction=True
        )

    prompts = []
    manifests = []
    for _ in range(num_requests):
        target_tokens = _positive_int(mean_input_tokens, stddev_input_tokens, rng)
        text, actual_tokens, suffix_segments = construct(
            target_tokens,
            shared_prefix,
            include_instruction=not bool(shared_prefix),
        )
        segments = [*shared_segments, *suffix_segments]
        prompts.append((text, actual_tokens))
        manifests.append(
            _manifest(
                {
                    "source": dataset.source,
                    "mode": "concatenate",
                    "seed": seed,
                    "record_indices": [item["record_index"] for item in segments],
                    "segments": segments,
                    "target_input_tokens": target_tokens,
                    "actual_input_tokens": actual_tokens,
                }
            )
        )
    return prompts, manifests


def sample_prompt_requests(
    dataset: PromptDataset,
    num_requests: int,
    repeat_count: int,
    min_input_tokens: int,
    max_input_tokens: int,
    get_token_length: Callable[[str], int],
    seed: int = 11111,
) -> Tuple[List[Prompt], List[PromptEvidence]]:
    """Select intact records within a token range."""

    rng = random.Random(seed)
    candidates = list(dataset.records)
    rng.shuffle(candidates)
    unique_request_count = math.ceil(num_requests / repeat_count)
    sampled = []
    for record_index, prompt in candidates:
        prompt_len = get_token_length(prompt)
        if min_input_tokens <= prompt_len <= max_input_tokens:
            sampled.append(
                (
                    prompt,
                    prompt_len,
                    _manifest(
                        {
                            "source": dataset.source,
                            "mode": "sample",
                            "seed": seed,
                            "record_indices": [record_index],
                            "target_input_tokens": None,
                            "actual_input_tokens": prompt_len,
                        }
                    ),
                )
            )
            if len(sampled) == unique_request_count:
                break
    if len(sampled) < unique_request_count:
        raise ValueError(
            f"{dataset.source} dataset has only {len(sampled)} matching prompts; "
            f"{unique_request_count} required in token range "
            f"{min_input_tokens}:{max_input_tokens}"
        )
    repeated = (sampled * repeat_count)[:num_requests]
    rng.shuffle(repeated)
    return (
        [(prompt, prompt_len) for prompt, prompt_len, _ in repeated],
        [dict(evidence) for _, _, evidence in repeated],
    )


def prepare_prompt_requests(
    source: PromptDatasetSource,
    prompt_mode: str,
    num_requests: int,
    repeat_count: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    get_token_length: Callable[[str], int],
    seed: int,
    shared_prefix_tokens: int = 0,
) -> Tuple[List[Prompt], List[PromptEvidence]]:
    """Load and construct prompts through the same pipeline for every source."""

    adapter = get_prompt_dataset_adapter(source.adapter)
    dataset = load_prompt_dataset(source)
    effective_mode = adapter.fixed_prompt_mode or prompt_mode
    if effective_mode == "concatenate":
        if repeat_count != 1:
            raise ValueError("concatenated dataset prompts require repeat_count=1")
        return concatenate_prompt_requests(
            dataset,
            num_requests,
            mean_input_tokens,
            stddev_input_tokens,
            mean_output_tokens,
            get_token_length,
            seed,
            shared_prefix_tokens,
        )
    if effective_mode == "sample":
        return sample_prompt_requests(
            dataset,
            num_requests,
            repeat_count,
            max(1, mean_input_tokens - stddev_input_tokens),
            mean_input_tokens + stddev_input_tokens,
            get_token_length,
            seed,
        )
    raise ValueError(f"Unsupported dataset prompt mode: {effective_mode}")

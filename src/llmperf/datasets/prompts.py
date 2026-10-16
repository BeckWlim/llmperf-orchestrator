"""Unified prompt dataset loading and deterministic request construction."""

from dataclasses import dataclass
import codecs
import gzip
import hashlib
import json
import math
import os
from os import PathLike
from pathlib import Path
import random
import re
import tempfile
from typing import (
    Any,
    BinaryIO,
    Callable,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from datasets import (
    Dataset,
    config as datasets_config,
    disable_progress_bars,
    load_dataset,
)
from datasets.exceptions import DatasetGenerationError
from filelock import FileLock, Timeout as FileLockTimeout
import ijson
import pyarrow as pa

Prompt = Tuple[str, int]
PromptEvidence = Dict[str, Any]
PromptRecord = Tuple[int, str]
DatasetPath = Union[str, PathLike[str]]
ReadableBinary = Union[BinaryIO, gzip.GzipFile]

BUILTIN_SONNET_SOURCE = "builtin-sonnet"
SHAREGPT_ADAPTER = "sharegpt"
SHAREGPT_USER_ADAPTER = "sharegpt-user"
DOCUMENT_TEXT_ADAPTER = "document-text"
TEXT_ADAPTER = "text"
_NORMALIZED_INDEX_COLUMN = "record_index"
_NORMALIZED_PROMPT_COLUMN = "prompt"
_NORMALIZED_INDEX_VERSION = "1"
_INDEX_BATCH_SIZE = 2048
_READ_CHUNK_BYTES = 1024 * 1024
_CONTENT_HASH_NAME = re.compile(r"[0-9a-f]{64}\Z")
_SHAREGPT_USER_ROLES = ("human", "user")
_NORMALIZED_ARROW_SCHEMA = pa.schema(
    [
        pa.field(_NORMALIZED_INDEX_COLUMN, pa.int64(), nullable=False),
        pa.field(_NORMALIZED_PROMPT_COLUMN, pa.string(), nullable=False),
    ]
)
_BUILTIN_SONNET_INSTRUCTION = (
    "Randomly stream lines from the following text with "
    "{output_tokens} output tokens. Don't generate eos tokens:\n\n"
)


class IndexedPromptRecords(Protocol):
    """Minimal indexed corpus capability used by deterministic prompt selection."""

    def __len__(self) -> int: ...

    def __getitem__(self, position: int) -> PromptRecord: ...


PromptRecords = Union[Tuple[PromptRecord, ...], IndexedPromptRecords]


@dataclass(frozen=True)
class HuggingFacePromptRecords:
    """Lazy row access over a normalized, Arrow-backed Hugging Face Dataset."""

    dataset: Dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, position: int) -> PromptRecord:
        row = self.dataset[position]
        if not isinstance(row, Mapping):
            raise ValueError("Normalized prompt dataset returned a non-record row")
        raw_record_index = row.get(_NORMALIZED_INDEX_COLUMN)
        raw_prompt = row.get(_NORMALIZED_PROMPT_COLUMN)
        if isinstance(raw_record_index, bool) or not isinstance(raw_record_index, int):
            raise ValueError("Normalized prompt dataset has an invalid record index")
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise ValueError("Normalized prompt dataset has an invalid prompt")
        return raw_record_index, raw_prompt


@dataclass(frozen=True)
class PromptDataset:
    """Normalized text records from either the bundled or an external source."""

    source: str
    records: PromptRecords
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
    load: Callable[["PromptDatasetSource"], PromptDataset]
    file_formats: Tuple[str, ...] = ()
    default_file_format: Optional[str] = None
    fixed_prompt_mode: Optional[str] = None


@dataclass(frozen=True)
class PromptDatasetSource:
    """Explicit adapter selection and optional external artifact location."""

    adapter: str
    path: Optional[Path] = None
    filename: Optional[str] = None

    def __post_init__(self) -> None:
        adapter = get_prompt_dataset_adapter(self.adapter)
        normalized_path = (
            Path(self.path).expanduser() if self.path is not None else None
        )
        if adapter.requires_path and normalized_path is None:
            raise ValueError(f"Dataset adapter {self.adapter!r} requires a path")
        if not adapter.requires_path and normalized_path is not None:
            raise ValueError(f"Dataset adapter {self.adapter!r} does not accept a path")
        normalized_filename = self.filename.strip() if self.filename else None
        if not adapter.requires_path and normalized_filename is not None:
            raise ValueError(
                f"Dataset adapter {self.adapter!r} does not accept a filename"
            )
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "filename", normalized_filename)

    @classmethod
    def builtin_sonnet(cls) -> "PromptDatasetSource":
        return cls(adapter=BUILTIN_SONNET_SOURCE)

    @classmethod
    def external(
        cls, adapter: str, path: DatasetPath, filename: Optional[str] = None
    ) -> "PromptDatasetSource":
        validate_external_dataset_adapter(adapter)
        dataset_path = Path(path)
        return cls(adapter=adapter, path=dataset_path, filename=filename)

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


def _required_dataset_path(source: PromptDatasetSource) -> Path:
    if source.path is None:
        raise ValueError(f"Dataset adapter {source.adapter!r} requires a dataset path")
    return source.path


def _logical_filename(source: PromptDatasetSource, dataset_path: Path) -> str:
    return source.filename or dataset_path.name


def _file_format(source: PromptDatasetSource, adapter: PromptDatasetAdapter) -> str:
    dataset_path = _required_dataset_path(source)
    logical_filename = _logical_filename(source, dataset_path).lower()
    detected_format: Optional[str] = None
    if logical_filename.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
        detected_format = "json"
    elif logical_filename.endswith((".parquet", ".parquet.gz")):
        detected_format = "parquet"
    elif logical_filename.endswith(".arrow"):
        detected_format = "arrow"
    elif logical_filename.endswith((".txt", ".text", ".txt.gz", ".text.gz")):
        detected_format = "text"
    selected_format = detected_format or adapter.default_file_format
    if selected_format is None or selected_format not in adapter.file_formats:
        supported_formats = ", ".join(adapter.file_formats)
        raise ValueError(
            f"Dataset adapter {adapter.name!r} cannot read file "
            f"{_logical_filename(source, dataset_path)!r}; supported formats: "
            f"{supported_formats}"
        )
    return selected_format


def _datasets_cache_path() -> Path:
    configured_directory = os.environ.get("HF_DATASETS_CACHE")
    selected_directory = (
        Path(configured_directory)
        if configured_directory
        else Path(datasets_config.HF_DATASETS_CACHE)
    )
    return selected_directory.expanduser()


def _datasets_cache_directory() -> str:
    return str(_datasets_cache_path())


def _json_binary_stream(
    dataset_path: Path, logical_filename: str
) -> ContextManager[ReadableBinary]:
    if logical_filename.lower().endswith(".gz"):
        return gzip.open(dataset_path, "rb")
    return dataset_path.open("rb")


def _is_json_array(dataset_path: Path, logical_filename: str) -> bool:
    try:
        with _json_binary_stream(dataset_path, logical_filename) as stream:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return False
                content = chunk.lstrip(b"\xef\xbb\xbf \t\r\n")
                if content:
                    return content.startswith(b"[")
    except OSError as exc:
        raise ValueError(f"Unable to inspect dataset {dataset_path}: {exc}") from exc


def _artifact_digest(dataset_path: Path) -> str:
    resolved_path = dataset_path.resolve()
    if _CONTENT_HASH_NAME.fullmatch(resolved_path.name):
        return resolved_path.name
    digest = hashlib.sha256()
    try:
        with resolved_path.open("rb") as stream:
            while True:
                chunk = stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Unable to hash dataset {resolved_path}: {exc}") from exc
    return digest.hexdigest()


def _normalized_index_path(dataset_path: Path, adapter: PromptDatasetAdapter) -> Path:
    artifact_digest = _artifact_digest(dataset_path)
    cache_identity = f"{_NORMALIZED_INDEX_VERSION}\0{adapter.name}\0{artifact_digest}"
    index_digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
    index_directory = _datasets_cache_path() / "llmperf_prompt_indexes"
    try:
        index_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Unable to create dataset index cache {index_directory}: {exc}"
        ) from exc
    return index_directory / f"{index_digest}.arrow"


def _first_sharegpt_prompt(
    raw_conversation: object, accepted_roles: Optional[Tuple[str, ...]]
) -> Optional[str]:
    if not isinstance(raw_conversation, list) or not raw_conversation:
        return None
    first_turn = raw_conversation[0]
    if not isinstance(first_turn, Mapping):
        return None
    if accepted_roles is not None:
        raw_role = first_turn.get("from")
        if not isinstance(raw_role, str):
            return None
        normalized_role = raw_role.strip().lower()
        if normalized_role not in accepted_roles:
            return None
    raw_prompt = first_turn.get("value")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        return None
    return raw_prompt


def _sharegpt_record_prompt(
    raw_record: object, accepted_roles: Optional[Tuple[str, ...]]
) -> Optional[str]:
    if not isinstance(raw_record, Mapping):
        return None
    return _first_sharegpt_prompt(raw_record.get("conversations"), accepted_roles)


def _normalize_sharegpt_batch(
    batch: Mapping[str, List[object]],
    record_indices: List[int],
    accepted_roles: Optional[Tuple[str, ...]],
) -> Dict[str, List[Union[int, str]]]:
    raw_conversations = batch.get("conversations")
    if not isinstance(raw_conversations, list):
        return {_NORMALIZED_INDEX_COLUMN: [], _NORMALIZED_PROMPT_COLUMN: []}
    normalized_indices: List[Union[int, str]] = []
    normalized_prompts: List[Union[int, str]] = []
    for record_index, raw_conversation in zip(
        record_indices, raw_conversations, strict=True
    ):
        prompt = _first_sharegpt_prompt(raw_conversation, accepted_roles)
        if prompt is None:
            continue
        normalized_indices.append(record_index)
        normalized_prompts.append(prompt)
    return {
        _NORMALIZED_INDEX_COLUMN: normalized_indices,
        _NORMALIZED_PROMPT_COLUMN: normalized_prompts,
    }


def _normalize_text_batch(
    batch: Mapping[str, List[object]], record_indices: List[int]
) -> Dict[str, List[Union[int, str]]]:
    raw_lines = batch.get("text")
    if not isinstance(raw_lines, list):
        return {_NORMALIZED_INDEX_COLUMN: [], _NORMALIZED_PROMPT_COLUMN: []}
    normalized_indices: List[Union[int, str]] = []
    normalized_prompts: List[Union[int, str]] = []
    for record_index, raw_line in zip(record_indices, raw_lines, strict=True):
        if not isinstance(raw_line, str) or not raw_line.strip():
            continue
        normalized_line = raw_line.lstrip("\ufeff") if record_index == 0 else raw_line
        if not normalized_line.strip():
            continue
        normalized_indices.append(record_index)
        normalized_prompts.append(normalized_line)
    return {
        _NORMALIZED_INDEX_COLUMN: normalized_indices,
        _NORMALIZED_PROMPT_COLUMN: normalized_prompts,
    }


def _load_indexed_dataset(
    source: PromptDatasetSource,
    adapter: PromptDatasetAdapter,
    normalize: Callable[..., Dict[str, List[Union[int, str]]]],
    normalize_arguments: Optional[Dict[str, object]] = None,
) -> Dataset:
    dataset_path = _required_dataset_path(source)
    selected_format = _file_format(source, adapter)
    try:
        disable_progress_bars()
        source_dataset = load_dataset(
            selected_format,
            data_files={"train": str(dataset_path)},
            split="train",
            cache_dir=_datasets_cache_directory(),
            keep_in_memory=False,
        )
        normalized_dataset = source_dataset.map(
            normalize,
            batched=True,
            with_indices=True,
            remove_columns=source_dataset.column_names,
            fn_kwargs=normalize_arguments,
            desc=f"Indexing {adapter.name} prompts",
        )
    except (DatasetGenerationError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to index {adapter.name} dataset {dataset_path}: {exc}"
        ) from exc
    return normalized_dataset


def _write_normalized_batch(
    writer: pa.RecordBatchStreamWriter,
    record_indices: List[int],
    prompts: List[str],
) -> None:
    if not record_indices:
        return
    record_batch = pa.record_batch(
        [
            pa.array(record_indices, type=pa.int64()),
            pa.array(prompts, type=pa.string()),
        ],
        schema=_NORMALIZED_ARROW_SCHEMA,
    )
    writer.write_batch(record_batch)


def _materialize_sharegpt_json(
    dataset_path: Path,
    logical_filename: str,
    index_path: Path,
    accepted_roles: Optional[Tuple[str, ...]],
    top_level_array: bool,
) -> None:
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{index_path.stem}.",
        suffix=".incomplete",
        dir=index_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        record_indices: List[int] = []
        prompts: List[str] = []
        with _json_binary_stream(dataset_path, logical_filename) as source_stream:
            prefix = source_stream.read(len(codecs.BOM_UTF8))
            if prefix != codecs.BOM_UTF8:
                source_stream.seek(0)
            with pa.OSFile(str(temporary_path), "wb") as arrow_stream:
                with pa.ipc.new_stream(
                    arrow_stream, _NORMALIZED_ARROW_SCHEMA
                ) as writer:
                    record_prefix = "item" if top_level_array else ""
                    records = ijson.items(
                        source_stream,
                        record_prefix,
                        multiple_values=not top_level_array,
                    )
                    for record_index, raw_record in enumerate(records):
                        prompt = _sharegpt_record_prompt(raw_record, accepted_roles)
                        if prompt is None:
                            continue
                        record_indices.append(record_index)
                        prompts.append(prompt)
                        if len(record_indices) == _INDEX_BATCH_SIZE:
                            _write_normalized_batch(writer, record_indices, prompts)
                            record_indices.clear()
                            prompts.clear()
                    _write_normalized_batch(writer, record_indices, prompts)
        os.replace(temporary_path, index_path)
    except (OSError, ijson.JSONError, pa.ArrowException, ValueError) as exc:
        raise ValueError(
            f"Unable to stream-index ShareGPT JSON dataset {dataset_path}: {exc}"
        ) from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_sharegpt_json_index(
    source: PromptDatasetSource,
    adapter: PromptDatasetAdapter,
    accepted_roles: Optional[Tuple[str, ...]],
) -> Dataset:
    dataset_path = _required_dataset_path(source)
    logical_filename = _logical_filename(source, dataset_path)
    top_level_array = _is_json_array(dataset_path, logical_filename)
    index_path = _normalized_index_path(dataset_path, adapter)
    lock_path = index_path.with_suffix(".lock")
    try:
        with FileLock(lock_path, timeout=3600):
            if index_path.is_file():
                try:
                    return Dataset.from_file(str(index_path))
                except (OSError, pa.ArrowException):
                    pass
            _materialize_sharegpt_json(
                dataset_path,
                logical_filename,
                index_path,
                accepted_roles,
                top_level_array,
            )
            return Dataset.from_file(str(index_path))
    except FileLockTimeout as exc:
        raise ValueError(f"Timed out waiting for dataset index {index_path}") from exc
    except (OSError, pa.ArrowException) as exc:
        raise ValueError(f"Unable to open dataset index {index_path}: {exc}") from exc


def _load_sharegpt_source(
    source: PromptDatasetSource,
    adapter_name: str,
    accepted_roles: Optional[Tuple[str, ...]],
) -> PromptDataset:
    adapter = get_prompt_dataset_adapter(adapter_name)
    dataset_path = _required_dataset_path(source)
    selected_format = _file_format(source, adapter)
    if selected_format == "json":
        normalized_dataset = _load_sharegpt_json_index(source, adapter, accepted_roles)
    else:
        normalized_dataset = _load_indexed_dataset(
            source,
            adapter,
            _normalize_sharegpt_batch,
            {"accepted_roles": accepted_roles},
        )
    if not normalized_dataset:
        role_label = "user-first " if accepted_roles is not None else ""
        raise ValueError(
            f"ShareGPT dataset has no usable {role_label}first-turn prompts"
        )
    return PromptDataset(
        source=adapter_name,
        records=HuggingFacePromptRecords(normalized_dataset),
        separator="\n\n",
    )


def _load_sharegpt(source: PromptDatasetSource) -> PromptDataset:
    return _load_sharegpt_source(source, SHAREGPT_ADAPTER, None)


def _load_sharegpt_user(source: PromptDatasetSource) -> PromptDataset:
    return _load_sharegpt_source(source, SHAREGPT_USER_ADAPTER, _SHAREGPT_USER_ROLES)


def _load_text(source: PromptDatasetSource) -> PromptDataset:
    adapter = get_prompt_dataset_adapter(TEXT_ADAPTER)
    normalized_dataset = _load_indexed_dataset(source, adapter, _normalize_text_batch)
    if not normalized_dataset:
        raise ValueError("Text dataset has no usable non-empty line prompts")
    return PromptDataset(
        source=TEXT_ADAPTER,
        records=HuggingFacePromptRecords(normalized_dataset),
        separator="\n\n",
    )


def _load_document_text(source: PromptDatasetSource) -> PromptDataset:
    adapter = get_prompt_dataset_adapter(DOCUMENT_TEXT_ADAPTER)
    normalized_dataset = _load_indexed_dataset(source, adapter, _normalize_text_batch)
    if not normalized_dataset:
        raise ValueError("Document-text dataset has no usable non-empty documents")
    return PromptDataset(
        source=DOCUMENT_TEXT_ADAPTER,
        records=HuggingFacePromptRecords(normalized_dataset),
        separator="\n\n",
    )


def _load_sonnet_adapter(source: PromptDatasetSource) -> PromptDataset:
    if source.path is not None:
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
        file_formats=("json", "parquet", "arrow"),
        default_file_format="json",
    ),
    SHAREGPT_USER_ADAPTER: PromptDatasetAdapter(
        name=SHAREGPT_USER_ADAPTER,
        requires_path=True,
        load=_load_sharegpt_user,
        file_formats=("json", "parquet", "arrow"),
        default_file_format="json",
    ),
    DOCUMENT_TEXT_ADAPTER: PromptDatasetAdapter(
        name=DOCUMENT_TEXT_ADAPTER,
        requires_path=True,
        load=_load_document_text,
        file_formats=("parquet", "arrow"),
        default_file_format="parquet",
    ),
    TEXT_ADAPTER: PromptDatasetAdapter(
        name=TEXT_ADAPTER,
        requires_path=True,
        load=_load_text,
        file_formats=("text",),
        default_file_format="text",
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

    return get_prompt_dataset_adapter(source.adapter).load(source)


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
        self.order = list(range(len(dataset.records)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.cycle = 0

    def next(self) -> Tuple[int, str, int]:
        if self.cursor == len(self.order):
            self.cycle += 1
            self.order = list(range(len(self.dataset.records)))
            self.rng.shuffle(self.order)
            self.cursor = 0
        record_position = self.order[self.cursor]
        record_index, text = self.dataset.records[record_position]
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
    candidate_positions = list(range(len(dataset.records)))
    rng.shuffle(candidate_positions)
    unique_request_count = math.ceil(num_requests / repeat_count)
    sampled = []
    for record_position in candidate_positions:
        record_index, prompt = dataset.records[record_position]
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

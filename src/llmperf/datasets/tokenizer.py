"""Tokenizer loading and process-local caching."""

from functools import lru_cache
import os
import pathlib

from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEFAULT_TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"
TOKENIZER_PATH = "LLMPERF_TOKENIZER_PATH"
TOKENIZER_FAST = "LLMPERF_TOKENIZER_FAST"
TOKENIZERS_BACKEND_CLASS_ERROR = (
    "Tokenizer class TokenizersBackend does not exist or is not currently imported"
)


def _load_tokenizer(pretrained_model_name_or_path, **load_options):
    """Load new generic tokenizer metadata on Transformers 4 or 5."""

    try:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **load_options
        )
    except ValueError as exc:
        if not load_options.get(
            "use_fast", True
        ) or TOKENIZERS_BACKEND_CLASS_ERROR not in str(exc):
            raise
        fallback_options = dict(load_options)
        fallback_options.pop("use_fast", None)
        fallback_options["extra_special_tokens"] = {}
        return PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path, **fallback_options
        )


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load one tokenizer per process, preferring an explicit local directory."""

    configured_path = os.environ.get(TOKENIZER_PATH)
    if configured_path:
        tokenizer_path = pathlib.Path(configured_path).expanduser()
        if not tokenizer_path.is_dir():
            raise ValueError(
                f"{TOKENIZER_PATH} must point to a tokenizer directory: "
                f"{tokenizer_path}"
            )
        load_options = {"local_files_only": True}
        configured_use_fast = os.environ.get(TOKENIZER_FAST)
        if configured_use_fast is not None:
            load_options["use_fast"] = configured_use_fast.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        return _load_tokenizer(str(tokenizer_path), **load_options)
    return _load_tokenizer(DEFAULT_TOKENIZER_ID)

import json
from functools import lru_cache
import os
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llmperf.version import PROTOCOL_VERSION

RESULTS_VERSION = PROTOCOL_VERSION
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


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

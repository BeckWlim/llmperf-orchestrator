from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from datetime import datetime


class WorkerInput(BaseModel):
    """Serializable input recipe; rendered only after acquiring a request slot."""

    input_name: str = "default"
    index: int = 0
    seed: int = 11111
    prefix_seed: Optional[int] = None
    tokens: int = 550
    token_stddev: int = 0
    prefix_tokens: int = 0
    mode: str = "sample"
    repeat_count: int = 1
    text: Optional[str] = None
    dataset_adapter: Optional[str] = None
    dataset_filename: Optional[str] = None
    calibrated: bool = False
    atomic: bool = False
    output_token_count: str = "provider"
    output_tokens: int = 150
    probe: Optional[Dict[str, Any]] = None
    probe_family: int = 0
    probe_occurrence: int = 0


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        prompt: The prompt to provide to the LLM API.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: The name of the LLM API to send the request to.
        metadata: Additional metadata to attach to the request for logging or validation purposes.
    """

    model: str
    prompt: Tuple[str, int]
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[float] = None
    preparation: Optional[WorkerInput] = None
    deadline_at: Optional[datetime] = None

import json

import pytest

from llmperf.models import RequestConfig
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIStreamError,
    decode_sse_line,
)


def _event(document):
    return b"data:" + json.dumps(document).encode("utf-8")


def test_empty_events():
    assert decode_sse_line(_event({"choices": []})) == {"kind": "metadata"}
    assert decode_sse_line(b"event: message") == {"kind": "ignore"}
    assert decode_sse_line(b": keepalive") == {"kind": "ignore"}
    assert decode_sse_line(b"data: [DONE]") == {"kind": "done"}


def test_content():
    event = decode_sse_line(
        _event(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "think ",
                            "content": "answer",
                        }
                    }
                ]
            }
        )
    )

    assert event == {"kind": "text", "text": "think answer"}


def test_error_code():
    with pytest.raises(OpenAIStreamError) as error:
        decode_sse_line(
            _event({"error": {"code": 403, "message": "model not enabled"}})
        )

    assert error.value.code == 403
    assert str(error.value) == "model not enabled"


def test_request_timeout():
    config = RequestConfig(
        model="model",
        prompt=("prompt", 1),
        timeout_seconds=29.0,
    )

    assert config.timeout_seconds == 29.0

"""Release identifier and independently versioned document formats."""

from typing import Literal

ProtocolVersion = Literal["1.0.0"]

RELEASE_VERSION = "1.0.3"
PROTOCOL_VERSION: ProtocolVersion = "1.0.0"
TASK_FORMAT_VERSION: ProtocolVersion = PROTOCOL_VERSION

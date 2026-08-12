"""Command-line entry point for the LLMPerf FastAPI backend."""

import os

import uvicorn

from llmperf.logging import configure_logging
from llmperf_backend.config import load_config


def main() -> None:
    config = load_config()
    configure_logging(
        config.server.log_level,
        color=os.environ.get("LLMPERF_LOG_COLOR", "auto").lower(),
    )
    uvicorn.run(
        "llmperf_backend.app:create_app",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        workers=config.server.workers,
        reload=config.server.reload,
        log_config=None,
    )


if __name__ == "__main__":
    main()

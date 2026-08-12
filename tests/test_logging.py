import logging

from llmperf.logging import ColorFormatter, LoggerNameFilter, RESET, route_library_logs


def test_server_alias():
    record = logging.LogRecord(
        "uvicorn.error", logging.INFO, __file__, 1, "Application startup", (), None
    )

    LoggerNameFilter().filter(record)

    assert record.name == "llmperf_backend.server"


def test_library_routing():
    logger = logging.getLogger("example.private")
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False

    route_library_logs("example.private")

    assert logger.handlers == []
    assert logger.propagate is True


def test_color_level():
    record = logging.LogRecord(
        "llmperf", logging.WARNING, __file__, 1, "Proxy retry", (), None
    )

    rendered = ColorFormatter(color=True).format(record)

    assert "\033[33mWARNING" in rendered
    assert f"WARNING{RESET}" in rendered
    assert rendered.endswith("llmperf: Proxy retry")


def test_plain_level():
    record = logging.LogRecord(
        "llmperf", logging.ERROR, __file__, 1, "Proxy failed", (), None
    )

    rendered = ColorFormatter(color=False).format(record)

    assert "\033[" not in rendered
    assert " ERROR llmperf: Proxy failed" in rendered

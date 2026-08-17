from unittest.mock import Mock

from llmperf_backend.huggingface import configure_huggingface_http


def test_http_proxy(monkeypatch):
    configure = Mock()
    monkeypatch.setattr(
        "llmperf_backend.huggingface.configure_http_backend", configure
    )

    configure_huggingface_http("http://proxy.internal:3128")

    factory = configure.call_args.kwargs["backend_factory"]
    session = factory()
    assert session.proxies == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }


def test_http_default(monkeypatch):
    configure = Mock()
    monkeypatch.setattr(
        "llmperf_backend.huggingface.configure_http_backend", configure
    )

    configure_huggingface_http(None)

    configure.assert_called_once_with()

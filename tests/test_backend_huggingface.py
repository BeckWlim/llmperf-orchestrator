from unittest.mock import Mock

from llmperf_backend.huggingface import configure_huggingface_http


def test_http_proxy(monkeypatch):
    configure = Mock()
    client = Mock()
    client_class = Mock(return_value=client)
    monkeypatch.setattr("llmperf_backend.huggingface.set_client_factory", configure)
    monkeypatch.setattr("llmperf_backend.huggingface.httpx.Client", client_class)

    configure_huggingface_http("http://proxy.internal:3128")

    factory = configure.call_args.args[0]
    assert factory() is client
    client_class.assert_called_once_with(
        follow_redirects=True,
        timeout=None,
        proxy="http://proxy.internal:3128",
    )


def test_http_default(monkeypatch):
    configure = Mock()
    client = Mock()
    client_class = Mock(return_value=client)
    monkeypatch.setattr("llmperf_backend.huggingface.set_client_factory", configure)
    monkeypatch.setattr("llmperf_backend.huggingface.httpx.Client", client_class)

    configure_huggingface_http(None)

    factory = configure.call_args.args[0]
    assert factory() is client
    client_class.assert_called_once_with(follow_redirects=True, timeout=None)

from unittest.mock import Mock

import pytest

from llmperf import utils


@pytest.fixture(autouse=True)
def clear_tokenizer_cache():
    utils.get_tokenizer.cache_clear()
    yield
    utils.get_tokenizer.cache_clear()


def test_local_only(tmp_path, monkeypatch):
    tokenizer = object()
    loader = Mock(return_value=tokenizer)
    monkeypatch.setenv(utils.TOKENIZER_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained", loader)

    assert utils.get_tokenizer() is tokenizer
    assert utils.get_tokenizer() is tokenizer

    loader.assert_called_once_with(str(tmp_path), local_files_only=True)


def test_bad_local_path(tmp_path, monkeypatch):
    missing = tmp_path / "missing-tokenizer"
    loader = Mock()
    monkeypatch.setenv(utils.TOKENIZER_PATH_ENV, str(missing))
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained", loader)

    with pytest.raises(ValueError, match=utils.TOKENIZER_PATH_ENV):
        utils.get_tokenizer()

    loader.assert_not_called()


def test_default_id(monkeypatch):
    tokenizer = object()
    loader = Mock(return_value=tokenizer)
    monkeypatch.delenv(utils.TOKENIZER_PATH_ENV, raising=False)
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained", loader)

    assert utils.get_tokenizer() is tokenizer

    loader.assert_called_once_with(utils.DEFAULT_TOKENIZER_ID)


def test_fast_mode(tmp_path, monkeypatch):
    tokenizer = object()
    loader = Mock(return_value=tokenizer)
    monkeypatch.setenv(utils.TOKENIZER_PATH_ENV, str(tmp_path))
    monkeypatch.setenv(utils.TOKENIZER_USE_FAST_ENV, "false")
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained", loader)

    assert utils.get_tokenizer() is tokenizer

    loader.assert_called_once_with(str(tmp_path), local_files_only=True, use_fast=False)

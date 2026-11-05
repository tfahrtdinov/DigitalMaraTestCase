import pytest
import requests

import loggers


@pytest.fixture(autouse=True)
def fresh():
    loggers.tracking_uri.cache_clear()
    yield
    loggers.tracking_uri.cache_clear()


def test_no_uri_uses_local_server(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    response = requests.Response()
    response.status_code = 200
    monkeypatch.setattr(requests, "get", lambda *a, **k: response)
    assert loggers.tracking_uri() == loggers.LOCAL_SERVER


def test_server_that_answers_is_used(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://server:5000")
    response = requests.Response()
    response.status_code = 200
    monkeypatch.setattr(requests, "get", lambda *a, **k: response)
    assert loggers.tracking_uri() == "http://server:5000"


def test_silent_server_falls_back_to_local_store(monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://server:5000")

    def refuse(*a, **k):
        raise requests.ConnectionError

    monkeypatch.setattr(requests, "get", refuse)
    assert loggers.tracking_uri() == loggers.LOCAL_STORE
    assert "does not answer" in capsys.readouterr().out

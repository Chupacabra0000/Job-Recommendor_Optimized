import importlib

import pytest
import requests


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


def test__get_returns_json_on_success(monkeypatch):
    import hh_client

    def fake_get(url, params, headers, timeout):
        assert url == "https://example.com/api"
        assert params == {"q": "python"}
        assert timeout == 12
        assert headers["User-Agent"]
        return DummyResponse(status_code=200, json_data={"ok": True})

    monkeypatch.setattr(hh_client.requests, "get", fake_get)
    result = hh_client._get("https://example.com/api", params={"q": "python"}, timeout=12)
    assert result == {"ok": True}


def test__get_retries_on_429_then_succeeds(monkeypatch):
    import hh_client

    calls = {"count": 0}
    sleeps = []

    def fake_get(url, params, headers, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResponse(status_code=429, text="rate limited", headers={"Retry-After": "0.1"})
        return DummyResponse(status_code=200, json_data={"items": [1, 2, 3]})

    monkeypatch.setattr(hh_client.requests, "get", fake_get)
    monkeypatch.setattr(hh_client.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(hh_client.random, "random", lambda: 0.5)

    result = hh_client._get("https://example.com/api")
    assert result == {"items": [1, 2, 3]}
    assert calls["count"] == 2
    assert len(sleeps) == 1


def test__get_raises_on_non_retryable_http_error(monkeypatch):
    import hh_client

    monkeypatch.setattr(
        hh_client.requests,
        "get",
        lambda url, params, headers, timeout: DummyResponse(status_code=404, text="not found"),
    )

    with pytest.raises(RuntimeError, match="HH API request failed"):
        hh_client._get("https://example.com/api")


def test__get_raises_after_retry_exhausted(monkeypatch):
    import hh_client

    monkeypatch.setenv("HH_MAX_RETRIES", "2")
    importlib.reload(hh_client)

    monkeypatch.setattr(hh_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(hh_client.random, "random", lambda: 0.5)

    def fake_get(url, params, headers, timeout):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(hh_client.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        hh_client._get("https://example.com/api")


def test_search_vacancies_builds_params(monkeypatch):
    import hh_client

    captured = {}

    def fake__get(url, params=None, timeout=30):
        captured["url"] = url
        captured["params"] = params
        return {"items": []}

    monkeypatch.setattr(hh_client, "_get", fake__get)

    hh_client.search_vacancies(
        text="python developer",
        area=1,
        page=2,
        per_page=20,
        period_days=7,
        order_by="publication_time",
    )

    assert captured["url"].endswith("/vacancies")
    assert captured["params"] == {
        "page": 2,
        "per_page": 20,
        "text": "python developer",
        "area": 1,
        "period": 7,
        "order_by": "publication_time",
    }


def test_fetch_vacancies_paginates_and_truncates(monkeypatch):
    import hh_client

    payloads = [
        {"items": [{"id": 1}, {"id": 2}], "pages": 3},
        {"items": [{"id": 3}, {"id": 4}], "pages": 3},
        {"items": [{"id": 5}], "pages": 3},
    ]

    monkeypatch.setattr(hh_client, "search_vacancies", lambda **kwargs: payloads[kwargs["page"]])
    monkeypatch.setattr(hh_client.time, "sleep", lambda s: None)

    result = hh_client.fetch_vacancies(max_items=4, sleep_s=0)
    assert result == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]


def test_vacancy_details_calls_expected_endpoint(monkeypatch):
    import hh_client

    captured = {}

    def fake__get(url, params=None, timeout=30):
        captured["url"] = url
        return {"id": "123"}

    monkeypatch.setattr(hh_client, "_get", fake__get)
    result = hh_client.vacancy_details("123")

    assert result == {"id": "123"}
    assert captured["url"].endswith("/vacancies/123")

import importlib
import pytest


class DummyResponse:
    def __init__(self, status_code, payload=None, headers=None, text=''):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def test__get_raises_immediately_on_non_retryable_http_error(monkeypatch):
    hh_client = importlib.import_module('hh_client')
    monkeypatch.setattr(
        hh_client.requests,
        'get',
        lambda *args, **kwargs: DummyResponse(400, text='bad request'),
    )
    monkeypatch.setattr(hh_client.time, 'sleep', lambda *_: None)

    with pytest.raises(RuntimeError, match='HH API request failed'):
        hh_client._get('https://example.test')


def test_search_vacancies_builds_expected_params(monkeypatch):
    hh_client = importlib.import_module('hh_client')
    seen = {}

    def fake_get(url, params=None, timeout=30):
        seen['url'] = url
        seen['params'] = params
        return {'items': []}

    monkeypatch.setattr(hh_client, '_get', fake_get)
    hh_client.search_vacancies(text='python', area=113, page=2, per_page=25, period_days=3, order_by='publication_time')

    assert seen['url'].endswith('/vacancies')
    assert seen['params'] == {
        'page': 2,
        'per_page': 25,
        'text': 'python',
        'area': 113,
        'period': 3,
        'order_by': 'publication_time',
    }

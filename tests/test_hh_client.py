import importlib


class DummyResponse:
    def __init__(self, status_code, payload=None, headers=None, text=''):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def test__get_retries_on_429_then_returns_json(monkeypatch):
    hh_client = importlib.import_module('hh_client')
    calls = {'n': 0}

    def fake_get(url, params=None, headers=None, timeout=30):
        calls['n'] += 1
        if calls['n'] == 1:
            return DummyResponse(429, headers={'Retry-After': '0'})
        return DummyResponse(200, payload={'ok': True, 'params': params})

    monkeypatch.setattr(hh_client.requests, 'get', fake_get)
    monkeypatch.setattr(hh_client.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(hh_client.random, 'random', lambda: 0.0)

    payload = hh_client._get('https://example.test', params={'q': 'python'})
    assert payload['ok'] is True
    assert calls['n'] == 2


def test_fetch_vacancies_paginates_and_truncates(monkeypatch):
    hh_client = importlib.import_module('hh_client')
    responses = {
        0: {'items': [{'id': '1'}, {'id': '2'}], 'pages': 3},
        1: {'items': [{'id': '3'}, {'id': '4'}], 'pages': 3},
        2: {'items': [{'id': '5'}], 'pages': 3},
    }

    def fake_search_vacancies(**kwargs):
        return responses[kwargs['page']]

    monkeypatch.setattr(hh_client, 'search_vacancies', fake_search_vacancies)
    monkeypatch.setattr(hh_client.time, 'sleep', lambda *_: None)

    items = hh_client.fetch_vacancies(text='ml', area=1, max_items=4, per_page=2)
    assert [item['id'] for item in items] == ['1', '2', '3', '4']

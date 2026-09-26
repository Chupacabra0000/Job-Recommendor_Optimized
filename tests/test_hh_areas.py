import importlib
import json
from pathlib import Path


class DummyResponse:
    def __init__(self, status_code, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


TREE = [
    {
        'id': '113',
        'name': 'Россия',
        'areas': [
            {'id': '1', 'name': 'Москва', 'areas': [{'id': '2', 'name': 'Зеленоград', 'areas': []}]},
            {'id': '3', 'name': 'Татарстан', 'areas': [{'id': '4', 'name': 'Казань', 'areas': []}]},
        ],
    }
]


def test_fetch_areas_tree_uses_fresh_cache(monkeypatch, tmp_path):
    hh_areas = importlib.import_module('hh_areas')
    cache_path = tmp_path / 'hh_areas_cache.json'
    monkeypatch.setattr(hh_areas, 'CACHE_PATH', str(cache_path))
    cache_path.write_text(json.dumps({'_cached_at': 99999999999, 'tree': TREE}, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr(hh_areas.requests, 'get', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('network not expected')))
    assert hh_areas.fetch_areas_tree() == TREE


def test_fetch_areas_tree_falls_back_to_stale_cache(monkeypatch, tmp_path):
    hh_areas = importlib.import_module('hh_areas')
    cache_path = tmp_path / 'hh_areas_cache.json'
    monkeypatch.setattr(hh_areas, 'CACHE_PATH', str(cache_path))
    cache_path.write_text(json.dumps({'_cached_at': 1, 'tree': TREE}, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr(hh_areas.time, 'time', lambda: hh_areas.CACHE_TTL_SECONDS + 100)
    monkeypatch.setattr(hh_areas.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(hh_areas.requests, 'get', lambda *args, **kwargs: DummyResponse(500, text='boom'))
    assert hh_areas.fetch_areas_tree() == TREE


def test_list_regions_and_cities_collects_nested_leaves():
    hh_areas = importlib.import_module('hh_areas')
    tree = [
        {
            'name': 'Россия',
            'areas': [
                {
                    'id': '10',
                    'name': 'Регион',
                    'areas': [
                        {'id': '11', 'name': 'Подрегион', 'areas': [{'id': '12', 'name': 'Город', 'areas': []}]}
                    ],
                }
            ],
        }
    ]
    regions, cities_by_region = hh_areas.list_regions_and_cities(tree)
    assert regions == [{'id': '10', 'name': 'Регион'}]
    assert cities_by_region['10'] == [{'id': '12', 'name': 'Город'}]

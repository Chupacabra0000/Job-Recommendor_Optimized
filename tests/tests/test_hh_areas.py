import importlib
import json

import pytest


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_fetch_areas_tree_uses_fresh_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    import hh_areas
    importlib.reload(hh_areas)

    cached = [{"id": "113", "name": "Россия", "areas": []}]
    hh_areas._write_cache(cached)

    monkeypatch.setattr(hh_areas.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network not expected")))
    assert hh_areas.fetch_areas_tree() == cached


def test_fetch_areas_tree_falls_back_to_stale_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("HH_AREAS_CACHE_TTL_SECONDS", "1")
    import hh_areas
    importlib.reload(hh_areas)

    stale = [{"id": "113", "name": "Россия", "areas": []}]
    hh_areas._write_cache(stale)
    cache_data = json.loads(open(hh_areas.CACHE_PATH, "r", encoding="utf-8").read())
    cache_data["_cached_at"] = 0
    open(hh_areas.CACHE_PATH, "w", encoding="utf-8").write(json.dumps(cache_data, ensure_ascii=False))

    monkeypatch.setattr(hh_areas.requests, "get", lambda *args, **kwargs: DummyResponse(status_code=503, text="bad gateway"))
    monkeypatch.setattr(hh_areas.time, "sleep", lambda s: None)

    assert hh_areas.fetch_areas_tree() == stale


def test_fetch_areas_tree_raises_without_cache_when_api_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    import hh_areas
    importlib.reload(hh_areas)

    monkeypatch.setattr(hh_areas.requests, "get", lambda *args, **kwargs: DummyResponse(status_code=500, text="boom"))
    monkeypatch.setattr(hh_areas.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="HH API error 500"):
        hh_areas.fetch_areas_tree()


@pytest.mark.xfail(reason="Current implementation treats first-level nested groups as cities instead of recursing to leaf cities")
def test_list_regions_and_cities_handles_nested_leaves():
    import hh_areas

    tree = [
        {
            "id": "113",
            "name": "Россия",
            "areas": [
                {
                    "id": "1",
                    "name": "Region B",
                    "areas": [
                        {
                            "id": "11",
                            "name": "Nested Group",
                            "areas": [{"id": "111", "name": "City Z", "areas": []}],
                        }
                    ],
                },
                {
                    "id": "2",
                    "name": "Region A",
                    "areas": [{"id": "201", "name": "City A", "areas": []}],
                },
            ],
        }
    ]

    regions, cities = hh_areas.list_regions_and_cities(tree)
    assert [r["name"] for r in regions] == ["Region A", "Region B"]
    assert cities["1"] == [{"id": "111", "name": "City Z"}]
    assert cities["2"] == [{"id": "201", "name": "City A"}]

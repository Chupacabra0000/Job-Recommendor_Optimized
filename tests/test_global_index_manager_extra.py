import importlib
import sys
import types
import numpy as np


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for text in texts:
            s = str(text).lower()
            out.append([
                1.0 if 'python' in s else 0.0,
                1.0 if 'sql' in s else 0.0,
                float(len(s.split())) or 1.0,
            ])
        arr = np.asarray(out, dtype=np.float32)
        denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / denom


def install_fake_sentence_transformers():
    sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)


def test_ensure_vectors_for_ids_and_load_vectors(monkeypatch):
    install_fake_sentence_transformers()
    gim = importlib.import_module('global_index_manager')

    model = FakeSentenceTransformer('fake')
    ids = np.array([11, 22], dtype=np.int64)
    gim._ensure_vectors_for_ids(model, ids, {11: 'python backend', 22: 'sql analytics'}, dim=3)
    vecs = gim._load_vectors_for_ids(ids)

    assert vecs.shape == (2, 3)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(vecs[1]), 1.0, atol=1e-6)


def test_refresh_global_index_short_circuits_when_recent(monkeypatch):
    install_fake_sentence_transformers()
    gim = importlib.import_module('global_index_manager')

    monkeypatch.setattr(gim, 'init_db', lambda: None)
    monkeypatch.setattr(gim, 'get_global_index_state', lambda key: '2026-03-09T11:00:00Z' if 'last_refresh' in key else None)
    monkeypatch.setattr(gim, '_utcnow', lambda: gim.datetime(2026, 3, 9, 12, 0, tzinfo=gim.timezone.utc))

    called = {'fetch': False}
    monkeypatch.setattr(gim, 'fetch_vacancies', lambda **kwargs: called.__setitem__('fetch', True))

    did_refresh, msg = gim.refresh_global_index(gim.GlobalIndexConfig(area_id=1, period_days=7), force=False, min_hours_between_refresh=6)
    assert did_refresh is False
    assert 'fresh' in msg.lower()
    assert called['fetch'] is False

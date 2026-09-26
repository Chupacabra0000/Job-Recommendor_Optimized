import importlib
import sys
import types
import numpy as np
import pandas as pd


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
                float(len(s.split())),
            ])
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


def install_fake_sentence_transformers():
    sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)


def test_job_recommendation_system_scores_and_explains(monkeypatch, tmp_path):
    install_fake_sentence_transformers()
    model_mod = importlib.import_module('model')

    def fake_to_parquet(self, path, index=False):
        self.to_csv(path, index=index)

    def fake_read_parquet(path):
        return pd.read_csv(path)

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fake_to_parquet, raising=False)
    monkeypatch.setattr(pd, 'read_parquet', fake_read_parquet)

    jobs_csv = tmp_path / 'jobs.csv'
    pd.DataFrame([
        {'position': 'Python Developer', 'requisite_skill': 'Python SQL', 'offer_details': 'APIs'},
        {'position': 'Designer', 'requisite_skill': 'Figma', 'offer_details': 'UI'},
    ]).to_csv(jobs_csv, index=False)

    system = model_mod.JobRecommendationSystem(str(jobs_csv), model_name='fake-model')
    scored = system.score_all_jobs('Experienced Python SQL engineer')
    assert scored.loc[0, 'similarity_score'] > scored.loc[1, 'similarity_score']

    explanation = system.explain_match('Python SQL engineer', 'Python backend and SQL work', top_k=5)
    assert 'python' in explanation['matched_keywords']


def test_global_index_manager_helpers(monkeypatch):
    install_fake_sentence_transformers()
    gim = importlib.import_module('global_index_manager')

    hashed = gim._vid_to_int64('abc-123')
    assert isinstance(hashed, int)
    assert hashed == gim._vid_to_int64('abc-123')
    assert gim._vid_to_int64('42') == 42

    text = gim._job_text_from_item({
        'name': 'Python Dev',
        'employer': {'name': 'ACME'},
        'snippet': {'requirement': 'Python', 'responsibility': 'Build APIs'},
        'schedule': {'name': 'Remote'},
    })
    assert 'Python Dev' in text and 'ACME' in text and 'Remote' in text

    ids_map = gim._build_item_map([{'id': '1', 'name': 'A'}, {'id': '2', 'name': 'B'}])
    assert sorted(ids_map.keys()) == ['1', '2']

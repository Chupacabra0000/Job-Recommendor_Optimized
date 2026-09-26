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
        rows = []
        for text in texts:
            s = str(text).lower()
            rows.append([
                1.0 if 'python' in s else 0.0,
                1.0 if 'sql' in s else 0.0,
                float(len(s.split())),
            ])
        arr = np.asarray(rows, dtype=np.float32)
        denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / denom


def install_fake_sentence_transformers():
    sys.modules['sentence_transformers'] = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)


def test_normalize_rows_and_empty_resume_score(monkeypatch, tmp_path):
    install_fake_sentence_transformers()
    model_mod = importlib.import_module('model')

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, path, index=False: self.to_csv(path, index=index), raising=False)
    monkeypatch.setattr(pd, 'read_parquet', lambda path: pd.read_csv(path))

    jobs_csv = tmp_path / 'jobs.csv'
    pd.DataFrame([
        {'position': 'Python Developer', 'requisite_skill': 'Python SQL'},
        {'position': 'Designer', 'requisite_skill': 'Figma'},
    ]).to_csv(jobs_csv, index=False)

    system = model_mod.JobRecommendationSystem(str(jobs_csv), model_name='fake-model')
    scored = system.score_all_jobs('')
    assert scored['similarity_score'].isna().all()

    arr = np.array([[3.0, 4.0]], dtype=np.float32)
    normed = model_mod._normalize_rows(arr)
    np.testing.assert_allclose(normed, np.array([[0.6, 0.8]], dtype=np.float32), atol=1e-6)


def test_explain_match_handles_empty_texts():
    install_fake_sentence_transformers()
    model_mod = importlib.import_module('model')
    out = model_mod.JobRecommendationSystem.explain_match(None, '', 'job text')
    assert out == {'resume_keywords': [], 'job_keywords': [], 'matched_keywords': []}

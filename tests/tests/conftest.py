import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, batch_size=None, show_progress_bar=None, normalize_embeddings=False):
            if isinstance(texts, str):
                texts = [texts]
            rows = []
            for text in texts:
                t = (text or "").lower()
                vec = np.array([
                    1.0 if "python" in t else 0.0,
                    1.0 if "sql" in t else 0.0,
                    1.0 if "data" in t else 0.0,
                ], dtype=np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                rows.append(vec)
            return np.vstack(rows)

        def get_sentence_embedding_dimension(self):
            return 3

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return FakeSentenceTransformer


@pytest.fixture
def sample_jobs_csv(tmp_path):
    df = pd.DataFrame(
        [
            {
                "workplace": "Remote",
                "working_mode": "Full-time",
                "position": "Python Engineer",
                "job_role_and_duties": "Build APIs and pipelines",
                "requisite_skill": "python sql data",
                "offer_details": "Fast growing team",
                "salary": "1000",
            },
            {
                "workplace": "Office",
                "working_mode": "Hybrid",
                "position": "Graphic Designer",
                "job_role_and_duties": "Create visuals",
                "requisite_skill": "figma branding",
                "offer_details": "Creative team",
                "salary": "900",
            },
        ]
    )
    csv_path = Path(tmp_path) / "jobs.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def patch_parquet(monkeypatch):
    import pandas as pd

    def fake_to_parquet(self, path, index=False, *args, **kwargs):
        self.to_pickle(path)

    def fake_read_parquet(path, *args, **kwargs):
        return pd.read_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet, raising=True)
    return pd

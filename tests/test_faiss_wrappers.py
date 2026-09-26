import importlib
import sys
import types
import numpy as np


class FakeIndexFlatIP:
    def __init__(self, d):
        self.d = d


class FakeIndexIDMap2:
    def __init__(self, base):
        self.base = base
        self.vectors = None
        self.ids = None

    def add_with_ids(self, vectors, ids):
        self.vectors = vectors
        self.ids = ids

    def search(self, query_vec, top_k):
        scores = (self.vectors @ query_vec.T).reshape(-1)
        order = np.argsort(scores)[::-1][:top_k]
        return scores[order][None, :], self.ids[order][None, :]


def install_fake_faiss(storage):
    mod = types.SimpleNamespace()
    mod.IndexFlatIP = FakeIndexFlatIP
    mod.IndexIDMap2 = FakeIndexIDMap2
    def write_index(index, path):
        storage[path] = index
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'fake')
    mod.write_index = write_index
    mod.read_index = lambda path: storage[path]
    sys.modules['faiss'] = mod


def test_saved_search_faiss_wrapper_roundtrip(tmp_path, monkeypatch):
    storage = {}
    install_fake_faiss(storage)
    faiss_search_index = importlib.import_module('faiss_search_index')

    index = faiss_search_index.build_index(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([101, 202], dtype=np.int64),
    )
    faiss_search_index.save_index(7, index)
    loaded = faiss_search_index.load_index(7)
    assert loaded.ids.tolist() == [101, 202]

    faiss_search_index.delete_index_dir(7)
    assert faiss_search_index.load_index(7) is None


def test_global_faiss_search_returns_best_match(monkeypatch):
    storage = {}
    install_fake_faiss(storage)
    global_faiss_index = importlib.import_module('global_faiss_index')

    index = global_faiss_index.build_index(
        np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32),
        np.array([1, 2], dtype=np.int64),
    )
    global_faiss_index.save_index(1, 7, index, np.array([1, 2], dtype=np.int64))
    loaded, ids = global_faiss_index.load_index_and_ids(1, 7)
    scores, found_ids = global_faiss_index.search(loaded, np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)

    assert ids.tolist() == [1, 2]
    assert found_ids.tolist() == [1, 2]
    assert scores[0] >= scores[1]

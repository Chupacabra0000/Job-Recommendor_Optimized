import importlib
import sys
import types
import pickle
import numpy as np


class FakeIndexFlatIP:
    def __init__(self, d):
        self.d = d


class FakeIndexIDMap2:
    def __init__(self, base):
        self.base = base
        self.vectors = np.empty((0, base.d), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int64)

    def add_with_ids(self, vectors, ids):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)

    def search(self, query, k):
        q = np.asarray(query, dtype=np.float32)
        scores = q @ self.vectors.T
        order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        sorted_ids = self.ids[order]
        return sorted_scores, sorted_ids


def install_fake_faiss():
    def write_index(index, path):
        with open(path, 'wb') as f:
            pickle.dump(index, f)

    def read_index(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    sys.modules['faiss'] = types.SimpleNamespace(
        IndexFlatIP=FakeIndexFlatIP,
        IndexIDMap2=FakeIndexIDMap2,
        write_index=write_index,
        read_index=read_index,
    )


def test_faiss_search_index_save_load_and_delete_dir():
    install_fake_faiss()
    fsi = importlib.import_module('faiss_search_index')
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([11, 22], dtype=np.int64)
    index = fsi.build_index(vectors, ids)

    fsi.save_index(55, index)
    loaded = fsi.load_index(55)
    scores, out_ids = loaded.search(np.array([[1.0, 0.0]], dtype=np.float32), 2)
    assert out_ids[0, 0] == 11
    assert scores[0, 0] >= scores[0, 1]

    fsi.delete_index_dir(55)
    assert fsi.load_index(55) is None


def test_global_faiss_index_roundtrip_and_search():
    install_fake_faiss()
    gfi = importlib.import_module('global_faiss_index')
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([101, 202], dtype=np.int64)
    index = gfi.build_index(vectors, ids)

    gfi.save_index(1, 7, index, ids)
    idx2, ids2 = gfi.load_index_and_ids(1, 7)
    assert ids2.tolist() == [101, 202]

    scores, found = gfi.search(idx2, np.array([[0.0, 1.0]], dtype=np.float32), top_k=2)
    assert found.tolist() == [202, 101]
    assert scores[0] >= scores[1]

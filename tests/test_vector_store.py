import importlib
import numpy as np


def test_append_and_load_vectors_and_deduplicate():
    vector_store = importlib.import_module('vector_store')
    model_name = 'unit/test-model'
    vector_store.init_store(model_name, dim=3)

    ids = np.array([1, 2, 2], dtype=np.int64)
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [9.0, 9.0, 9.0],
    ], dtype=np.float32)

    all_ids, appended = vector_store.append_vectors(model_name, ids, vecs)
    assert all_ids.tolist() == [1, 2]
    assert appended.shape == (2, 3)

    ids_loaded = vector_store.load_ids(model_name)
    mm = vector_store.load_memmap(model_name)
    assert ids_loaded.tolist() == [1, 2]
    np.testing.assert_allclose(np.asarray(mm), vecs[:2])


def test_compact_store_keeps_selected_ids_in_original_order():
    vector_store = importlib.import_module('vector_store')
    model_name = 'unit/test-compact'
    vector_store.init_store(model_name, dim=2)
    vector_store.append_vectors(
        model_name,
        np.array([10, 20, 30], dtype=np.int64),
        np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )

    vector_store.compact_store(model_name, np.array([30, 10], dtype=np.int64))
    ids_loaded = vector_store.load_ids(model_name)
    mm = vector_store.load_memmap(model_name)

    assert ids_loaded.tolist() == [10, 30]
    np.testing.assert_allclose(np.asarray(mm), np.array([[1, 0], [1, 1]], dtype=np.float32))

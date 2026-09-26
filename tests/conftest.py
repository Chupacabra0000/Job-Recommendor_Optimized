import os
import sys
import importlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv('APP_DB_PATH', str(tmp_path / 'test_app.db'))
    monkeypatch.setenv('ARTIFACT_DIR', str(tmp_path / 'artifacts'))
    monkeypatch.setenv('HH_MAX_RETRIES', '3')
    monkeypatch.setenv('HH_RETRY_BASE_SLEEP', '0')
    monkeypatch.chdir(PROJECT_ROOT)

    modules_to_reload = [
        'db',
        'vector_store',
        'hh_client',
        'hh_areas',
        'faiss_search_index',
        'global_faiss_index',
        'global_index_manager',
        'search_cleanup',
        'tfidf_terms',
        'model',
    ]
    for name in modules_to_reload:
        if name in sys.modules:
            importlib.reload(sys.modules[name])

    yield

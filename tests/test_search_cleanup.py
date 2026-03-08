import importlib


def test_cleanup_helpers_delegate_to_db_and_index(monkeypatch):
    search_cleanup = importlib.import_module('search_cleanup')
    deleted_dirs = []
    deleted_resumes = []

    monkeypatch.setattr(search_cleanup, 'enforce_saved_search_limit', lambda user_id, keep_n=3: [11, 12])
    monkeypatch.setattr(search_cleanup, 'delete_saved_searches_for_resume', lambda user_id, resume_id: [21])
    monkeypatch.setattr(search_cleanup, 'delete_resume', lambda user_id, resume_id: deleted_resumes.append((user_id, resume_id)))
    monkeypatch.setattr(search_cleanup, 'delete_index_dir', lambda sid: deleted_dirs.append(sid))

    assert search_cleanup.enforce_limit_and_cleanup(7, keep_n=2) == [11, 12]
    assert deleted_dirs[:2] == [11, 12]

    assert search_cleanup.delete_resume_and_cleanup(7, 99) == [21]
    assert deleted_dirs[-1] == 21
    assert deleted_resumes == [(7, 99)]

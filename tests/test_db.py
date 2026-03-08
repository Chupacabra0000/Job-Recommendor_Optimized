import importlib


def test_user_resume_favorites_and_sessions_flow():
    db = importlib.import_module('db')
    db.init_db()

    ok, _ = db.create_user('USER@example.com', 'secret1')
    assert ok is True
    user = db.authenticate('user@example.com', 'secret1')
    assert user is not None

    resume_id = db.create_resume(user['id'], 'CV', 'Python developer')
    resumes = db.list_resumes(user['id'])
    assert resumes[0]['id'] == resume_id

    db.add_favorite(user['id'], 'vac-1')
    db.add_favorite(user['id'], 'vac-1')
    assert db.list_favorites(user['id']) == ['vac-1']
    db.remove_favorite(user['id'], 'vac-1')
    assert db.list_favorites(user['id']) == []

    token = db.create_session(user['id'], days_valid=1)
    session_user = db.get_user_by_token(token)
    assert session_user['id'] == user['id']
    db.delete_session(token)
    assert db.get_user_by_token(token) is None

    db.delete_resume(user['id'], resume_id)
    assert db.list_resumes(user['id']) == []


def test_saved_search_and_global_state_helpers():
    db = importlib.import_module('db')
    db.init_db()
    ok, _ = db.create_user('user2@example.com', 'secret1')
    assert ok is True
    user = db.authenticate('user2@example.com', 'secret1')

    sid1, created1 = db.create_or_get_saved_search(user['id'], 'rid:1', 1, 7, resume_id=1)
    sid2, created2 = db.create_or_get_saved_search(user['id'], 'rid:1', 1, 7, resume_id=1, resume_label='Updated')
    assert created1 is True
    assert created2 is False
    assert sid1 == sid2

    sid3, _ = db.create_or_get_saved_search(user['id'], 'rid:2', 1, 7, resume_id=2)
    sid4, _ = db.create_or_get_saved_search(user['id'], 'rid:3', 1, 7, resume_id=3)
    deleted = db.enforce_saved_search_limit(user['id'], keep_n=2)
    assert len(deleted) == 1
    assert deleted[0] in {sid1, sid3, sid4}

    db.upsert_saved_search_results(sid3, [
        {
            'vacancy_id': '100',
            'published_at': '2026-01-01T00:00:00Z',
            'title': 'Python Engineer',
            'employer': 'ACME',
            'url': 'https://example.test/jobs/100',
            'snippet_req': 'Python',
            'snippet_resp': 'Build services',
            'salary_text': '1000',
            'score': 0.9,
        }
    ])
    rows = db.list_saved_search_results(sid3)
    assert rows[0]['vacancy_id'] == '100'
    assert rows[0]['score'] == 0.9

    db.upsert_global_vacancies([
        {
            'vacancy_id': '100',
            'area_id': 1,
            'published_at': '2026-01-01T00:00:00Z',
            'title': 'Python Engineer',
            'employer': 'ACME',
            'url': 'https://example.test/jobs/100',
            'snippet_req': 'Python',
            'snippet_resp': 'Build services',
            'salary_text': '1000',
        }
    ])
    assert db.global_has_vacancy_ids(['100', '404']) == {'100'}
    db.set_global_index_state('k', 'v')
    assert db.get_global_index_state('k') == 'v'

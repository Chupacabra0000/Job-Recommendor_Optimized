import importlib


def _mk_user_and_search(db):
    db.init_db()
    ok, _ = db.create_user('saved@example.com', 'secret1')
    assert ok
    user = db.authenticate('saved@example.com', 'secret1')
    rid = db.create_resume(user['id'], 'Resume A', 'python sql')
    sid, created = db.create_or_get_saved_search(
        user_id=user['id'],
        resume_key=f'rid:{rid}',
        area_id=1,
        timeframe_days=7,
        resume_id=rid,
        resume_label='Resume A',
    )
    assert created is True
    return user['id'], rid, sid


def test_create_or_get_saved_search_updates_existing_record():
    db = importlib.import_module('db')
    user_id, rid, sid1 = _mk_user_and_search(db)

    sid2, created = db.create_or_get_saved_search(
        user_id=user_id,
        resume_key=f'rid:{rid}',
        area_id=1,
        timeframe_days=7,
        resume_id=rid,
        resume_label='Updated label',
        update_interval_hours=12,
        refresh_window_hours=6,
    )
    assert sid2 == sid1
    assert created is False

    saved = db.list_saved_searches(user_id)
    assert len(saved) == 1
    assert saved[0]['resume_label'] == 'Updated label'
    assert saved[0]['update_interval_hours'] == 12
    assert saved[0]['refresh_window_hours'] == 6


def test_saved_search_results_upsert_scores_prune_and_timeline():
    db = importlib.import_module('db')
    user_id, _rid, sid = _mk_user_and_search(db)

    db.upsert_saved_search_results(sid, [
        {
            'vacancy_id': 'vac-1',
            'published_at': '2026-03-01T10:00:00+00:00',
            'title': 'Python Dev',
            'employer': 'ACME',
            'url': 'https://example/1',
            'snippet_req': 'Python',
            'snippet_resp': 'Build APIs',
            'salary_text': '1000',
            'score': 0.91,
        },
        {
            'vacancy_id': 'vac-2',
            'published_at': '2026-02-20T10:00:00+00:00',
            'title': 'Analyst',
            'employer': 'DataCo',
            'url': 'https://example/2',
            'snippet_req': 'SQL',
            'snippet_resp': 'Dashboards',
            'salary_text': '800',
            'score': 0.5,
        },
    ])

    db.upsert_saved_search_results(sid, [{
        'vacancy_id': 'vac-1',
        'published_at': '2026-03-01T10:00:00+00:00',
        'title': 'Python Dev Senior',
        'employer': 'ACME',
        'url': 'https://example/1b',
        'snippet_req': 'Python FastAPI',
        'snippet_resp': 'Build systems',
        'salary_text': '1200',
        'score': None,
    }])

    by_score = db.list_saved_search_results(sid, order_by_score=True)
    assert [r['vacancy_id'] for r in by_score] == ['vac-1', 'vac-2']
    assert by_score[0]['title'] == 'Python Dev Senior'
    assert by_score[0]['score'] == 0.91  # preserved because replacement score was None

    db.set_saved_search_scores(sid, {'vac-2': 0.99})
    by_score = db.list_saved_search_results(sid, order_by_score=True)
    assert [r['vacancy_id'] for r in by_score] == ['vac-2', 'vac-1']

    deleted = db.prune_saved_search_results(sid, '2026-03-01T00:00:00+00:00')
    assert deleted == 1
    remaining = db.list_saved_search_results(sid, order_by_score=False)
    assert [r['vacancy_id'] for r in remaining] == ['vac-1']

    db.touch_ranked(sid)
    timeline = db.list_default_timeline(user_id)
    assert len(timeline) == 1
    assert timeline[0]['vacancy_id'] == 'vac-1'


def test_delete_saved_searches_for_resume_and_delete_saved_search_remove_results():
    db = importlib.import_module('db')
    db.init_db()
    ok, _ = db.create_user('cleanup@example.com', 'secret1')
    assert ok
    user = db.authenticate('cleanup@example.com', 'secret1')
    rid1 = db.create_resume(user['id'], 'CV1', 'text1')
    rid2 = db.create_resume(user['id'], 'CV2', 'text2')

    sid1, _ = db.create_or_get_saved_search(user['id'], f'rid:{rid1}', 1, 7, resume_id=rid1, resume_label='CV1')
    sid2, _ = db.create_or_get_saved_search(user['id'], f'rid:{rid2}', 1, 7, resume_id=rid2, resume_label='CV2')
    db.upsert_saved_search_results(sid1, [{'vacancy_id': 'x', 'published_at': '2026-03-01', 'title': 'X'}])
    db.upsert_saved_search_results(sid2, [{'vacancy_id': 'y', 'published_at': '2026-03-01', 'title': 'Y'}])

    deleted_ids = db.delete_saved_searches_for_resume(user['id'], rid1)
    assert deleted_ids == [sid1]
    assert [row['id'] for row in db.list_saved_searches(user['id'])] == [sid2]

    db.delete_saved_search(sid2)
    assert db.list_saved_searches(user['id']) == []
    assert db.list_saved_search_results(sid2) == []

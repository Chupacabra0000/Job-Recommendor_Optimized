import importlib


def test_global_vacancies_state_and_pruning_flow():
    db = importlib.import_module('db')
    db.init_db()

    db.upsert_global_vacancies([
        {
            'vacancy_id': '100',
            'area_id': 1,
            'published_at': '2026-03-01T10:00:00+00:00',
            'title': 'Python Dev',
            'employer': 'ACME',
            'url': 'https://example/100',
            'snippet_req': 'Python',
            'snippet_resp': 'APIs',
            'salary_text': '1000',
        },
        {
            'vacancy_id': '200',
            'area_id': 1,
            'published_at': '2026-02-01T10:00:00+00:00',
            'title': 'Analyst',
            'employer': 'DataCo',
            'url': 'https://example/200',
            'snippet_req': 'SQL',
            'snippet_resp': 'BI',
            'salary_text': '800',
        },
    ])

    assert db.global_has_vacancy_ids(['100', '999']) == {'100'}
    assert set(db.list_all_global_vacancy_ids()) == {'100', '200'}
    assert db.get_max_global_published_at(1) == '2026-03-01T10:00:00+00:00'

    db.set_global_index_state('k', '2026-03-01T10:00:00Z')
    db.set_global_index_state_if_newer('k', '2026-02-01T10:00:00Z')
    assert db.get_global_index_state('k') == '2026-03-01T10:00:00Z'
    db.set_global_index_state_if_newer('k', '2026-04-01T10:00:00Z')
    assert db.get_global_index_state('k') == '2026-04-01T10:00:00Z'

    deleted = db.prune_global_vacancies_for_area(1, '2026-03-01T00:00:00+00:00')
    assert deleted == 1
    assert set(db.list_all_global_vacancy_ids()) == {'100'}


def test_replace_global_vacancies_for_area_replaces_only_target_area():
    db = importlib.import_module('db')
    db.init_db()
    db.upsert_global_vacancies([
        {'vacancy_id': 'a1', 'area_id': 1, 'published_at': '2026-03-01', 'title': 'A1'},
        {'vacancy_id': 'b2', 'area_id': 2, 'published_at': '2026-03-01', 'title': 'B2'},
    ])

    db.replace_global_vacancies_for_area(1, [
        {'vacancy_id': 'a3', 'area_id': 1, 'published_at': '2026-03-02', 'title': 'A3', 'employer': '', 'url': '', 'snippet_req': '', 'snippet_resp': '', 'salary_text': ''},
    ])

    all_ids = set(db.list_all_global_vacancy_ids())
    assert all_ids == {'a3', 'b2'}

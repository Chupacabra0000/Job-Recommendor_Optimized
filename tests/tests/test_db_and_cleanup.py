import importlib
import os


def _load_db(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "test.db"))
    import db
    importlib.reload(db)
    db.init_db()
    return db


def test_user_auth_resume_and_favorites_flow(monkeypatch, tmp_path):
    db = _load_db(monkeypatch, tmp_path)

    ok, msg = db.create_user("Test@Example.com", "secret1")
    assert ok is True
    user = db.authenticate("test@example.com", "secret1")
    assert user["email"] == "test@example.com"
    assert db.authenticate("test@example.com", "wrong") is None

    resume_id = db.create_resume(user["id"], "cv", "python sql")
    resumes = db.list_resumes(user["id"])
    assert resumes[0]["id"] == resume_id

    db.add_favorite(user["id"], "vac-1")
    db.add_favorite(user["id"], "vac-1")
    assert db.list_favorites(user["id"]) == ["vac-1"]
    db.remove_favorite(user["id"], "vac-1")
    assert db.list_favorites(user["id"]) == []


def test_session_lifecycle_and_expired_token(monkeypatch, tmp_path):
    db = _load_db(monkeypatch, tmp_path)
    ok, _ = db.create_user("user@example.com", "secret1")
    assert ok
    user = db.authenticate("user@example.com", "secret1")

    token = db.create_session(user["id"], days_valid=1)
    loaded = db.get_user_by_token(token)
    assert loaded["id"] == user["id"]

    conn = db.get_conn()
    conn.execute("UPDATE sessions SET expires_at=? WHERE token=?", ("2000-01-01T00:00:00Z", token))
    conn.commit()
    conn.close()

    assert db.get_user_by_token(token) is None


def test_saved_search_upsert_limit_and_timeline(monkeypatch, tmp_path):
    db = _load_db(monkeypatch, tmp_path)
    ok, _ = db.create_user("user2@example.com", "secret1")
    assert ok
    user = db.authenticate("user2@example.com", "secret1")

    r1 = db.create_resume(user["id"], "Resume 1", "python")
    r2 = db.create_resume(user["id"], "Resume 2", "design")
    r3 = db.create_resume(user["id"], "Resume 3", "other")

    sid1, created1 = db.create_or_get_saved_search(user["id"], f"rid:{r1}", 1, 7, resume_id=r1, resume_label="Resume 1")
    sid1b, created1b = db.create_or_get_saved_search(user["id"], f"rid:{r1}", 1, 7, resume_id=r1, resume_label="Updated")
    sid2, _ = db.create_or_get_saved_search(user["id"], f"rid:{r2}", 1, 7, resume_id=r2)
    sid3, _ = db.create_or_get_saved_search(user["id"], f"rid:{r3}", 1, 7, resume_id=r3)

    assert created1 is True
    assert created1b is False
    assert sid1 == sid1b

    db.upsert_saved_search_results(sid1, [{
        "vacancy_id": "vac-1",
        "published_at": "2026-03-01T00:00:00Z",
        "title": "Python Engineer",
        "employer": "Acme",
        "url": "https://example.com/1",
        "snippet_req": "python",
        "snippet_resp": "apis",
        "salary_text": "1000",
        "score": 0.95,
    }])
    db.upsert_saved_search_results(sid2, [{
        "vacancy_id": "vac-2",
        "published_at": "2026-03-02T00:00:00Z",
        "title": "Designer",
        "employer": "Beta",
        "url": "https://example.com/2",
        "snippet_req": "figma",
        "snippet_resp": "design",
        "salary_text": "900",
        "score": 0.20,
    }])

    timeline = db.list_default_timeline(user["id"])
    assert [x["vacancy_id"] for x in timeline[:2]] == ["vac-1", "vac-2"]

    deleted = db.enforce_saved_search_limit(user["id"], keep_n=2)
    assert len(deleted) == 1
    remaining = db.list_saved_searches(user["id"])
    assert len(remaining) == 2


def test_global_vacancy_helpers(monkeypatch, tmp_path):
    db = _load_db(monkeypatch, tmp_path)

    rows = [
        {
            "vacancy_id": "1",
            "area_id": 113,
            "published_at": "2026-03-03T00:00:00Z",
            "title": "Python Engineer",
            "employer": "Acme",
            "url": "https://example.com/1",
            "snippet_req": "python",
            "snippet_resp": "apis",
            "salary_text": "1000",
        },
        {
            "vacancy_id": "2",
            "area_id": 113,
            "published_at": "2026-03-01T00:00:00Z",
            "title": "Designer",
            "employer": "Beta",
            "url": "https://example.com/2",
            "snippet_req": "figma",
            "snippet_resp": "design",
            "salary_text": "900",
        },
    ]
    db.upsert_global_vacancies(rows)
    assert db.global_has_vacancy_ids(["1", "3"]) == {"1"}
    assert db.get_max_global_published_at(113) == "2026-03-03T00:00:00Z"

    deleted = db.prune_global_vacancies_for_area(113, "2026-03-02T00:00:00Z")
    assert deleted == 1
    assert db.list_all_global_vacancy_ids() == ["1"]


def test_search_cleanup_calls_db_and_index(monkeypatch):
    import search_cleanup

    deleted_index_ids = []
    monkeypatch.setattr(search_cleanup, "enforce_saved_search_limit", lambda user_id, keep_n=3: [11, 12])
    monkeypatch.setattr(search_cleanup, "delete_saved_searches_for_resume", lambda user_id, resume_id: [21])
    monkeypatch.setattr(search_cleanup, "delete_resume", lambda user_id, resume_id: deleted_index_ids.append(("resume", user_id, resume_id)))
    monkeypatch.setattr(search_cleanup, "delete_index_dir", lambda sid: deleted_index_ids.append(sid))

    assert search_cleanup.enforce_limit_and_cleanup(7, keep_n=2) == [11, 12]
    assert search_cleanup.delete_resume_and_cleanup(7, 99) == [21]
    assert deleted_index_ids == [11, 12, 21, ("resume", 7, 99)]

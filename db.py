import base64
import datetime
import hashlib
import hmac
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

DB_PATH = os.getenv("APP_DB_PATH", "app.db")


# ---------------- connection helpers ----------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _tx():
    conn = get_conn()
    try:
        yield conn, conn.cursor()
        conn.commit()
    finally:
        conn.close()


# ---------------- schema / migrations ----------------
_SAVE_SEARCHES_DDL = """
CREATE TABLE IF NOT EXISTS saved_searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    resume_id INTEGER,
    resume_key TEXT NOT NULL,
    resume_label TEXT,
    area_id INTEGER NOT NULL,
    timeframe_days INTEGER NOT NULL,
    update_interval_hours INTEGER NOT NULL DEFAULT 24,
    refresh_window_hours INTEGER NOT NULL DEFAULT 24,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_ranked_at TEXT,
    last_refresh_at TEXT,
    UNIQUE(user_id, resume_key, area_id, timeframe_days),
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(resume_id) REFERENCES resumes(id)
)
"""


def _create_tables(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, job_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vacancy_embeddings (
            vacancy_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dim INTEGER NOT NULL,
            emb_blob BLOB NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (vacancy_id, model_name)
        )
        """
    )
    cur.execute(_SAVE_SEARCHES_DDL)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_search_results (
            search_id INTEGER NOT NULL,
            vacancy_id TEXT NOT NULL,
            published_at TEXT,
            title TEXT,
            employer TEXT,
            url TEXT,
            snippet_req TEXT,
            snippet_resp TEXT,
            salary_text TEXT,
            score REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (search_id, vacancy_id),
            FOREIGN KEY(search_id) REFERENCES saved_searches(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS global_vacancies (
            vacancy_id TEXT PRIMARY KEY,
            area_id INTEGER NOT NULL,
            published_at TEXT,
            title TEXT,
            employer TEXT,
            url TEXT,
            snippet_req TEXT,
            snippet_resp TEXT,
            salary_text TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS global_index_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def _migrate_saved_searches_resume_key(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA table_info(saved_searches)")
    cols = [r[1] for r in cur.fetchall()]
    if not cols or "resume_key" in cols:
        return

    cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old")
    cur.execute(_SAVE_SEARCHES_DDL)
    cur.execute(
        """
        INSERT INTO saved_searches (
            id, user_id, resume_id, resume_key, resume_label, area_id, timeframe_days,
            update_interval_hours, refresh_window_hours, created_at, last_ranked_at, last_refresh_at
        )
        SELECT
            id, user_id, resume_id,
            ('rid:' || resume_id) AS resume_key,
            NULL AS resume_label,
            area_id, timeframe_days,
            update_interval_hours, refresh_window_hours, created_at, last_ranked_at, last_refresh_at
        FROM saved_searches_old
        """
    )
    cur.execute("DROP TABLE saved_searches_old")


def init_db() -> None:
    with _tx() as (_, cur):
        _create_tables(cur)
        _migrate_saved_searches_resume_key(cur)


# ---------------- Password hashing (stdlib: PBKDF2) ----------------
# stored format: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 200_000
    dk = _pbkdf2_hash(password, salt, iterations)
    return "pbkdf2_sha256$%d$%s$%s" % (
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
        dk = _pbkdf2_hash(password, salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ---------------- Users ----------------
def create_user(email: str, password: str) -> Tuple[bool, str]:
    if len(password) < 6:
        return False, "Пароль слишком короткий (мин. 6 символов)."

    with _tx() as (_, cur):
        try:
            cur.execute(
                "INSERT INTO users(email, password_hash) VALUES(?, ?)",
                (email.strip().lower(), hash_password(password)),
            )
            return True, "Пользователь создан."
        except sqlite3.IntegrityError:
            return False, "Email уже зарегистрирован."


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if not row:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return dict(row)
    finally:
        conn.close()


# ---------------- Resumes ----------------
def list_resumes(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, name, text, created_at FROM resumes WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def create_resume(user_id: int, name: str, text: str) -> int:
    with _tx() as (_, cur):
        cur.execute(
            "INSERT INTO resumes(user_id, name, text) VALUES(?, ?, ?)",
            (user_id, name, text),
        )
        return int(cur.lastrowid)


def delete_resume(user_id: int, resume_id: int) -> None:
    """Deletes resume row only. Call search cleanup separately."""
    with _tx() as (_, cur):
        cur.execute("DELETE FROM resumes WHERE user_id = ? AND id = ?", (user_id, resume_id))


# ---------------- Favorites ----------------
def list_favorites(user_id: int) -> List[str]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT job_id FROM favorites WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
        return [str(r["job_id"]) for r in rows]
    finally:
        conn.close()


def add_favorite(user_id: int, job_id: str) -> None:
    with _tx() as (_, cur):
        cur.execute(
            "INSERT OR IGNORE INTO favorites(user_id, job_id) VALUES(?, ?)",
            (user_id, str(job_id)),
        )


def remove_favorite(user_id: int, job_id: str) -> None:
    with _tx() as (_, cur):
        cur.execute("DELETE FROM favorites WHERE user_id = ? AND job_id = ?", (user_id, str(job_id)))


# ---------------- Persistent sessions (login survive refresh) ----------------
def _rand_token(nbytes: int = 24) -> str:
    return base64.urlsafe_b64encode(os.urandom(nbytes)).decode("ascii").rstrip("=")


def create_session(user_id: int, days_valid: int = 30) -> str:
    token = _rand_token()
    now = datetime.datetime.utcnow()
    exp = now + datetime.timedelta(days=days_valid)

    with _tx() as (_, cur):
        cur.execute(
            "INSERT INTO sessions(token, user_id, expires_at) VALUES(?, ?, ?)",
            (token, user_id, exp.isoformat() + "Z"),
        )
    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None

    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT u.*, s.expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if not row:
            return None

        try:
            exp = datetime.datetime.fromisoformat(str(row["expires_at"]).replace("Z", ""))
            if datetime.datetime.utcnow() > exp:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                conn.commit()
                return None
        except Exception:
            return None

        payload = dict(row)
        payload.pop("expires_at", None)
        return payload
    finally:
        conn.close()


def delete_session(token: str) -> None:
    with _tx() as (_, cur):
        cur.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ---------------- Vacancy embeddings cache ----------------
def get_embedding(vacancy_id: str, model_name: str) -> Optional[Tuple[int, bytes]]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT dim, emb_blob FROM vacancy_embeddings WHERE vacancy_id = ? AND model_name = ?",
            (str(vacancy_id), str(model_name)),
        ).fetchone()
        if not row:
            return None
        return int(row["dim"]), row["emb_blob"]
    finally:
        conn.close()


def put_embedding(vacancy_id: str, model_name: str, dim: int, emb_blob: bytes) -> None:
    with _tx() as (_, cur):
        cur.execute(
            """
            INSERT INTO vacancy_embeddings(vacancy_id, model_name, dim, emb_blob, updated_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(vacancy_id, model_name) DO UPDATE SET
                dim=excluded.dim,
                emb_blob=excluded.emb_blob,
                updated_at=CURRENT_TIMESTAMP
            """,
            (str(vacancy_id), str(model_name), int(dim), sqlite3.Binary(emb_blob)),
        )


# ---------------- Saved searches (resume-based history) ----------------
def list_saved_searches(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT ss.*, r.name AS resume_name
            FROM saved_searches ss
            JOIN resumes r ON r.id = ss.resume_id
            WHERE ss.user_id = ?
            ORDER BY ss.created_at DESC, ss.id DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_latest_saved_search(user_id: int) -> Optional[Dict[str, Any]]:
    rows = list_saved_searches(user_id)
    return rows[0] if rows else None


def create_or_get_saved_search(
    user_id: int,
    resume_key: str,
    area_id: int,
    timeframe_days: int,
    resume_id: Optional[int] = None,
    resume_label: Optional[str] = None,
    update_interval_hours: int = 24,
    refresh_window_hours: int = 24,
) -> Tuple[int, bool]:
    """
    Returns (search_id, created_new).
    Unique by (user_id,resume_key,area_id,timeframe_days).
    resume_key examples:
      - "rid:123" for stored resumes
      - "pdf:<sha256>" for PDF text searches
    """
    with _tx() as (_, cur):
        row = cur.execute(
            """
            SELECT id FROM saved_searches
            WHERE user_id=? AND resume_key=? AND area_id=? AND timeframe_days=?
            """,
            (int(user_id), str(resume_key), int(area_id), int(timeframe_days)),
        ).fetchone()
        if row:
            sid = int(row["id"])
            cur.execute(
                """
                UPDATE saved_searches
                SET resume_id=?, resume_label=?, update_interval_hours=?, refresh_window_hours=?
                WHERE id=?
                """,
                (resume_id, resume_label, int(update_interval_hours), int(refresh_window_hours), sid),
            )
            return sid, False

        cur.execute(
            """
            INSERT INTO saved_searches (user_id, resume_id, resume_key, resume_label, area_id, timeframe_days, update_interval_hours, refresh_window_hours)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                int(user_id),
                resume_id,
                str(resume_key),
                resume_label,
                int(area_id),
                int(timeframe_days),
                int(update_interval_hours),
                int(refresh_window_hours),
            ),
        )
        return int(cur.lastrowid), True


def touch_ranked(search_id: int) -> None:
    with _tx() as (_, cur):
        cur.execute(
            "UPDATE saved_searches SET last_ranked_at=CURRENT_TIMESTAMP WHERE id=?",
            (int(search_id),),
        )


def touch_refreshed(search_id: int) -> None:
    with _tx() as (_, cur):
        cur.execute(
            "UPDATE saved_searches SET last_refresh_at=CURRENT_TIMESTAMP WHERE id=?",
            (int(search_id),),
        )


def delete_saved_search(search_id: int) -> None:
    with _tx() as (_, cur):
        cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (int(search_id),))
        cur.execute("DELETE FROM saved_searches WHERE id=?", (int(search_id),))


def delete_saved_searches_for_resume(user_id: int, resume_id: int) -> List[int]:
    with _tx() as (_, cur):
        rows = cur.execute(
            "SELECT id FROM saved_searches WHERE user_id=? AND resume_id=?",
            (user_id, resume_id),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        for sid in ids:
            cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (sid,))
            cur.execute("DELETE FROM saved_searches WHERE id=?", (sid,))
        return ids


def enforce_saved_search_limit(user_id: int, keep_n: int = 3) -> List[int]:
    """
    Keep only last N searches for user. Returns deleted search_ids.
    """
    with _tx() as (_, cur):
        rows = cur.execute(
            "SELECT id FROM saved_searches WHERE user_id=? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        to_delete = ids[keep_n:]
        for sid in to_delete:
            cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (sid,))
            cur.execute("DELETE FROM saved_searches WHERE id=?", (sid,))
        return to_delete


# ---------------- Saved search results ----------------
def upsert_saved_search_results(search_id: int, rows: List[Dict[str, Any]]) -> None:
    """
    rows entries must include:
      vacancy_id, published_at, title, employer, url, snippet_req, snippet_resp, salary_text
    score can be provided (optional).
    """
    with _tx() as (_, cur):
        for r in rows:
            cur.execute(
                """
                INSERT INTO saved_search_results(
                  search_id, vacancy_id, published_at, title, employer, url,
                  snippet_req, snippet_resp, salary_text, score, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(search_id, vacancy_id) DO UPDATE SET
                  published_at=excluded.published_at,
                  title=excluded.title,
                  employer=excluded.employer,
                  url=excluded.url,
                  snippet_req=excluded.snippet_req,
                  snippet_resp=excluded.snippet_resp,
                  salary_text=excluded.salary_text,
                  score=COALESCE(excluded.score, saved_search_results.score),
                  updated_at=CURRENT_TIMESTAMP
                """,
                (
                    int(search_id),
                    str(r.get("vacancy_id")),
                    r.get("published_at"),
                    r.get("title"),
                    r.get("employer"),
                    r.get("url"),
                    r.get("snippet_req"),
                    r.get("snippet_resp"),
                    r.get("salary_text"),
                    r.get("score"),
                ),
            )


def set_saved_search_scores(search_id: int, scores: Dict[str, float]) -> None:
    """
    Update score for given vacancy_ids. Does not touch others.
    """
    with _tx() as (_, cur):
        for vid, sc in scores.items():
            cur.execute(
                "UPDATE saved_search_results SET score=?, updated_at=CURRENT_TIMESTAMP WHERE search_id=? AND vacancy_id=?",
                (float(sc), int(search_id), str(vid)),
            )


def prune_saved_search_results(search_id: int, cutoff_iso: str) -> int:
    with _tx() as (_, cur):
        cur.execute(
            "DELETE FROM saved_search_results WHERE search_id=? AND published_at IS NOT NULL AND published_at < ?",
            (int(search_id), str(cutoff_iso)),
        )
        return int(cur.rowcount)


def list_saved_search_results(search_id: int, order_by_score: bool = True) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        if order_by_score:
            rows = conn.execute(
                """
                SELECT * FROM saved_search_results
                WHERE search_id=?
                ORDER BY (score IS NULL) ASC, score DESC, published_at DESC
                """,
                (int(search_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM saved_search_results
                WHERE search_id=?
                ORDER BY published_at DESC
                """,
                (int(search_id),),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_default_timeline(user_id: int, limit: int = 5000) -> List[Dict[str, Any]]:
    """Merged saved vacancies across user's saved searches using MOST RECENT search score per vacancy."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT vacancy_id, published_at, title, employer, url, snippet_req, snippet_resp, salary_text, score
            FROM (
                SELECT
                    r.*,
                    s.last_ranked_at,
                    s.created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.vacancy_id
                        ORDER BY COALESCE(s.last_ranked_at, s.created_at) DESC
                    ) AS rn
                FROM saved_search_results r
                JOIN saved_searches s ON s.id = r.search_id
                WHERE s.user_id = ?
            )
            WHERE rn = 1
            ORDER BY (score IS NULL) ASC, score DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
        return [dict(x) for x in rows]
    finally:
        conn.close()


# ---------------- Global pre-index helpers ----------------
def upsert_global_vacancies(rows: List[Dict[str, Any]]) -> None:
    """
    rows entries must include:
      vacancy_id, area_id, published_at, title, employer, url, snippet_req, snippet_resp, salary_text
    """
    with _tx() as (_, cur):
        for r in rows:
            cur.execute(
                """
                INSERT INTO global_vacancies(
                    vacancy_id, area_id, published_at, title,
                    employer, url, snippet_req, snippet_resp, salary_text, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(vacancy_id) DO UPDATE SET
                    area_id=excluded.area_id,
                    published_at=excluded.published_at,
                    title=excluded.title,
                    employer=excluded.employer,
                    url=excluded.url,
                    snippet_req=excluded.snippet_req,
                    snippet_resp=excluded.snippet_resp,
                    salary_text=excluded.salary_text,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    str(r.get("vacancy_id")),
                    int(r.get("area_id")) if r.get("area_id") is not None else None,
                    r.get("published_at"),
                    r.get("title"),
                    r.get("employer"),
                    r.get("url"),
                    r.get("snippet_req"),
                    r.get("snippet_resp"),
                    r.get("salary_text"),
                ),
            )


def get_global_index_state(key: str) -> Optional[str]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT value FROM global_index_state WHERE key=?", (str(key),)).fetchone()
        return None if not row else row["value"]
    finally:
        conn.close()


def set_global_index_state(key: str, value: str) -> None:
    with _tx() as (_, cur):
        cur.execute(
            """
            INSERT INTO global_index_state(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(key), str(value)),
        )


def global_has_vacancy_ids(ids: List[str]) -> set:
    """Return set of vacancy_ids that already exist in global_vacancies."""
    ids = [str(x).strip() for x in ids if str(x).strip()]
    if not ids:
        return set()

    conn = get_conn()
    try:
        out = set()
        chunk_size = 500
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"SELECT vacancy_id FROM global_vacancies WHERE vacancy_id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out |= {str(r["vacancy_id"]) for r in rows}
        return out
    finally:
        conn.close()


def set_global_index_state_if_newer(key: str, value_iso_z: str) -> None:
    """
    Convenience: store only if newer lexicographically/ISO-wise.
    ISO Z timestamps compare correctly as strings in practice.
    """
    old = get_global_index_state(key)
    if (old is None) or (str(value_iso_z) > str(old)):
        set_global_index_state(key, value_iso_z)


def get_max_global_published_at(area_id: int) -> Optional[str]:
    """Newest published_at we have for this area_id (ISO string)."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT MAX(published_at) AS mx FROM global_vacancies WHERE area_id=? AND published_at IS NOT NULL",
            (int(area_id),),
        ).fetchone()
        if not row or not row["mx"]:
            return None
        return str(row["mx"])
    finally:
        conn.close()


def replace_global_vacancies_for_area(area_id: int, rows: List[Dict[str, Any]]) -> None:
    """
    Full rebuild behavior: replace the pool for a given area_id.
    Keeps table bounded over time.
    """
    with _tx() as (_, cur):
        cur.execute("DELETE FROM global_vacancies WHERE area_id = ?", (int(area_id),))
        cur.executemany(
            """
            INSERT INTO global_vacancies (
                vacancy_id, area_id, published_at, title, employer, url,
                snippet_req, snippet_resp, salary_text, updated_at
            ) VALUES (
                :vacancy_id, :area_id, :published_at, :title, :employer, :url,
                :snippet_req, :snippet_resp, :salary_text, CURRENT_TIMESTAMP
            )
            ON CONFLICT(vacancy_id) DO UPDATE SET
                area_id=excluded.area_id,
                published_at=excluded.published_at,
                title=excluded.title,
                employer=excluded.employer,
                url=excluded.url,
                snippet_req=excluded.snippet_req,
                snippet_resp=excluded.snippet_resp,
                salary_text=excluded.salary_text,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )


def prune_global_vacancies_for_area(area_id: int, cutoff_iso: str) -> int:
    """
    Best-effort prune based on published_at.
    cutoff_iso: ISO8601 string.
    Returns deleted row count.
    """
    with _tx() as (_, cur):
        cur.execute(
            """
            DELETE FROM global_vacancies
            WHERE area_id = ?
              AND published_at IS NOT NULL
              AND published_at < ?
            """,
            (int(area_id), cutoff_iso),
        )
        return int(cur.rowcount)


def list_all_global_vacancy_ids() -> List[str]:
    """
    Used for vector store compaction: keep vectors for all IDs still in global_vacancies
    across all areas.
    """
    conn = get_conn()
    try:
        rows = conn.execute("SELECT vacancy_id FROM global_vacancies").fetchall()
        return [str(r[0]) for r in rows if r and r[0]]
    finally:
        conn.close()

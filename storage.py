import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT,
                text TEXT,
                llm_score REAL,
                stylometry_score REAL,
                confidence REAL,
                attribution TEXT,
                status TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                creator_id TEXT,
                timestamp TEXT,
                attribution TEXT,
                confidence REAL,
                llm_score REAL,
                stylometry_score REAL,
                status TEXT,
                appeal_reasoning TEXT
            )
            """
        )


def save_submission(record: dict, db_path: str = DB_PATH) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO submissions
                (content_id, creator_id, text, llm_score, stylometry_score,
                 confidence, attribution, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["content_id"], record["creator_id"], record["text"],
                record["llm_score"], record["stylometry_score"], record["confidence"],
                record["attribution"], record["status"], _now(),
            ),
        )


def get_content(content_id: str, db_path: str = DB_PATH) -> dict | None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def update_status(content_id: str, status: str, db_path: str = DB_PATH) -> bool:
    with _conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE submissions SET status = ? WHERE content_id = ?",
            (status, content_id),
        )
    return cur.rowcount > 0


def append_audit(entry: dict, db_path: str = DB_PATH) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit
                (content_id, creator_id, timestamp, attribution, confidence,
                 llm_score, stylometry_score, status, appeal_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["content_id"], entry["creator_id"], _now(),
                entry["attribution"], entry["confidence"], entry["llm_score"],
                entry["stylometry_score"], entry["status"],
                entry.get("appeal_reasoning"),
            ),
        )


def recent_log(n: int = 20, db_path: str = DB_PATH) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]

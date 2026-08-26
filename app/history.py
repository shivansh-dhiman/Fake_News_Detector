import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "history.db"
MAX_ENTRIES = 100

_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
_connection.execute(
    """
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        input_preview TEXT NOT NULL,
        verdict TEXT NOT NULL,
        confidence REAL NOT NULL,
        result_json TEXT NOT NULL
    )
    """
)
_connection.commit()


def save_analysis(input_preview: str, result: dict) -> None:
    _connection.execute(
        "INSERT INTO history (created_at, input_preview, verdict, confidence, result_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            input_preview[:120],
            result["verdict"],
            result["confidence"],
            json.dumps(result),
        ),
    )
    _connection.execute(
        "DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT ?)",
        (MAX_ENTRIES,),
    )
    _connection.commit()


def get_recent(limit: int = 20) -> list[dict]:
    cursor = _connection.execute(
        "SELECT id, created_at, input_preview, verdict, confidence, result_json "
        "FROM history ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [
        {
            "id": row[0],
            "created_at": row[1],
            "input_preview": row[2],
            "verdict": row[3],
            "confidence": row[4],
            "result": json.loads(row[5]),
        }
        for row in cursor.fetchall()
    ]

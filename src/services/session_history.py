"""SQLite persistence for conversation sessions and messages."""
import sqlite3
import os
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionHistory:
    """Store conversation metadata and text messages in SQLite."""

    def __init__(self, path: Optional[str] = None, max_sessions: int = 50):
        project_root = Path(__file__).resolve().parents[2]
        self.path = Path(path or os.getenv("SESSION_HISTORY_DB", project_root / "data" / "voice_agent.sqlite3"))
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL, latency REAL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                """)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _session(self, connection, row) -> Dict[str, Any]:
        messages = connection.execute(
            "SELECT role, content, latency FROM messages WHERE session_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        return {
            "id": row["id"], "title": row["title"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "messages": [
                {key: message[key] for key in ("role", "content", "latency") if message[key] is not None}
                for message in messages
            ],
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                rows = connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
                return [self._session(connection, row) for row in rows]

    def create(self, title: str = "New conversation") -> Dict[str, Any]:
        now = self._now()
        session_id = uuid.uuid4().hex
        with self._lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title.strip() or "New conversation", now, now),
            )
                connection.execute(
                "DELETE FROM sessions WHERE id IN (SELECT id FROM sessions ORDER BY updated_at DESC LIMIT -1 OFFSET ?)",
                (self.max_sessions,),
            )
                row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
                return self._session(connection, row)

    def append(self, session_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            return None

        with self._lock, closing(self._connect()) as connection:
            with connection:
                row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
                if row is None:
                    return None
                title = content[:60] if role == "user" and row["title"] == "New conversation" else row["title"]
                connection.execute(
                "INSERT INTO messages (session_id, role, content, latency, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, message.get("latency"), self._now()),
            )
                connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, self._now(), session_id),
            )
                row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
                return self._session(connection, row)

    def delete(self, session_id: str) -> bool:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                return cursor.rowcount > 0
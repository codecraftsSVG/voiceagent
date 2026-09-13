"""SQLite-backed semantic response cache with dependency-free embeddings."""
import hashlib
import math
import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional


class SemanticCache:
    """Cache answers for near-duplicate questions using cosine similarity."""

    def __init__(self, path: str, threshold: float = 0.92, max_entries: int = 1000):
        self.path = Path(path)
        self.threshold = threshold
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL,
                    question TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    response TEXT NOT NULL,
                    audio BLOB,
                    audio_format TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """)
                connection.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_key ON semantic_cache(cache_key)")

    @staticmethod
    def _vector(text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        vector: Dict[str, float] = {}
        for feature in features:
            vector[feature] = vector.get(feature, 0.0) + 1.0
        magnitude = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {key: value / magnitude for key, value in vector.items()}

    @staticmethod
    def _serialize(vector: Dict[str, float]) -> str:
        return ",".join(f"{key}:{value}" for key, value in vector.items())

    @staticmethod
    def _deserialize(value: str) -> Dict[str, float]:
        return {
            key: float(number)
            for key, number in (part.split(":", 1) for part in value.split(",") if ":" in part)
        }

    @staticmethod
    def _similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())

    def lookup(self, question: str, context_key: str) -> Optional[Dict[str, Any]]:
        vector = self._vector(question)
        with self._lock, closing(self._connect()) as connection:
            with connection:
                rows = connection.execute(
                "SELECT * FROM semantic_cache WHERE cache_key LIKE ? ORDER BY last_used_at DESC",
                (context_key + ":%",),
                ).fetchall()
                best = None
                best_score = 0.0
                for row in rows:
                    score = self._similarity(vector, self._deserialize(row["vector"]))
                    if score >= self.threshold and score > best_score:
                        best, best_score = row, score
                if best is None:
                    return None
                connection.execute("UPDATE semantic_cache SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?", (best["id"],))
                return {
                "response": best["response"],
                "audio": best["audio"],
                "audio_format": best["audio_format"],
                "similarity": best_score,
                }

    def put(
        self,
        question: str,
        context_key: str,
        response: str,
        audio: Optional[bytes] = None,
        audio_format: Optional[str] = None,
    ) -> None:
        question_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()
        with self._lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                "INSERT INTO semantic_cache (cache_key, question, vector, response, audio, audio_format) VALUES (?, ?, ?, ?, ?, ?)",
                (context_key + ":" + question_hash, question, self._serialize(self._vector(question)), response, audio, audio_format),
            )
                connection.execute(
                "DELETE FROM semantic_cache WHERE id IN (SELECT id FROM semantic_cache ORDER BY last_used_at DESC LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )

    def update_audio(self, question: str, context_key: str, audio: bytes, audio_format: str) -> None:
        question_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()
        with self._lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                "UPDATE semantic_cache SET audio = ?, audio_format = ?, last_used_at = CURRENT_TIMESTAMP WHERE cache_key = ?",
                (audio, audio_format, context_key + ":" + question_hash),
            )

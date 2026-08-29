"""
src/db/chat_history.py
──────────────────────
Persistent chat history using SQLite (local) or PostgreSQL (production).

The DATABASE_URL setting controls which backend is used:
  - SQLite locally:      (empty) -> defaults to chat_history.db
  - PostgreSQL on HF:    postgresql://user:pass@host/dbname
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config.settings import settings

logger = logging.getLogger(__name__)


class ChatHistoryDB:
    """
    Persistent chat history that seamlessly switches between SQLite and PostgreSQL.
    If the DB connection fails, all methods become silent no-ops so the app never crashes.
    """

    def __init__(self, db_url: str | Path | None = None):
        url = str(db_url) if db_url else settings.DATABASE_URL
        
        if url and url.startswith("postgres"):
            self.is_postgres = True
            self.db_url = url.replace("postgres://", "postgresql://", 1)
        else:
            self.is_postgres = False
            if url and url.startswith("sqlite:///"):
                self.db_url = url.replace("sqlite:///", "")
            else:
                self.db_url = "chat_history.db"

        self._pool = None
        self._connected = False
        try:
            self._init_db()
            self._connected = True
        except Exception as e:
            import warnings
            warnings.warn(f"ChatHistoryDB: Could not connect to database: {e}. Running in no-op mode.")

    def _get_conn(self):
        """
        SQLite: a fresh connection per call, as before — connecting to a
        local file is essentially free, and caching one connection on this
        (process-wide singleton) instance would hand the same sqlite3
        connection to multiple Streamlit session threads at once, which
        sqlite3 does not allow.

        Postgres: pulled from a small connection pool instead of opened
        fresh each time. Every call used to pay a full TCP+TLS+auth
        round-trip to Supabase (100-300ms+), which was invisible against a
        local SQLite file but adds up fast over the network — especially
        since the sidebar alone opens one connection per saved chat to
        build its previews. The pool is created lazily on first use and
        shared for the lifetime of the process; psycopg2's pool is
        thread-safe, so this is safe across concurrent Streamlit sessions.

        A pooled connection can go stale if Supabase's side drops it after
        sitting idle (pool.getconn() doesn't validate before handing one
        out) — checked here with a trivial SELECT 1 so a dead connection
        is discarded and retried once instead of failing the real query.
        """
        if not self.is_postgres:
            conn = sqlite3.connect(self.db_url)
            conn.row_factory = sqlite3.Row
            return conn

        if self._pool is None:
            from psycopg2.pool import ThreadedConnectionPool
            self._pool = ThreadedConnectionPool(1, 5, self.db_url)

        conn = self._pool.getconn()
        try:
            conn.cursor().execute("SELECT 1")
        except Exception:
            self._pool.putconn(conn, close=True)
            conn = self._pool.getconn()
        return conn

    def _release_conn(self, conn) -> None:
        """Returns a Postgres connection to the pool instead of closing the
        socket outright; closes SQLite connections as before."""
        if self.is_postgres and self._pool is not None:
            self._pool.putconn(conn)
        else:
            conn.close()

    def _execute(self, conn, query: str, params: tuple = ()):
        """A simple wrapper to handle syntax differences between SQLite and Postgres."""
        if self.is_postgres:
            from psycopg2.extras import DictCursor
            # Convert SQLite placeholders to Postgres placeholders
            pg_query = query.replace("?", "%s")
            # Convert SQLite auto-increment to Postgres serial
            pg_query = pg_query.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
            
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(pg_query, params)
            return cur
        else:
            return conn.execute(query, params)

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            self._execute(conn, """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL DEFAULT 'default',
                    anime_name   TEXT DEFAULT '',
                    persona      TEXT DEFAULT 'Default',
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                )
            """)
            self._execute(conn, """
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    intent       TEXT DEFAULT '',
                    persona      TEXT DEFAULT 'Default',
                    created_at   TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")
            self._execute(conn, """
                CREATE TABLE IF NOT EXISTS model_usage (
                    usage_date   TEXT NOT NULL,
                    model_name   TEXT NOT NULL,
                    call_count   INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (usage_date, model_name)
                )
            """)
            self._execute(conn, """
                CREATE TABLE IF NOT EXISTS site_feedback (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      TEXT NOT NULL DEFAULT '',
                    rating       TEXT DEFAULT '',
                    comment      TEXT DEFAULT '',
                    created_at   TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            self._release_conn(conn)

    def create_session(
        self,
        user_id: str = "default",
        anime_name: str = "",
        persona: str = "Default",
    ) -> str:
        """Create a new chat session. Returns the session_id."""
        if not self._connected:
            return str(uuid.uuid4())  # return a dummy ID so the app doesn't crash
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            self._execute(
                conn,
                "INSERT INTO sessions (session_id, user_id, anime_name, persona, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (session_id, user_id, anime_name, persona, now, now),
            )
            conn.commit()
        finally:
            self._release_conn(conn)
        return session_id

    def save_turn(
        self,
        session_id: str,
        human_msg: str,
        ai_msg: str,
        intent: str = "",
        persona: str = "Default",
    ) -> None:
        """Persist one human/AI exchange to the database."""
        if not self._connected:
            return
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            self._execute(
                conn,
                "INSERT INTO turns (session_id,role,content,intent,persona,created_at) VALUES (?,?,?,?,?,?)",
                (session_id, "human", human_msg, intent, persona, now),
            )
            self._execute(
                conn,
                "INSERT INTO turns (session_id,role,content,intent,persona,created_at) VALUES (?,?,?,?,?,?)",
                (session_id, "ai", ai_msg, intent, persona, now),
            )
            self._execute(
                conn,
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
            conn.commit()
        finally:
            self._release_conn(conn)

    def load_history(self, session_id: str) -> list[BaseMessage]:
        """Retrieve all messages for a session as LangChain message objects."""
        if not self._connected:
            return []
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT role, content FROM turns WHERE session_id=? ORDER BY turn_id",
                (session_id,),
            )
            rows = cur.fetchall()
        finally:
            self._release_conn(conn)

        messages: list[BaseMessage] = []
        for row in rows:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            else:
                messages.append(AIMessage(content=row["content"]))
        return messages

    def list_sessions(self, user_id: str = "default") -> list[dict]:
        """List all sessions for a user, newest first."""
        if not self._connected:
            return []
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT session_id, anime_name, persona, created_at, updated_at FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            self._release_conn(conn)

    def get_session_preview(self, session_id: str) -> str:
        """Returns the first human message of a session (for UI labels)."""
        if not self._connected:
            return "New conversation"
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT content FROM turns WHERE session_id=? AND role='human' ORDER BY turn_id LIMIT 1",
                (session_id,),
            )
            row = cur.fetchone()
        finally:
            self._release_conn(conn)
            
        if row:
            text = row["content"]
            return text[:60] + "..." if len(text) > 60 else text
        return "New conversation"

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its turns."""
        if not self._connected:
            return
        conn = self._get_conn()
        try:
            self._execute(conn, "DELETE FROM turns WHERE session_id=?", (session_id,))
            self._execute(conn, "DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()
        finally:
            self._release_conn(conn)

    def update_session_meta(
        self,
        session_id: str,
        anime_name: str | None = None,
        persona: str | None = None,
    ) -> None:
        """Update session metadata (anime_name, persona) after a turn."""
        if not self._connected:
            return
        fields, values = [], []
        if anime_name is not None:
            fields.append("anime_name=?")
            values.append(anime_name)
        if persona is not None:
            fields.append("persona=?")
            values.append(persona)
        if not fields:
            return
            
        values.append(datetime.now(timezone.utc).isoformat())
        values.append(session_id)
        
        query = f"UPDATE sessions SET {', '.join(fields)}, updated_at=? WHERE session_id=?"

        conn = self._get_conn()
        try:
            self._execute(conn, query, tuple(values))
            conn.commit()
        finally:
            self._release_conn(conn)

    def get_stats(self) -> dict:
        """
        Aggregate stats for the admin panel: total sessions, total messages,
        and (SQLite only) the DB file size in MB. Postgres doesn't get a
        size figure here — computing it needs a privileged query the
        Supabase connection user may not have.
        """
        if not self._connected:
            return {"sessions": 0, "turns": 0, "db_size_mb": None}
        conn = self._get_conn()
        try:
            sessions_row = self._execute(conn, "SELECT COUNT(*) as c FROM sessions").fetchone()
            turns_row = self._execute(conn, "SELECT COUNT(*) as c FROM turns").fetchone()
        finally:
            self._release_conn(conn)

        db_size_mb = None
        if not self.is_postgres:
            db_path = Path(self.db_url)
            if db_path.exists():
                db_size_mb = db_path.stat().st_size / (1024 * 1024)

        return {
            "sessions": sessions_row["c"],
            "turns": turns_row["c"],
            "db_size_mb": db_size_mb,
        }

    def get_recent_feedback(self, limit: int = 50) -> list[dict]:
        """
        Most recent site_feedback rows for the admin panel, newest first.
        A rating and a comment are independent submissions (see
        add_site_feedback) — either field may be empty on a given row.
        """
        if not self._connected:
            return []
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT rating, comment, created_at FROM site_feedback "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        finally:
            self._release_conn(conn)
        return [dict(row) for row in rows]

    def _safe(self, fn, default=None):
        """Runs fn(conn) against a fresh connection, but never lets a
        transient DB hiccup (connection pool exhausted, a table not yet
        visible on a stale replica, a dropped connection under bursty
        traffic) crash the calling request — logs a warning and returns
        `default` instead. Used for the higher-frequency methods added
        alongside real-user launch (feedback, model-usage tracking) that
        get called on every single message/rerun, unlike the older
        once-per-action methods below, which only ever guarded against the
        DB being unreachable at startup, not a mid-request failure."""
        if not self._connected:
            return default
        try:
            conn = self._get_conn()
        except Exception as e:
            logger.warning(f"ChatHistoryDB: connection failed: {e}")
            return default
        try:
            return fn(conn)
        except Exception as e:
            logger.warning(f"ChatHistoryDB: query failed: {e}")
            return default
        finally:
            # A finally block's own exception silently replaces whatever the
            # try/except above was already handling (or its return value) —
            # a classic Python gotcha. Without this inner guard, a broken
            # connection that fails query AND close() would still escape
            # this method entirely, undermining the "never raises" contract
            # this whole helper exists for.
            try:
                self._release_conn(conn)
            except Exception as e:
                logger.warning(f"ChatHistoryDB: connection close failed: {e}")

    def get_model_usage_today(self, model_name: str) -> int:
        """Today's call count for a model — used to decide whether the
        primary LLM has hit its daily free-tier quota and should fall back
        to a secondary model. Fails open (returns 0) on any DB error, so a
        missing count never blocks or crashes a real request."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        def _run(conn):
            row = self._execute(
                conn,
                "SELECT call_count FROM model_usage WHERE usage_date=? AND model_name=?",
                (today, model_name),
            ).fetchone()
            return row["call_count"] if row else 0

        return self._safe(_run, default=0)

    def increment_model_usage(self, model_name: str) -> None:
        """Record one call against today's count for a model. Best-effort —
        a failure here should never block the actual LLM call it's tracking."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        def _run(conn):
            self._execute(
                conn,
                """
                INSERT INTO model_usage (usage_date, model_name, call_count) VALUES (?,?,1)
                ON CONFLICT (usage_date, model_name) DO UPDATE SET call_count = call_count + 1
                """,
                (today, model_name),
            )
            conn.commit()

        self._safe(_run)

    def add_site_feedback(
        self,
        user_id: str,
        rating: str = "",
        comment: str = "",
    ) -> None:
        """Log one overall-project feedback event — a rating click and a
        written comment are independent submissions (not upserted against
        each other), since this isn't tied to any single reply and a user
        may rate once, then leave a comment separately, or do either more
        than once over time. Best-effort: a DB hiccup here should never
        break the chat itself, just silently drop that one submission."""
        now = datetime.now(timezone.utc).isoformat()

        def _run(conn):
            self._execute(
                conn,
                "INSERT INTO site_feedback (user_id, rating, comment, created_at) VALUES (?,?,?,?)",
                (user_id, rating, comment, now),
            )
            conn.commit()

        self._safe(_run)

    def clear_all(self) -> None:
        """Delete ALL sessions and turns for EVERY user. Irreversible — admin-only."""
        if not self._connected:
            return
        conn = self._get_conn()
        try:
            self._execute(conn, "DELETE FROM turns")
            self._execute(conn, "DELETE FROM site_feedback")
            self._execute(conn, "DELETE FROM sessions")
            conn.commit()
        finally:
            self._release_conn(conn)


# ── Singleton ─────────────────────────────────────────────────────────────────
_db: ChatHistoryDB | None = None


def get_db() -> ChatHistoryDB:
    """Returns the shared ChatHistoryDB instance."""
    global _db
    if _db is None:
        _db = ChatHistoryDB()
    return _db

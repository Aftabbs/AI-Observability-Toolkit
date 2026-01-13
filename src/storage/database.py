"""Database connection management and schema initialization for SQLite."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from ..config import DATABASE_PATH, SQLITE_PRAGMAS


class Database:
    """SQLite database manager with connection pooling and schema management."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file. If None, uses config default.
        """
        self.db_path = db_path or DATABASE_PATH
        self._ensure_db_directory()
        self._init_schema()

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with pragmas applied.

        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name

        # Apply SQLite pragmas for performance and safety
        for pragma, value in SQLITE_PRAGMAS.items():
            conn.execute(f"PRAGMA {pragma}={value}")

        return conn

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database operations with automatic commit/rollback.

        Yields:
            SQLite cursor object
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def _init_schema(self):
        """Initialize database schema if it doesn't exist."""
        with self.get_cursor() as cursor:
            # Create traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    parent_trace_id TEXT,
                    session_id TEXT,
                    trace_type TEXT NOT NULL,
                    name TEXT,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_trace_id) REFERENCES traces(trace_id)
                )
            """)

            # Create LLM calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT DEFAULT 'groq',
                    prompt TEXT,
                    system_prompt TEXT,
                    response TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    temperature REAL,
                    max_tokens INTEGER,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                )
            """)

            # Create events table for chains/agents
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_name TEXT,
                    timestamp REAL NOT NULL,
                    data TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                )
            """)

            # Create aggregated metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_aggregated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_bucket TEXT NOT NULL,
                    model TEXT,
                    trace_type TEXT,
                    total_requests INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0.0,
                    avg_duration_ms REAL,
                    p50_duration_ms REAL,
                    p95_duration_ms REAL,
                    p99_duration_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(time_bucket, model, trace_type)
                )
            """)

            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    condition_json TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create alert triggers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL,
                    trace_id TEXT,
                    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(id)
                )
            """)

            # Create indexes for performance
            self._create_indexes(cursor)

            # Create full-text search virtual table
            self._create_fts_table(cursor)

    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create indexes for frequently queried columns."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_traces_type ON traces(trace_type)",
            "CREATE INDEX IF NOT EXISTS idx_traces_start_time ON traces(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)",
            "CREATE INDEX IF NOT EXISTS idx_traces_parent ON traces(parent_trace_id)",
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model)",
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_trace ON llm_calls(trace_id)",
            "CREATE INDEX IF NOT EXISTS idx_events_trace ON events(trace_id)",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics_aggregated(time_bucket)",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

    def _create_fts_table(self, cursor: sqlite3.Cursor):
        """Create full-text search table for prompts and responses."""
        try:
            # Check if FTS table already exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='llm_calls_fts'
            """)

            if not cursor.fetchone():
                cursor.execute("""
                    CREATE VIRTUAL TABLE llm_calls_fts USING fts5(
                        trace_id UNINDEXED,
                        prompt,
                        response,
                        content=llm_calls,
                        content_rowid=id
                    )
                """)

                # Create triggers to keep FTS table in sync
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS llm_calls_fts_insert
                    AFTER INSERT ON llm_calls
                    BEGIN
                        INSERT INTO llm_calls_fts(rowid, trace_id, prompt, response)
                        VALUES (new.id, new.trace_id, new.prompt, new.response);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS llm_calls_fts_delete
                    AFTER DELETE ON llm_calls
                    BEGIN
                        INSERT INTO llm_calls_fts(llm_calls_fts, rowid, trace_id, prompt, response)
                        VALUES ('delete', old.id, old.trace_id, old.prompt, old.response);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS llm_calls_fts_update
                    AFTER UPDATE ON llm_calls
                    BEGIN
                        INSERT INTO llm_calls_fts(llm_calls_fts, rowid, trace_id, prompt, response)
                        VALUES ('delete', old.id, old.trace_id, old.prompt, old.response);
                        INSERT INTO llm_calls_fts(rowid, trace_id, prompt, response)
                        VALUES (new.id, new.trace_id, new.prompt, new.response);
                    END
                """)
        except sqlite3.OperationalError:
            # FTS5 might not be available in some SQLite builds
            pass

    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the last row ID.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Last inserted row ID
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.lastrowid

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an UPDATE query and return number of affected rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_delete(self, query: str, params: tuple = ()) -> int:
        """Execute a DELETE query and return number of affected rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount


# Global database instance
_db_instance: Optional[Database] = None


def get_database(db_path: Optional[str] = None) -> Database:
    """Get or create the global database instance.

    Args:
        db_path: Optional path to database file

    Returns:
        Database instance
    """
    global _db_instance
    if _db_instance is None or (db_path and db_path != _db_instance.db_path):
        _db_instance = Database(db_path)
    return _db_instance

"""Repository classes for data access operations."""

from typing import Dict, List, Optional, Any
import time

from .database import get_database, Database
from ..utils import (
    serialize_to_json,
    deserialize_from_json,
    calculate_duration_ms,
    format_time_bucket,
)


class TraceRepository:
    """Repository for trace operations."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def create_trace(
        self,
        trace_id: str,
        trace_type: str,
        name: str,
        start_time: float,
        session_id: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new trace record.

        Args:
            trace_id: Unique trace identifier
            trace_type: Type of trace ('llm', 'chain', 'agent', 'tool')
            name: Name of the operation
            start_time: Start timestamp
            session_id: Optional session identifier
            parent_trace_id: Optional parent trace ID for nested operations
            metadata: Optional metadata dictionary

        Returns:
            trace_id
        """
        query = """
            INSERT INTO traces (
                trace_id, parent_trace_id, session_id, trace_type,
                name, start_time, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        """

        params = (
            trace_id,
            parent_trace_id,
            session_id,
            trace_type,
            name,
            start_time,
            serialize_to_json(metadata) if metadata else None,
        )

        self.db.execute_insert(query, params)
        return trace_id

    def update_trace_completion(
        self,
        trace_id: str,
        end_time: float,
        start_time: float,
        status: str = "success",
        error_message: Optional[str] = None,
    ):
        """Update trace with completion data.

        Args:
            trace_id: Trace identifier
            end_time: End timestamp
            start_time: Start timestamp (for duration calculation)
            status: Status ('success' or 'error')
            error_message: Optional error message if status is 'error'
        """
        duration_ms = calculate_duration_ms(start_time, end_time)

        query = """
            UPDATE traces
            SET end_time = ?, duration_ms = ?, status = ?, error_message = ?
            WHERE trace_id = ?
        """

        params = (end_time, duration_ms, status, error_message, trace_id)
        self.db.execute_update(query, params)

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace dictionary or None
        """
        query = "SELECT * FROM traces WHERE trace_id = ?"
        results = self.db.execute_query(query, (trace_id,))
        return results[0] if results else None

    def get_traces_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all traces for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of trace dictionaries
        """
        query = """
            SELECT * FROM traces
            WHERE session_id = ?
            ORDER BY start_time DESC
        """
        return self.db.execute_query(query, (session_id,))

    def get_traces_by_time_range(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get traces within a time range.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of trace dictionaries
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        if start_time and end_time:
            query = """
                SELECT * FROM traces
                WHERE start_time >= ? AND start_time <= ?
                ORDER BY start_time DESC
            """
            return self.db.execute_query(query, (start_time, end_time))
        elif start_time:
            query = """
                SELECT * FROM traces
                WHERE start_time >= ?
                ORDER BY start_time DESC
            """
            return self.db.execute_query(query, (start_time,))
        else:
            query = "SELECT * FROM traces ORDER BY start_time DESC LIMIT 1000"
            return self.db.execute_query(query)

    def get_child_traces(self, parent_trace_id: str) -> List[Dict[str, Any]]:
        """Get all child traces of a parent trace.

        Args:
            parent_trace_id: Parent trace identifier

        Returns:
            List of child trace dictionaries
        """
        query = """
            SELECT * FROM traces
            WHERE parent_trace_id = ?
            ORDER BY start_time ASC
        """
        return self.db.execute_query(query, (parent_trace_id,))


class LLMCallRepository:
    """Repository for LLM call operations."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def create_llm_call(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cost_usd: float,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: str = "groq",
    ) -> int:
        """Create an LLM call record.

        Args:
            trace_id: Associated trace ID
            model: Model name
            prompt: Input prompt
            response: LLM response
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens
            cost_usd: Cost in USD
            system_prompt: Optional system prompt
            temperature: Optional temperature parameter
            max_tokens: Optional max tokens parameter
            provider: Provider name (default 'groq')

        Returns:
            Row ID of inserted record
        """
        query = """
            INSERT INTO llm_calls (
                trace_id, model, provider, prompt, system_prompt, response,
                input_tokens, output_tokens, total_tokens, cost_usd,
                temperature, max_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            trace_id,
            model,
            provider,
            prompt,
            system_prompt,
            response,
            input_tokens,
            output_tokens,
            total_tokens,
            cost_usd,
            temperature,
            max_tokens,
        )

        return self.db.execute_insert(query, params)

    def get_llm_call(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get LLM call by trace ID.

        Args:
            trace_id: Trace identifier

        Returns:
            LLM call dictionary or None
        """
        query = "SELECT * FROM llm_calls WHERE trace_id = ?"
        results = self.db.execute_query(query, (trace_id,))
        return results[0] if results else None

    def search_llm_calls(self, search_term: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search LLM calls using full-text search.

        Args:
            search_term: Search term
            limit: Maximum number of results

        Returns:
            List of matching LLM call dictionaries
        """
        try:
            query = """
                SELECT llm_calls.*
                FROM llm_calls
                JOIN llm_calls_fts ON llm_calls.id = llm_calls_fts.rowid
                WHERE llm_calls_fts MATCH ?
                LIMIT ?
            """
            return self.db.execute_query(query, (search_term, limit))
        except Exception:
            # Fallback to simple LIKE search if FTS is not available
            query = """
                SELECT * FROM llm_calls
                WHERE prompt LIKE ? OR response LIKE ?
                LIMIT ?
            """
            search_pattern = f"%{search_term}%"
            return self.db.execute_query(query, (search_pattern, search_pattern, limit))


class EventRepository:
    """Repository for event operations (chains/agents)."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def create_event(
        self,
        trace_id: str,
        event_type: str,
        event_name: str,
        timestamp: float,
        data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create an event record.

        Args:
            trace_id: Associated trace ID
            event_type: Type of event ('chain_start', 'tool_start', etc.)
            event_name: Name of the event
            timestamp: Event timestamp
            data: Optional event data dictionary

        Returns:
            Row ID of inserted record
        """
        query = """
            INSERT INTO events (trace_id, event_type, event_name, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
        """

        params = (
            trace_id,
            event_type,
            event_name,
            timestamp,
            serialize_to_json(data) if data else None,
        )

        return self.db.execute_insert(query, params)

    def get_events_for_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all events for a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            List of event dictionaries
        """
        query = """
            SELECT * FROM events
            WHERE trace_id = ?
            ORDER BY timestamp ASC
        """
        return self.db.execute_query(query, (trace_id,))


class MetricsRepository:
    """Repository for aggregated metrics operations."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def upsert_aggregated_metrics(
        self,
        time_bucket: str,
        model: str,
        trace_type: str,
        metrics: Dict[str, Any],
    ):
        """Insert or update aggregated metrics for a time bucket.

        Args:
            time_bucket: Time bucket string (e.g., '2026-01-13-14')
            model: Model name
            trace_type: Type of trace
            metrics: Dictionary with metric values
        """
        query = """
            INSERT INTO metrics_aggregated (
                time_bucket, model, trace_type, total_requests, total_errors,
                total_tokens, total_cost_usd, avg_duration_ms,
                p50_duration_ms, p95_duration_ms, p99_duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(time_bucket, model, trace_type) DO UPDATE SET
                total_requests = excluded.total_requests,
                total_errors = excluded.total_errors,
                total_tokens = excluded.total_tokens,
                total_cost_usd = excluded.total_cost_usd,
                avg_duration_ms = excluded.avg_duration_ms,
                p50_duration_ms = excluded.p50_duration_ms,
                p95_duration_ms = excluded.p95_duration_ms,
                p99_duration_ms = excluded.p99_duration_ms
        """

        params = (
            time_bucket,
            model,
            trace_type,
            metrics.get("total_requests", 0),
            metrics.get("total_errors", 0),
            metrics.get("total_tokens", 0),
            metrics.get("total_cost_usd", 0.0),
            metrics.get("avg_duration_ms", 0.0),
            metrics.get("p50_duration_ms", 0.0),
            metrics.get("p95_duration_ms", 0.0),
            metrics.get("p99_duration_ms", 0.0),
        )

        self.db.execute_insert(query, params)

    def get_metrics_by_time_range(
        self, start_bucket: str, end_bucket: str
    ) -> List[Dict[str, Any]]:
        """Get aggregated metrics for a time range.

        Args:
            start_bucket: Start time bucket
            end_bucket: End time bucket

        Returns:
            List of metric dictionaries
        """
        query = """
            SELECT * FROM metrics_aggregated
            WHERE time_bucket >= ? AND time_bucket <= ?
            ORDER BY time_bucket DESC
        """
        return self.db.execute_query(query, (start_bucket, end_bucket))


class AlertRepository:
    """Repository for alert operations."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def create_alert(
        self, alert_type: str, condition_json: str, is_active: bool = True
    ) -> int:
        """Create an alert configuration.

        Args:
            alert_type: Type of alert ('cost_threshold', 'error_rate', 'latency')
            condition_json: JSON string with alert conditions
            is_active: Whether alert is active

        Returns:
            Alert ID
        """
        query = """
            INSERT INTO alerts (alert_type, condition_json, is_active)
            VALUES (?, ?, ?)
        """
        return self.db.execute_insert(query, (alert_type, condition_json, is_active))

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts.

        Returns:
            List of alert dictionaries
        """
        query = "SELECT * FROM alerts WHERE is_active = 1"
        return self.db.execute_query(query)

    def create_alert_trigger(
        self,
        alert_id: int,
        message: str,
        trace_id: Optional[str] = None,
    ) -> int:
        """Create an alert trigger record.

        Args:
            alert_id: Alert configuration ID
            message: Alert message
            trace_id: Optional associated trace ID

        Returns:
            Trigger ID
        """
        query = """
            INSERT INTO alert_triggers (alert_id, trace_id, message)
            VALUES (?, ?, ?)
        """
        return self.db.execute_insert(query, (alert_id, trace_id, message))

    def get_recent_triggers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert triggers.

        Args:
            limit: Maximum number of triggers to return

        Returns:
            List of trigger dictionaries
        """
        query = """
            SELECT at.*, a.alert_type, a.condition_json
            FROM alert_triggers at
            JOIN alerts a ON at.alert_id = a.id
            ORDER BY at.triggered_at DESC
            LIMIT ?
        """
        return self.db.execute_query(query, (limit,))

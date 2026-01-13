"""Cost tracking and analytics module."""

from typing import Dict, List, Optional, Any
import time

from ..storage.database import get_database, Database
from ..utils import format_time_bucket, safe_divide


class CostTracker:
    """Track and analyze API costs."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def get_total_cost(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> float:
        """Get total cost for a time period.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            model: Filter by model (optional)
            session_id: Filter by session (optional)

        Returns:
            Total cost in USD
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT COALESCE(SUM(llm_calls.cost_usd), 0) as total_cost
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        if model:
            query += " AND llm_calls.model = ?"
            params.append(model)

        if session_id:
            query += " AND traces.session_id = ?"
            params.append(session_id)

        results = self.db.execute_query(query, tuple(params))
        return results[0]["total_cost"] if results else 0.0

    def get_cost_by_model(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get cost breakdown by model.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of dictionaries with model, total_cost, total_requests, total_tokens
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                llm_calls.model,
                COALESCE(SUM(llm_calls.cost_usd), 0) as total_cost,
                COUNT(*) as total_requests,
                COALESCE(SUM(llm_calls.total_tokens), 0) as total_tokens
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " GROUP BY llm_calls.model ORDER BY total_cost DESC"

        return self.db.execute_query(query, tuple(params))

    def get_cost_by_session(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get cost breakdown by session.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of dictionaries with session_id, total_cost, total_requests
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                traces.session_id,
                COALESCE(SUM(llm_calls.cost_usd), 0) as total_cost,
                COUNT(*) as total_requests
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE traces.session_id IS NOT NULL
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " GROUP BY traces.session_id ORDER BY total_cost DESC"

        return self.db.execute_query(query, tuple(params))

    def get_cost_over_time(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        bucket_type: str = "hourly",
    ) -> List[Dict[str, Any]]:
        """Get cost trends over time.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            bucket_type: Time bucket type ('hourly' or 'daily')

        Returns:
            List of dictionaries with time_bucket, total_cost, total_requests
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        # Use aggregated metrics table for better performance
        bucket_format = "%Y-%m-%d-%H" if bucket_type == "hourly" else "%Y-%m-%d"

        query = f"""
            SELECT
                strftime('{bucket_format}', datetime(traces.start_time, 'unixepoch')) as time_bucket,
                COALESCE(SUM(llm_calls.cost_usd), 0) as total_cost,
                COUNT(*) as total_requests
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " GROUP BY time_bucket ORDER BY time_bucket ASC"

        return self.db.execute_query(query, tuple(params))

    def get_token_usage(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get token usage statistics.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            model: Filter by model (optional)

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                COALESCE(SUM(llm_calls.input_tokens), 0) as input_tokens,
                COALESCE(SUM(llm_calls.output_tokens), 0) as output_tokens,
                COALESCE(SUM(llm_calls.total_tokens), 0) as total_tokens
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        if model:
            query += " AND llm_calls.model = ?"
            params.append(model)

        results = self.db.execute_query(query, tuple(params))
        return results[0] if results else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def get_cost_per_request(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        model: Optional[str] = None,
    ) -> float:
        """Get average cost per request.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            model: Filter by model (optional)

        Returns:
            Average cost per request in USD
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                COALESCE(SUM(llm_calls.cost_usd), 0) as total_cost,
                COUNT(*) as total_requests
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        if model:
            query += " AND llm_calls.model = ?"
            params.append(model)

        results = self.db.execute_query(query, tuple(params))

        if results and results[0]["total_requests"] > 0:
            return safe_divide(results[0]["total_cost"], results[0]["total_requests"])
        return 0.0

    def get_most_expensive_requests(
        self,
        limit: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get the most expensive requests.

        Args:
            limit: Number of requests to return
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of expensive request dictionaries
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                traces.trace_id,
                traces.name,
                traces.start_time,
                llm_calls.model,
                llm_calls.total_tokens,
                llm_calls.cost_usd
            FROM llm_calls
            JOIN traces ON llm_calls.trace_id = traces.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " ORDER BY llm_calls.cost_usd DESC LIMIT ?"
        params.append(limit)

        return self.db.execute_query(query, tuple(params))

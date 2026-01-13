"""Latency tracking and analytics module."""

from typing import Dict, List, Optional, Any
import time

from ..storage.database import get_database, Database
from ..utils import safe_divide


class LatencyTracker:
    """Track and analyze latency metrics."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def get_average_latency(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        trace_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> float:
        """Get average latency in milliseconds.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            trace_type: Filter by trace type (optional)
            model: Filter by model (optional)

        Returns:
            Average latency in milliseconds
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT AVG(traces.duration_ms) as avg_latency
            FROM traces
        """

        params = []
        conditions = ["traces.duration_ms IS NOT NULL"]

        if start_time:
            conditions.append("traces.start_time >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("traces.start_time <= ?")
            params.append(end_time)

        if trace_type:
            conditions.append("traces.trace_type = ?")
            params.append(trace_type)

        if model:
            query += " JOIN llm_calls ON traces.trace_id = llm_calls.trace_id"
            conditions.append("llm_calls.model = ?")
            params.append(model)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        results = self.db.execute_query(query, tuple(params))
        return results[0]["avg_latency"] if results and results[0]["avg_latency"] else 0.0

    def get_percentiles(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        trace_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get latency percentiles (P50, P95, P99).

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            trace_type: Filter by trace type (optional)
            model: Filter by model (optional)

        Returns:
            Dictionary with p50, p95, p99 latencies in milliseconds
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT traces.duration_ms
            FROM traces
        """

        params = []
        conditions = ["traces.duration_ms IS NOT NULL"]

        if start_time:
            conditions.append("traces.start_time >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("traces.start_time <= ?")
            params.append(end_time)

        if trace_type:
            conditions.append("traces.trace_type = ?")
            params.append(trace_type)

        if model:
            query += " JOIN llm_calls ON traces.trace_id = llm_calls.trace_id"
            conditions.append("llm_calls.model = ?")
            params.append(model)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY traces.duration_ms ASC"

        results = self.db.execute_query(query, tuple(params))

        if not results:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        durations = [r["duration_ms"] for r in results]
        n = len(durations)

        def percentile(data, p):
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = int(k) + 1
            if c >= len(data):
                return data[f]
            d0 = data[f]
            d1 = data[c]
            return d0 + (d1 - d0) * (k - f)

        return {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
        }

    def get_latency_by_trace_type(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get latency breakdown by trace type.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of dictionaries with trace_type, avg_latency, min_latency, max_latency, count
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                trace_type,
                AVG(duration_ms) as avg_latency,
                MIN(duration_ms) as min_latency,
                MAX(duration_ms) as max_latency,
                COUNT(*) as count
            FROM traces
            WHERE duration_ms IS NOT NULL
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " GROUP BY trace_type ORDER BY avg_latency DESC"

        return self.db.execute_query(query, tuple(params))

    def get_latency_by_model(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get latency breakdown by model.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of dictionaries with model, avg_latency, count
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                llm_calls.model,
                AVG(traces.duration_ms) as avg_latency,
                COUNT(*) as count
            FROM traces
            JOIN llm_calls ON traces.trace_id = llm_calls.trace_id
            WHERE traces.duration_ms IS NOT NULL
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " GROUP BY llm_calls.model ORDER BY avg_latency DESC"

        return self.db.execute_query(query, tuple(params))

    def get_latency_over_time(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        bucket_type: str = "hourly",
    ) -> List[Dict[str, Any]]:
        """Get latency trends over time.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            bucket_type: Time bucket type ('hourly' or 'daily')

        Returns:
            List of dictionaries with time_bucket, avg_latency, count
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        bucket_format = "%Y-%m-%d-%H" if bucket_type == "hourly" else "%Y-%m-%d"

        query = f"""
            SELECT
                strftime('{bucket_format}', datetime(start_time, 'unixepoch')) as time_bucket,
                AVG(duration_ms) as avg_latency,
                COUNT(*) as count
            FROM traces
            WHERE duration_ms IS NOT NULL
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " GROUP BY time_bucket ORDER BY time_bucket ASC"

        return self.db.execute_query(query, tuple(params))

    def get_slowest_requests(
        self,
        limit: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get the slowest requests.

        Args:
            limit: Number of requests to return
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of slow request dictionaries
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                traces.trace_id,
                traces.trace_type,
                traces.name,
                traces.start_time,
                traces.duration_ms
            FROM traces
            WHERE traces.duration_ms IS NOT NULL
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " ORDER BY traces.duration_ms DESC LIMIT ?"
        params.append(limit)

        return self.db.execute_query(query, tuple(params))

    def get_latency_distribution(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        num_buckets: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get latency distribution histogram.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            num_buckets: Number of histogram buckets

        Returns:
            List of dictionaries with bucket_min, bucket_max, count
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        # First, get min and max latencies
        query_minmax = """
            SELECT
                MIN(duration_ms) as min_latency,
                MAX(duration_ms) as max_latency
            FROM traces
            WHERE duration_ms IS NOT NULL
        """

        params = []

        if start_time:
            query_minmax += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query_minmax += " AND start_time <= ?"
            params.append(end_time)

        results = self.db.execute_query(query_minmax, tuple(params))

        if not results or not results[0]["min_latency"]:
            return []

        min_lat = results[0]["min_latency"]
        max_lat = results[0]["max_latency"]
        bucket_width = (max_lat - min_lat) / num_buckets

        # Create histogram buckets
        distribution = []
        for i in range(num_buckets):
            bucket_min = min_lat + (i * bucket_width)
            bucket_max = bucket_min + bucket_width

            query_count = """
                SELECT COUNT(*) as count
                FROM traces
                WHERE duration_ms >= ? AND duration_ms < ?
            """

            count_params = [bucket_min, bucket_max]

            if start_time:
                query_count += " AND start_time >= ?"
                count_params.append(start_time)

            if end_time:
                query_count += " AND start_time <= ?"
                count_params.append(end_time)

            count_result = self.db.execute_query(query_count, tuple(count_params))
            count = count_result[0]["count"] if count_result else 0

            distribution.append({
                "bucket_min": bucket_min,
                "bucket_max": bucket_max,
                "count": count,
            })

        return distribution

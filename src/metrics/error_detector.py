"""Error detection and analysis module."""

from typing import Dict, List, Optional, Any
import time
from collections import Counter

from ..storage.database import get_database, Database
from ..utils import safe_divide


class ErrorDetector:
    """Detect and analyze errors in LLM operations."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def get_error_rate(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        trace_type: Optional[str] = None,
    ) -> float:
        """Get error rate as a percentage.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            trace_type: Filter by trace type (optional)

        Returns:
            Error rate as a percentage (0-100)
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
            FROM traces
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        if trace_type:
            query += " AND trace_type = ?"
            params.append(trace_type)

        results = self.db.execute_query(query, tuple(params))

        if results and results[0]["total_requests"] > 0:
            error_count = results[0]["error_count"]
            total_requests = results[0]["total_requests"]
            return (error_count / total_requests) * 100
        return 0.0

    def get_error_count(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        trace_type: Optional[str] = None,
    ) -> int:
        """Get total number of errors.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            trace_type: Filter by trace type (optional)

        Returns:
            Number of errors
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT COUNT(*) as error_count
            FROM traces
            WHERE status = 'error'
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        if trace_type:
            query += " AND trace_type = ?"
            params.append(trace_type)

        results = self.db.execute_query(query, tuple(params))
        return results[0]["error_count"] if results else 0

    def get_errors_by_type(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get error breakdown by trace type.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of dictionaries with trace_type, error_count, total_count, error_rate
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                trace_type,
                COUNT(*) as total_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
            FROM traces
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " GROUP BY trace_type ORDER BY error_count DESC"

        results = self.db.execute_query(query, tuple(params))

        # Calculate error rate for each type
        for result in results:
            result["error_rate"] = safe_divide(
                result["error_count"], result["total_count"]
            ) * 100

        return results

    def get_recent_errors(
        self,
        limit: int = 50,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent errors.

        Args:
            limit: Number of errors to return
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of error dictionaries
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                trace_id,
                trace_type,
                name,
                start_time,
                duration_ms,
                error_message
            FROM traces
            WHERE status = 'error'
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        return self.db.execute_query(query, tuple(params))

    def get_error_patterns(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        min_occurrences: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get common error patterns.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            min_occurrences: Minimum number of occurrences to include

        Returns:
            List of dictionaries with error_message, count
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                error_message,
                COUNT(*) as count
            FROM traces
            WHERE status = 'error' AND error_message IS NOT NULL
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " GROUP BY error_message HAVING count >= ? ORDER BY count DESC"
        params.append(min_occurrences)

        return self.db.execute_query(query, tuple(params))

    def get_error_rate_over_time(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
        bucket_type: str = "hourly",
    ) -> List[Dict[str, Any]]:
        """Get error rate trends over time.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)
            bucket_type: Time bucket type ('hourly' or 'daily')

        Returns:
            List of dictionaries with time_bucket, total_count, error_count, error_rate
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        bucket_format = "%Y-%m-%d-%H" if bucket_type == "hourly" else "%Y-%m-%d"

        query = f"""
            SELECT
                strftime('{bucket_format}', datetime(start_time, 'unixepoch')) as time_bucket,
                COUNT(*) as total_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
            FROM traces
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time)

        query += " GROUP BY time_bucket ORDER BY time_bucket ASC"

        results = self.db.execute_query(query, tuple(params))

        # Calculate error rate for each time bucket
        for result in results:
            result["error_rate"] = safe_divide(
                result["error_count"], result["total_count"]
            ) * 100

        return results

    def detect_anomalies(
        self,
        hours: int = 24,
        threshold_multiplier: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Detect anomalous error rates using simple statistical method.

        Args:
            hours: Number of hours to analyze
            threshold_multiplier: Multiplier for standard deviation threshold

        Returns:
            List of anomalous time periods
        """
        error_rates = self.get_error_rate_over_time(hours=hours, bucket_type="hourly")

        if len(error_rates) < 3:
            return []

        # Calculate mean and standard deviation
        rates = [r["error_rate"] for r in error_rates]
        mean_rate = sum(rates) / len(rates)

        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        std_dev = variance ** 0.5

        threshold = mean_rate + (threshold_multiplier * std_dev)

        # Find anomalies
        anomalies = []
        for result in error_rates:
            if result["error_rate"] > threshold:
                anomalies.append({
                    "time_bucket": result["time_bucket"],
                    "error_rate": result["error_rate"],
                    "threshold": threshold,
                    "deviation": result["error_rate"] - mean_rate,
                })

        return anomalies

    def get_errors_by_model(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get error breakdown by model.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            List of dictionaries with model, error_count, total_count, error_rate
        """
        if hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        query = """
            SELECT
                llm_calls.model,
                COUNT(*) as total_count,
                SUM(CASE WHEN traces.status = 'error' THEN 1 ELSE 0 END) as error_count
            FROM traces
            JOIN llm_calls ON traces.trace_id = llm_calls.trace_id
            WHERE 1=1
        """

        params = []

        if start_time:
            query += " AND traces.start_time >= ?"
            params.append(start_time)

        if end_time:
            query += " AND traces.start_time <= ?"
            params.append(end_time)

        query += " GROUP BY llm_calls.model ORDER BY error_count DESC"

        results = self.db.execute_query(query, tuple(params))

        # Calculate error rate for each model
        for result in results:
            result["error_rate"] = safe_divide(
                result["error_count"], result["total_count"]
            ) * 100

        return results

    def get_error_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive error summary.

        Args:
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            hours: Number of hours back from now (optional)

        Returns:
            Dictionary with error statistics
        """
        error_rate = self.get_error_rate(start_time, end_time, hours)
        error_count = self.get_error_count(start_time, end_time, hours)
        errors_by_type = self.get_errors_by_type(start_time, end_time, hours)
        recent_errors = self.get_recent_errors(limit=10, start_time=start_time, end_time=end_time, hours=hours)
        error_patterns = self.get_error_patterns(start_time, end_time, hours)

        return {
            "error_rate": error_rate,
            "error_count": error_count,
            "errors_by_type": errors_by_type,
            "recent_errors": recent_errors,
            "common_patterns": error_patterns,
        }

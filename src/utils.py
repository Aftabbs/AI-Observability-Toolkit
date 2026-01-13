"""Utility functions for AI Observability Toolkit."""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


def generate_trace_id() -> str:
    """Generate a unique trace ID using UUID4.

    Returns:
        A unique trace ID string
    """
    return str(uuid.uuid4())


def generate_session_id() -> str:
    """Generate a unique session ID using UUID4.

    Returns:
        A unique session ID string
    """
    return str(uuid.uuid4())


def get_current_timestamp() -> float:
    """Get the current Unix timestamp in seconds.

    Returns:
        Current timestamp as a float
    """
    return time.time()


def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert Unix timestamp to datetime object.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        datetime object
    """
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime object to Unix timestamp.

    Args:
        dt: datetime object

    Returns:
        Unix timestamp as a float
    """
    return dt.timestamp()


def calculate_duration_ms(start_time: float, end_time: float) -> float:
    """Calculate duration in milliseconds between two timestamps.

    Args:
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds

    Returns:
        Duration in milliseconds
    """
    return (end_time - start_time) * 1000


def serialize_to_json(data: Any) -> str:
    """Serialize data to JSON string with error handling.

    Args:
        data: Data to serialize

    Returns:
        JSON string, or empty string if serialization fails
    """
    try:
        return json.dumps(data, default=str, ensure_ascii=False)
    except Exception:
        return "{}"


def deserialize_from_json(json_str: str) -> Dict[str, Any]:
    """Deserialize JSON string to dictionary with error handling.

    Args:
        json_str: JSON string to deserialize

    Returns:
        Dictionary, or empty dict if deserialization fails
    """
    try:
        return json.loads(json_str) if json_str else {}
    except Exception:
        return {}


def truncate_string(text: str, max_length: int) -> str:
    """Truncate string to maximum length, adding ellipsis if truncated.

    Args:
        text: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def sanitize_text(text: str, mask_patterns: Optional[list] = None) -> str:
    """Sanitize text by masking sensitive patterns.

    Args:
        text: Text to sanitize
        mask_patterns: List of regex patterns to mask (not implemented yet)

    Returns:
        Sanitized text
    """
    # Basic sanitization - can be extended with regex patterns
    # For now, just return the text as-is
    # Future: Implement PII masking for emails, phone numbers, API keys, etc.
    return text


def format_time_bucket(timestamp: float, bucket_type: str = "hourly") -> str:
    """Format timestamp into a time bucket string for aggregation.

    Args:
        timestamp: Unix timestamp
        bucket_type: Type of bucket ('hourly', 'daily')

    Returns:
        Time bucket string (e.g., '2026-01-13-14' for hourly)
    """
    dt = timestamp_to_datetime(timestamp)

    if bucket_type == "hourly":
        return dt.strftime("%Y-%m-%d-%H")
    elif bucket_type == "daily":
        return dt.strftime("%Y-%m-%d")
    else:
        return dt.strftime("%Y-%m-%d-%H")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default


def format_cost(cost_usd: float) -> str:
    """Format cost in USD to a readable string.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted cost string (e.g., '$0.0012')
    """
    return f"${cost_usd:.4f}"


def format_duration(duration_ms: float) -> str:
    """Format duration in milliseconds to a readable string.

    Args:
        duration_ms: Duration in milliseconds

    Returns:
        Formatted duration string (e.g., '1234ms' or '1.23s')
    """
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    else:
        return f"{duration_ms / 1000:.2f}s"


def format_tokens(tokens: int) -> str:
    """Format token count to a readable string.

    Args:
        tokens: Number of tokens

    Returns:
        Formatted token string (e.g., '1.2K' or '1.2M')
    """
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1_000_000:
        return f"{tokens / 1000:.1f}K"
    else:
        return f"{tokens / 1_000_000:.1f}M"

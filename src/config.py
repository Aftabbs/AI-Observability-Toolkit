"""Configuration management for AI Observability Toolkit."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "./observability.db")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq Pricing Configuration (per 1M tokens in USD)
# Source: https://groq.com/pricing (as of January 2026)
GROQ_PRICING = {
    "llama3-8b-8192": {
        "input": 0.05,   # $0.05 per 1M input tokens
        "output": 0.08,  # $0.08 per 1M output tokens
    },
    "llama3-70b-8192": {
        "input": 0.59,
        "output": 0.79,
    },
    "llama-3.1-8b-instant": {
        "input": 0.05,
        "output": 0.08,
    },
    "llama-3.1-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
    },
    "llama-3.2-1b-preview": {
        "input": 0.04,
        "output": 0.04,
    },
    "llama-3.2-3b-preview": {
        "input": 0.06,
        "output": 0.06,
    },
    "llama-3.2-11b-vision-preview": {
        "input": 0.18,
        "output": 0.18,
    },
    "llama-3.2-90b-vision-preview": {
        "input": 0.90,
        "output": 0.90,
    },
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
    },
    "mixtral-8x7b-32768": {
        "input": 0.24,
        "output": 0.24,
    },
    "gemma-7b-it": {
        "input": 0.07,
        "output": 0.07,
    },
    "gemma2-9b-it": {
        "input": 0.20,
        "output": 0.20,
    },
}

# Default pricing for unknown models
DEFAULT_PRICING = {
    "input": 0.10,
    "output": 0.10,
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost in USD for a Groq API call.

    Args:
        model: The model name (e.g., 'llama3-8b-8192')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = GROQ_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# Observability Settings
ENABLE_PROMPT_LOGGING = os.getenv("ENABLE_PROMPT_LOGGING", "true").lower() == "true"
ENABLE_RESPONSE_LOGGING = os.getenv("ENABLE_RESPONSE_LOGGING", "true").lower() == "true"
ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "true").lower() == "true"

# Data Retention
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))

# Dashboard Settings
DASHBOARD_AUTO_REFRESH_SECONDS = int(os.getenv("DASHBOARD_AUTO_REFRESH_SECONDS", "5"))

# Trace Settings
MAX_PROMPT_LENGTH = 10000  # Maximum characters to store for prompts
MAX_RESPONSE_LENGTH = 10000  # Maximum characters to store for responses

# Alert Thresholds (defaults)
DEFAULT_ALERT_THRESHOLDS = {
    "cost_per_hour": 10.0,  # $10/hour
    "error_rate_percent": 5.0,  # 5% error rate
    "latency_p95_ms": 5000,  # 5 seconds
}

# SQLite Configuration
SQLITE_PRAGMAS = {
    "journal_mode": "WAL",  # Write-Ahead Logging for better concurrency
    "cache_size": -64000,  # 64MB cache
    "foreign_keys": 1,  # Enable foreign key constraints
    "synchronous": "NORMAL",  # Balance between safety and performance
}

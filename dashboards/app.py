"""Main Streamlit dashboard application for AI Observability Toolkit."""

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time

from src.storage.database import get_database
from src.metrics.cost_tracker import CostTracker
from src.metrics.latency_tracker import LatencyTracker
from src.metrics.error_detector import ErrorDetector
from src.utils import format_cost, format_duration, format_tokens

# Page configuration
st.set_page_config(
    page_title="AI Observability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database and metrics trackers
@st.cache_resource
def init_trackers():
    """Initialize database and metrics trackers."""
    db = get_database()
    cost_tracker = CostTracker(db)
    latency_tracker = LatencyTracker(db)
    error_detector = ErrorDetector(db)
    return db, cost_tracker, latency_tracker, error_detector


db, cost_tracker, latency_tracker, error_detector = init_trackers()

# Sidebar navigation
st.sidebar.title("📊 AI Observability")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📈 Real-time Monitoring",
        "📉 Historical Analytics",
        "🔍 Request Inspector",
        "🚨 Alerts",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
    index=2,
)

# Convert time range to hours
time_range_hours = {
    "Last Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "Last 7 Days": 168,
}
hours = time_range_hours[time_range]

# Auto-refresh setting
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

if auto_refresh:
    time.sleep(5)
    st.rerun()

# Main content
if page == "🏠 Overview":
    st.title("🏠 Overview Dashboard")
    st.markdown(f"**Time Range:** {time_range}")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_cost = cost_tracker.get_total_cost(hours=hours)
        st.metric("Total Cost", format_cost(total_cost))

    with col2:
        token_usage = cost_tracker.get_token_usage(hours=hours)
        st.metric("Total Tokens", format_tokens(token_usage["total_tokens"]))

    with col3:
        avg_latency = latency_tracker.get_average_latency(hours=hours)
        st.metric("Avg Latency", format_duration(avg_latency))

    with col4:
        error_rate = error_detector.get_error_rate(hours=hours)
        st.metric("Error Rate", f"{error_rate:.2f}%")

    # Cost and latency charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("💰 Cost by Model")
        cost_by_model = cost_tracker.get_cost_by_model(hours=hours)
        if cost_by_model:
            import pandas as pd
            df = pd.DataFrame(cost_by_model)
            df['total_cost'] = df['total_cost'].apply(lambda x: round(x, 4))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data available for the selected time range.")

    with col_right:
        st.subheader("⚡ Latency by Model")
        latency_by_model = latency_tracker.get_latency_by_model(hours=hours)
        if latency_by_model:
            import pandas as pd
            df = pd.DataFrame(latency_by_model)
            df['avg_latency'] = df['avg_latency'].apply(lambda x: round(x, 2))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data available for the selected time range.")

elif page == "📈 Real-time Monitoring":
    st.title("📈 Real-time Monitoring")
    st.markdown("**Refreshing every 5 seconds** (enable auto-refresh in sidebar)")

    # Current metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cost_1h = cost_tracker.get_total_cost(hours=1)
        st.metric("Cost (1h)", format_cost(cost_1h))

    with col2:
        tokens_1h = cost_tracker.get_token_usage(hours=1)
        st.metric("Tokens (1h)", format_tokens(tokens_1h["total_tokens"]))

    with col3:
        latency_1h = latency_tracker.get_average_latency(hours=1)
        st.metric("Latency (1h)", format_duration(latency_1h))

    with col4:
        error_rate_1h = error_detector.get_error_rate(hours=1)
        st.metric("Errors (1h)", f"{error_rate_1h:.2f}%")

    # Recent requests
    st.subheader("📋 Recent Requests")
    from src.storage.repositories import TraceRepository, LLMCallRepository
    trace_repo = TraceRepository(db)
    llm_repo = LLMCallRepository(db)

    recent_traces = trace_repo.get_traces_by_time_range(hours=1)

    if recent_traces:
        import pandas as pd
        from datetime import datetime

        # Get LLM call details for each trace
        for trace in recent_traces[:20]:  # Limit to 20 most recent
            trace['timestamp'] = datetime.fromtimestamp(trace['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            trace['duration'] = format_duration(trace['duration_ms']) if trace['duration_ms'] else 'N/A'

            # Get LLM call data if available
            llm_call = llm_repo.get_llm_call(trace['trace_id'])
            if llm_call:
                trace['model'] = llm_call['model']
                trace['tokens'] = llm_call['total_tokens']
                trace['cost'] = format_cost(llm_call['cost_usd'])
            else:
                trace['model'] = 'N/A'
                trace['tokens'] = 0
                trace['cost'] = '$0.0000'

        df = pd.DataFrame(recent_traces[:20])
        display_columns = ['timestamp', 'name', 'model', 'status', 'duration', 'tokens', 'cost']
        st.dataframe(df[display_columns], use_container_width=True)
    else:
        st.info("No recent requests in the last hour.")

elif page == "📉 Historical Analytics":
    st.title("📉 Historical Analytics")
    st.markdown(f"**Time Range:** {time_range}")

    # Percentiles
    st.subheader("⏱️ Latency Percentiles")
    percentiles = latency_tracker.get_percentiles(hours=hours)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P50", format_duration(percentiles['p50']))
    with col2:
        st.metric("P95", format_duration(percentiles['p95']))
    with col3:
        st.metric("P99", format_duration(percentiles['p99']))

    # Time series charts
    st.subheader("📊 Trends Over Time")

    # Cost over time
    cost_over_time = cost_tracker.get_cost_over_time(hours=hours, bucket_type="hourly")
    if cost_over_time:
        import pandas as pd
        import plotly.express as px

        df_cost = pd.DataFrame(cost_over_time)
        fig_cost = px.line(
            df_cost,
            x='time_bucket',
            y='total_cost',
            title='Cost Over Time',
            labels={'total_cost': 'Cost (USD)', 'time_bucket': 'Time'},
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("No cost data available for the selected time range.")

    # Error rate over time
    error_over_time = error_detector.get_error_rate_over_time(hours=hours, bucket_type="hourly")
    if error_over_time:
        import pandas as pd
        import plotly.express as px

        df_errors = pd.DataFrame(error_over_time)
        fig_errors = px.line(
            df_errors,
            x='time_bucket',
            y='error_rate',
            title='Error Rate Over Time',
            labels={'error_rate': 'Error Rate (%)', 'time_bucket': 'Time'},
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    else:
        st.info("No error data available for the selected time range.")

elif page == "🔍 Request Inspector":
    st.title("🔍 Request Inspector")

    # Search functionality
    search_term = st.text_input("🔎 Search prompts/responses", placeholder="Enter search term...")

    from src.storage.repositories import TraceRepository, LLMCallRepository
    trace_repo = TraceRepository(db)
    llm_repo = LLMCallRepository(db)

    if search_term:
        # Full-text search
        results = llm_repo.search_llm_calls(search_term, limit=50)

        if results:
            st.success(f"Found {len(results)} matching requests")

            for llm_call in results:
                with st.expander(f"📝 {llm_call['model']} - {llm_call['trace_id'][:8]}..."):
                    trace = trace_repo.get_trace(llm_call['trace_id'])

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", llm_call['model'])
                    with col2:
                        st.metric("Tokens", llm_call['total_tokens'])
                    with col3:
                        st.metric("Cost", format_cost(llm_call['cost_usd']))
                    with col4:
                        if trace:
                            st.metric("Duration", format_duration(trace['duration_ms']) if trace['duration_ms'] else 'N/A')

                    st.markdown("**Prompt:**")
                    st.code(llm_call['prompt'] if llm_call['prompt'] else 'N/A', language="text")

                    st.markdown("**Response:**")
                    st.code(llm_call['response'] if llm_call['response'] else 'N/A', language="text")
        else:
            st.warning("No results found.")
    else:
        # Show recent requests
        st.info("Enter a search term to search through prompts and responses, or browse recent requests below.")

        recent_traces = trace_repo.get_traces_by_time_range(hours=hours)

        if recent_traces:
            for trace in recent_traces[:10]:
                llm_call = llm_repo.get_llm_call(trace['trace_id'])

                if llm_call:
                    with st.expander(f"📝 {llm_call['model']} - {trace['name']} - {trace['status']}"):
                        from datetime import datetime

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Timestamp", datetime.fromtimestamp(trace['start_time']).strftime('%H:%M:%S'))
                        with col2:
                            st.metric("Tokens", llm_call['total_tokens'])
                        with col3:
                            st.metric("Cost", format_cost(llm_call['cost_usd']))
                        with col4:
                            st.metric("Duration", format_duration(trace['duration_ms']) if trace['duration_ms'] else 'N/A')

                        st.markdown("**Prompt:**")
                        st.code(llm_call['prompt'][:500] if llm_call['prompt'] else 'N/A', language="text")

                        st.markdown("**Response:**")
                        st.code(llm_call['response'][:500] if llm_call['response'] else 'N/A', language="text")
        else:
            st.info("No requests found for the selected time range.")

elif page == "🚨 Alerts":
    st.title("🚨 Alerts Configuration")

    st.info("Alert functionality coming soon! You can manually monitor metrics on other pages.")

    # Show current thresholds
    st.subheader("📋 Current Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        cost_24h = cost_tracker.get_total_cost(hours=24)
        st.metric("Cost (24h)", format_cost(cost_24h))
        st.caption("Threshold: $10/day")

    with col2:
        error_rate_24h = error_detector.get_error_rate(hours=24)
        st.metric("Error Rate (24h)", f"{error_rate_24h:.2f}%")
        st.caption("Threshold: 5%")

    with col3:
        percentiles_24h = latency_tracker.get_percentiles(hours=24)
        st.metric("P95 Latency (24h)", format_duration(percentiles_24h['p95']))
        st.caption("Threshold: 5000ms")

    # Recent errors
    st.subheader("🔴 Recent Errors")
    recent_errors = error_detector.get_recent_errors(limit=10, hours=hours)

    if recent_errors:
        import pandas as pd
        from datetime import datetime

        for error in recent_errors:
            error['timestamp'] = datetime.fromtimestamp(error['start_time']).strftime('%Y-%m-%d %H:%M:%S')

        df = pd.DataFrame(recent_errors)
        display_cols = ['timestamp', 'trace_type', 'name', 'error_message']
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.success("No errors in the selected time range!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**AI Observability Toolkit v0.1.0**")
st.sidebar.markdown("Built with LangChain, Groq & Streamlit")

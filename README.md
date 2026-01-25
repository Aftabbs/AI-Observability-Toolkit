# AI Observability Toolkit

<img width="1105" height="673" alt="image" src="https://github.com/user-attachments/assets/ba0d4f8d-b96e-41ab-b4a5-b45cee69c006" />

    
Comprehensive monitoring, tracing, and debugging toolkit for production AI/LLM applications built with LangChain, LangGraph, and Groq. 

## Features  
 
✅ **Fully Implemented:**
- **Request Tracing**: Automatically track every LLM call, chain, agent, and tool execution 
- **Cost Monitoring**: Real-time token usage and API cost tracking with Groq pricing
- **Latency Analytics**: P50, P95, P99 response time metrics and latency distributions
- **Error Detection**: Automatic error tracking, categorization, and anomaly detection
- **Multi-level Support**: Basic LLM calls, Chains, Agents, and LangGraph workflows
- **Session Tracking**: Group related requests for conversation and user-level analytics
- **Interactive Dashboard**: Streamlit-based real-time monitoring and historical analytics
- **Full-text Search**: Search through all prompts and responses
- **SQLite Storage**: Efficient local database with full-text search capabilities

## Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd ai-observability-toolkit

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Basic Usage

```python
from langchain_groq import ChatGroq
from src import ObservabilityCallback

# Initialize the callback
callback = ObservabilityCallback(
    session_id="my_session_123",
    metadata={"user_id": "user_456"}
)

# Use with any LangChain LLM
llm = ChatGroq(model="llama3-8b-8192", callbacks=[callback])
response = llm.invoke("What is observability?")

# All calls are automatically traced to SQLite database!
```

### 3. Launch Dashboard

```bash
streamlit run dashboards/app.py
```

The dashboard will open in your browser with:
- 🏠 **Overview**: Key metrics and model breakdowns
- 📈 **Real-time Monitoring**: Live requests and current metrics
- 📉 **Historical Analytics**: Trends, charts, and percentiles
- 🔍 **Request Inspector**: Full-text search of prompts/responses
- 🚨 **Alerts**: Error monitoring and thresholds

## Project Structure

```
ai-observability-toolkit/
├── src/
│   ├── tracers/
│   │   ├── observability_callback.py  # Main LangChain callback handler
│   │   └── context.py                 # Trace context management
│   ├── metrics/
│   │   ├── cost_tracker.py            # Cost analytics
│   │   ├── latency_tracker.py         # Latency analytics
│   │   └── error_detector.py          # Error detection
│   ├── storage/
│   │   ├── database.py                # SQLite schema and connection
│   │   └── repositories.py            # Data access layer
│   ├── config.py                      # Groq pricing and settings
│   └── utils.py                       # Helper functions
├── dashboards/
│   └── app.py                         # Streamlit dashboard
├── examples/
│   ├── basic_llm_call.py              # Simple LLM example
│   ├── chain_example.py               # LangChain chain example
│   ├── agent_example.py               # Agent with tools
│   ├── langgraph_example.py           # LangGraph workflow
│   └── session_tracking.py            # Session-based tracking
└── observability.db                   # SQLite database (created automatically)
```

## Examples

### Basic LLM Call

```python
from langchain_groq import ChatGroq
from src import ObservabilityCallback

callback = ObservabilityCallback()
llm = ChatGroq(model="llama3-8b-8192", callbacks=[callback])
response = llm.invoke("Explain AI observability")
```

### LangChain Chain (LCEL)

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src import ObservabilityCallback

callback = ObservabilityCallback()
llm = ChatGroq(model="llama3-8b-8192", callbacks=[callback])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

chain = prompt | llm
result = chain.invoke(
    {"question": "What is LangChain?"},
    config={"callbacks": [callback]}
)
```

### LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from src import ObservabilityCallback

callback = ObservabilityCallback()

# Define your graph nodes and edges...
app = workflow.compile()

result = app.invoke(
    initial_state,
    config={"callbacks": [callback]}
)
```

### Session Tracking

```python
from src import ObservabilityCallback, generate_session_id

session_id = generate_session_id()

callback = ObservabilityCallback(
    session_id=session_id,
    metadata={"user_id": "user_123", "conversation": "support"}
)

# All subsequent LLM calls will be grouped under this session
```

## Run Examples

```bash
# Basic LLM call
python examples/basic_llm_call.py

# Chain example
python examples/chain_example.py

# Agent example (requires langchain hub)
python examples/agent_example.py

# LangGraph example (requires langgraph)
python examples/langgraph_example.py

# Session tracking
python examples/session_tracking.py
```
<img width="1918" height="802" alt="image" src="https://github.com/user-attachments/assets/32747830-17b9-455d-ba79-3e17c362bcc4" />


<img width="1915" height="820" alt="image" src="https://github.com/user-attachments/assets/4984171f-e330-4558-b6dc-b4e8966720b7" />


## Dashboard Features

### Overview Page
- Total cost, tokens, latency, and error rate
- Cost breakdown by model
- Latency breakdown by model

### Real-time Monitoring
- Last hour metrics with auto-refresh
- Recent requests table
- Live cost and token burn rate

### Historical Analytics
- Latency percentiles (P50, P95, P99)
- Cost over time charts
- Error rate trends
- Customizable time ranges

### Request Inspector
- Full-text search across all prompts and responses
- Detailed view of individual requests
- Token counts, costs, and latencies
- Complete prompt and response display

### Alerts
- Current metrics vs thresholds
- Recent errors list
- Error patterns and frequencies

## Use Cases

- 🐛 **Debug Production Failures**: Trace exact inputs/outputs that caused errors
- 💰 **Optimize Costs**: Identify expensive requests and high-token operations
- ⚡ **Improve Performance**: Find slow prompts, chains, and bottlenecks
- 📊 **Monitor Quality**: Track error rates and model performance over time
- 👥 **User Analytics**: Analyze costs and performance per user/session
- 🔔 **Set Up Alerts**: Monitor for cost spikes and error rate increases

## Configuration

Edit `.env` file:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
DATABASE_PATH=./observability.db
ENABLE_PROMPT_LOGGING=true
ENABLE_RESPONSE_LOGGING=true
DATA_RETENTION_DAYS=30
```

## Groq Models Supported

The toolkit includes pricing for all Groq models:
- llama3-8b-8192
- llama3-70b-8192
- llama-3.1-8b-instant
- llama-3.1-70b-versatile
- mixtral-8x7b-32768
- gemma-7b-it
- gemma2-9b-it
- And more...

## Why This Matters

You can't improve what you can't measure. AI systems are especially opaque and unpredictable. This toolkit gives you:

- ✅ **Full Visibility**: See every LLM interaction with complete context
- ✅ **Cost Control**: Track spending in real-time, prevent budget overruns
- ✅ **Performance Insights**: Identify and fix bottlenecks
- ✅ **Production Ready**: SQLite-based, no external dependencies
- ✅ **Zero Friction**: Just add callbacks to existing LangChain code

## License

MIT License - feel free to use in your projects!

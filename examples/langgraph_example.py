"""Example: LangGraph workflow with observability tracing.

This example demonstrates how to use the ObservabilityCallback with LangGraph
state machines for complex multi-step workflows.

Note: Requires langgraph package: pip install langgraph
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict
from langchain_groq import ChatGroq
from src import ObservabilityCallback

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Error: langgraph not installed. Install it with: pip install langgraph")
    sys.exit(1)

# Define the state
class AgentState(TypedDict):
    messages: list
    current_step: str
    result: str

def main():
    """Run a LangGraph workflow with observability."""

    print("=" * 60)
    print("LangGraph Workflow Example with Observability")
    print("=" * 60)

    # Initialize the observability callback
    callback = ObservabilityCallback(
        session_id="langgraph_example_session",
        metadata={"example": "langgraph", "type": "state_machine"},
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        callbacks=[callback],
    )

    # Define workflow nodes
    def analyze_question(state: AgentState) -> AgentState:
        """Analyze the user question."""
        print("Step 1: Analyzing question...")
        state["current_step"] = "analyze"
        state["messages"].append("Question analyzed")
        return state

    def generate_response(state: AgentState) -> AgentState:
        """Generate a response using LLM."""
        print("Step 2: Generating response...")
        response = llm.invoke("Explain observability in AI systems in 2 sentences.")
        state["current_step"] = "generate"
        state["result"] = response.content
        state["messages"].append("Response generated")
        return state

    def format_output(state: AgentState) -> AgentState:
        """Format the final output."""
        print("Step 3: Formatting output...")
        state["current_step"] = "format"
        state["result"] = f"✓ {state['result']}"
        state["messages"].append("Output formatted")
        return state

    # Create the workflow graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_question)
    workflow.add_node("generate", generate_response)
    workflow.add_node("format", format_output)

    # Add edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "format")
    workflow.add_edge("format", END)

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    print("\n--- Workflow Execution ---")

    initial_state = {
        "messages": [],
        "current_step": "",
        "result": "",
    }

    # Execute with callback
    result = app.invoke(
        initial_state,
        config={"callbacks": [callback]}
    )

    print(f"\nFinal Result: {result['result']}")
    print(f"Steps taken: {' -> '.join(result['messages'])}")
    print(f"✓ Workflow traced with all state transitions")

    print("\n" + "=" * 60)
    print("LangGraph workflow has been traced!")
    print("Check the dashboard to see the event timeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()

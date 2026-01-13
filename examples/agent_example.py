"""Example: LangChain agent with observability tracing.

This example demonstrates how to use the ObservabilityCallback with LangChain agents.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from src import ObservabilityCallback

def search_tool(query: str) -> str:
    """Simple mock search tool."""
    return f"Search results for: {query} - Mock result about AI and machine learning"

def calculator_tool(expression: str) -> str:
    """Simple calculator tool."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Run an agent with observability."""

    print("=" * 60)
    print("LangChain Agent Example with Observability")
    print("=" * 60)

    # Initialize the observability callback
    callback = ObservabilityCallback(
        session_id="agent_example_session",
        metadata={"example": "agent", "type": "react"},
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        callbacks=[callback],
    )

    # Create tools
    tools = [
        Tool(
            name="Search",
            func=search_tool,
            description="Useful for searching information online",
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Useful for mathematical calculations",
        ),
    ]

    # Get the ReAct prompt from LangChain hub (or use a simple one)
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Fallback to a simple prompt if hub is unavailable
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "Answer the following question: {input}\nYou have access to: {tools}"
        )

    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[callback],
    )

    # Run agent with a question
    question = "What is 25 * 4, and then search for information about that number?"

    print(f"\nQuestion: {question}")
    print("\n--- Agent Execution ---")

    try:
        result = agent_executor.invoke(
            {"input": question},
            config={"callbacks": [callback]}
        )

        print(f"\nFinal Answer: {result['output']}")
        print(f"✓ Agent execution traced with tool calls")
    except Exception as e:
        print(f"Agent execution failed: {e}")
        print("Note: This may require additional LangChain hub dependencies")

    print("\n" + "=" * 60)
    print("Agent execution has been traced!")
    print("Check the dashboard to see agent actions and tool calls.")
    print("=" * 60)


if __name__ == "__main__":
    main()

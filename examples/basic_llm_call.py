"""Example: Basic LLM call with observability tracing.

This example demonstrates how to use the ObservabilityCallback with a simple
LLM call using Groq's ChatGroq model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from src import ObservabilityCallback

def main():
    """Run a basic LLM call with observability."""

    print("=" * 60)
    print("Basic LLM Call Example with Observability")
    print("=" * 60)

    # Initialize the observability callback
    callback = ObservabilityCallback(
        session_id="example_session_001",
        metadata={"example": "basic_llm_call", "user": "demo_user"},
    )

    # Initialize Groq LLM with callback
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        callbacks=[callback],
    )

    # Make some LLM calls
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "What are the benefits of using LLM observability?",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Request {i} ---")
        print(f"Prompt: {prompt}")

        response = llm.invoke(prompt)

        print(f"Response: {response.content[:200]}...")
        print(f"✓ Traced and logged to database")

    print("\n" + "=" * 60)
    print("All requests have been traced!")
    print("View the dashboard to see metrics:")
    print("  streamlit run dashboards/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

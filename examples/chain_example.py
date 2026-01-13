"""Example: LangChain chain with observability tracing.

This example demonstrates how to use the ObservabilityCallback with LangChain
chains using LCEL (LangChain Expression Language).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src import ObservabilityCallback

def main():
    """Run a chain with observability."""

    print("=" * 60)
    print("LangChain Chain Example with Observability")
    print("=" * 60)

    # Initialize the observability callback
    callback = ObservabilityCallback(
        session_id="chain_example_session",
        metadata={"example": "chain", "type": "sequential"},
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        callbacks=[callback],
    )

    # Create a simple chain using LCEL
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that explains concepts clearly."),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # Run the chain with different questions
    questions = [
        "What is LangChain?",
        "How does observability help in production AI systems?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Chain Execution {i} ---")
        print(f"Question: {question}")

        # The chain invocation will be traced
        response = chain.invoke(
            {"question": question},
            config={"callbacks": [callback]}
        )

        print(f"Response: {response[:200]}...")
        print(f"✓ Chain execution traced with nested operations")

    print("\n" + "=" * 60)
    print("All chain executions have been traced!")
    print("Check the dashboard to see the trace hierarchy.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Example: Session-based tracking with observability.

This example demonstrates how to group multiple requests under a single session
for tracking user interactions and conversation flows.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from src import ObservabilityCallback, generate_session_id
from src.metrics.cost_tracker import CostTracker
from src.storage.database import get_database

def simulate_conversation(session_id: str, user_id: str):
    """Simulate a multi-turn conversation with session tracking."""

    print(f"\n{'=' * 60}")
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    print('=' * 60)

    # Initialize the observability callback with session ID
    callback = ObservabilityCallback(
        session_id=session_id,
        metadata={"user_id": user_id, "conversation_type": "customer_support"},
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        callbacks=[callback],
    )

    # Simulate a conversation
    conversation = [
        "Hello, can you help me understand what observability is?",
        "How does it help in production AI systems?",
        "What are the key metrics I should track?",
        "Thank you for the explanation!",
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {message}")

        response = llm.invoke(message)

        print(f"Assistant: {response.content[:150]}...")

    print(f"\n{'=' * 60}")
    print("Conversation completed!")
    print('=' * 60)

def main():
    """Run session tracking example with multiple users."""

    print("=" * 60)
    print("Session-Based Tracking Example")
    print("=" * 60)

    # Simulate conversations for different users
    users = [
        ("user_alice", "Alice"),
        ("user_bob", "Bob"),
    ]

    for user_id, user_name in users:
        session_id = generate_session_id()
        simulate_conversation(session_id, user_id)

    # Show session analytics
    print("\n" + "=" * 60)
    print("Session Analytics")
    print("=" * 60)

    db = get_database()
    cost_tracker = CostTracker(db)

    # Get cost by session
    session_costs = cost_tracker.get_cost_by_session(hours=1)

    if session_costs:
        print("\nCost breakdown by session:")
        for session in session_costs:
            print(f"  Session: {session['session_id'][:12]}...")
            print(f"    Requests: {session['total_requests']}")
            print(f"    Total Cost: ${session['total_cost']:.4f}")
    else:
        print("\nNo session data found.")

    print("\n" + "=" * 60)
    print("Session tracking complete!")
    print("View the dashboard to see session-level metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()

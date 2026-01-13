"""Trace context management using thread-local storage."""

import threading
from typing import Optional, Dict, Any


class TraceContext:
    """Thread-local trace context for managing trace hierarchy."""

    def __init__(self):
        self._local = threading.local()

    def _ensure_stack(self):
        """Ensure the trace stack exists in thread-local storage."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []

    def _ensure_session(self):
        """Ensure the session ID exists in thread-local storage."""
        if not hasattr(self._local, "session_id"):
            self._local.session_id = None

    def push_trace(
        self,
        trace_id: str,
        trace_type: str,
        name: str,
        start_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Push a new trace onto the context stack.

        Args:
            trace_id: Unique trace identifier
            trace_type: Type of trace ('llm', 'chain', 'agent', 'tool')
            name: Name of the operation
            start_time: Start timestamp
            metadata: Optional metadata
        """
        self._ensure_stack()

        trace_context = {
            "trace_id": trace_id,
            "trace_type": trace_type,
            "name": name,
            "start_time": start_time,
            "parent_trace_id": self.get_current_trace_id(),
            "metadata": metadata or {},
        }

        self._local.stack.append(trace_context)

    def pop_trace(self) -> Optional[Dict[str, Any]]:
        """Pop the current trace from the context stack.

        Returns:
            Trace context dictionary or None if stack is empty
        """
        self._ensure_stack()

        if self._local.stack:
            return self._local.stack.pop()
        return None

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID (top of stack).

        Returns:
            Current trace ID or None if stack is empty
        """
        self._ensure_stack()

        if self._local.stack:
            return self._local.stack[-1]["trace_id"]
        return None

    def get_current_trace(self) -> Optional[Dict[str, Any]]:
        """Get the current trace context (top of stack).

        Returns:
            Current trace context dictionary or None if stack is empty
        """
        self._ensure_stack()

        if self._local.stack:
            return self._local.stack[-1]
        return None

    def get_parent_trace_id(self) -> Optional[str]:
        """Get the parent trace ID of the current trace.

        Returns:
            Parent trace ID or None if no parent
        """
        trace = self.get_current_trace()
        return trace["parent_trace_id"] if trace else None

    def set_session_id(self, session_id: str):
        """Set the session ID for the current thread.

        Args:
            session_id: Session identifier
        """
        self._ensure_session()
        self._local.session_id = session_id

    def get_session_id(self) -> Optional[str]:
        """Get the session ID for the current thread.

        Returns:
            Session ID or None if not set
        """
        self._ensure_session()
        return self._local.session_id

    def clear(self):
        """Clear the entire trace context for the current thread."""
        self._local.stack = []
        self._local.session_id = None

    def get_stack_depth(self) -> int:
        """Get the current depth of the trace stack.

        Returns:
            Number of traces in the stack
        """
        self._ensure_stack()
        return len(self._local.stack)


# Global trace context instance
_trace_context = TraceContext()


def get_trace_context() -> TraceContext:
    """Get the global trace context instance.

    Returns:
        TraceContext instance
    """
    return _trace_context

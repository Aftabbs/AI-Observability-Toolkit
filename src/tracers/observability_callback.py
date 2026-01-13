"""Main observability callback handler for LangChain/LangGraph integration."""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish

from ..storage.database import get_database
from ..storage.repositories import (
    TraceRepository,
    LLMCallRepository,
    EventRepository,
)
from ..config import (
    calculate_cost,
    ENABLE_PROMPT_LOGGING,
    ENABLE_RESPONSE_LOGGING,
    MAX_PROMPT_LENGTH,
    MAX_RESPONSE_LENGTH,
)
from ..utils import generate_trace_id, get_current_timestamp, truncate_string
from .context import get_trace_context


class ObservabilityCallback(BaseCallbackHandler):
    """Callback handler for observability of LangChain/LangGraph operations.

    This callback captures:
    - All LLM calls with inputs, outputs, tokens, and costs
    - Chain executions with nested trace hierarchy
    - Agent actions and tool calls
    - Errors and exceptions
    """

    def __init__(
        self,
        database_path: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the observability callback.

        Args:
            database_path: Optional path to SQLite database
            session_id: Optional session identifier for grouping traces
            metadata: Optional metadata to attach to all traces
        """
        super().__init__()

        # Initialize database and repositories
        self.db = get_database(database_path)
        self.trace_repo = TraceRepository(self.db)
        self.llm_repo = LLMCallRepository(self.db)
        self.event_repo = EventRepository(self.db)

        # Initialize trace context
        self.context = get_trace_context()

        # Set session ID if provided
        if session_id:
            self.context.set_session_id(session_id)

        # Store global metadata
        self.global_metadata = metadata or {}

        # Track run IDs to trace IDs mapping
        self._run_id_to_trace_id: Dict[str, str] = {}
        self._run_start_times: Dict[str, float] = {}

    def _get_or_create_trace_id(self, run_id: UUID) -> str:
        """Get or create a trace ID for a run ID.

        Args:
            run_id: LangChain run ID

        Returns:
            Trace ID
        """
        run_id_str = str(run_id)
        if run_id_str not in self._run_id_to_trace_id:
            self._run_id_to_trace_id[run_id_str] = generate_trace_id()
        return self._run_id_to_trace_id[run_id_str]

    def _safe_execute(self, func, *args, **kwargs):
        """Safely execute a function, catching and logging any errors.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        try:
            func(*args, **kwargs)
        except Exception as e:
            # Never break LLM execution due to callback errors
            print(f"Observability callback error: {e}")

    # LLM Callbacks

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""

        def _on_llm_start():
            trace_id = self._get_or_create_trace_id(run_id)
            start_time = get_current_timestamp()
            self._run_start_times[str(run_id)] = start_time

            # Get parent trace ID from context or parent run ID
            parent_trace_id = (
                self._get_or_create_trace_id(parent_run_id)
                if parent_run_id
                else self.context.get_current_trace_id()
            )

            # Get model name
            model = serialized.get("name", serialized.get("id", ["unknown"])[-1])

            # Combine metadata
            combined_metadata = {**self.global_metadata}
            if metadata:
                combined_metadata.update(metadata)
            if tags:
                combined_metadata["tags"] = tags

            # Create trace
            self.trace_repo.create_trace(
                trace_id=trace_id,
                trace_type="llm",
                name=f"llm_{model}",
                start_time=start_time,
                session_id=self.context.get_session_id(),
                parent_trace_id=parent_trace_id,
                metadata=combined_metadata,
            )

            # Push to context stack
            self.context.push_trace(
                trace_id=trace_id,
                trace_type="llm",
                name=f"llm_{model}",
                start_time=start_time,
                metadata=combined_metadata,
            )

        self._safe_execute(_on_llm_start)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes."""

        def _on_llm_end():
            trace_id = self._get_or_create_trace_id(run_id)
            end_time = get_current_timestamp()
            start_time = self._run_start_times.get(str(run_id), end_time)

            # Pop from context stack
            self.context.pop_trace()

            # Extract token usage and model info
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})

            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

            model = llm_output.get("model_name", llm_output.get("model", "unknown"))

            # Calculate cost
            cost_usd = calculate_cost(model, input_tokens, output_tokens)

            # Extract prompt and response
            prompt = ""
            system_prompt = None
            response_text = ""

            if response.generations and len(response.generations) > 0:
                if len(response.generations[0]) > 0:
                    response_text = response.generations[0][0].text

            # Try to get prompt from kwargs or metadata
            if hasattr(response, "prompts") and response.prompts:
                prompt = response.prompts[0] if response.prompts else ""

            # Truncate if needed
            if ENABLE_PROMPT_LOGGING:
                prompt = truncate_string(prompt, MAX_PROMPT_LENGTH)
            else:
                prompt = "[logging disabled]"

            if ENABLE_RESPONSE_LOGGING:
                response_text = truncate_string(response_text, MAX_RESPONSE_LENGTH)
            else:
                response_text = "[logging disabled]"

            # Update trace with completion
            self.trace_repo.update_trace_completion(
                trace_id=trace_id,
                end_time=end_time,
                start_time=start_time,
                status="success",
            )

            # Create LLM call record
            self.llm_repo.create_llm_call(
                trace_id=trace_id,
                model=model,
                prompt=prompt,
                response=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                system_prompt=system_prompt,
                provider="groq",
            )

            # Cleanup
            del self._run_id_to_trace_id[str(run_id)]
            del self._run_start_times[str(run_id)]

        self._safe_execute(_on_llm_end)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM encounters an error."""

        def _on_llm_error():
            trace_id = self._get_or_create_trace_id(run_id)
            end_time = get_current_timestamp()
            start_time = self._run_start_times.get(str(run_id), end_time)

            # Pop from context stack
            self.context.pop_trace()

            # Update trace with error
            self.trace_repo.update_trace_completion(
                trace_id=trace_id,
                end_time=end_time,
                start_time=start_time,
                status="error",
                error_message=str(error),
            )

            # Cleanup
            if str(run_id) in self._run_id_to_trace_id:
                del self._run_id_to_trace_id[str(run_id)]
            if str(run_id) in self._run_start_times:
                del self._run_start_times[str(run_id)]

        self._safe_execute(_on_llm_error)

    # Chain Callbacks

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""

        def _on_chain_start():
            trace_id = self._get_or_create_trace_id(run_id)
            start_time = get_current_timestamp()
            self._run_start_times[str(run_id)] = start_time

            # Get parent trace ID
            parent_trace_id = (
                self._get_or_create_trace_id(parent_run_id)
                if parent_run_id
                else self.context.get_current_trace_id()
            )

            # Get chain name
            chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

            # Combine metadata
            combined_metadata = {**self.global_metadata}
            if metadata:
                combined_metadata.update(metadata)
            if tags:
                combined_metadata["tags"] = tags

            # Create trace
            self.trace_repo.create_trace(
                trace_id=trace_id,
                trace_type="chain",
                name=f"chain_{chain_name}",
                start_time=start_time,
                session_id=self.context.get_session_id(),
                parent_trace_id=parent_trace_id,
                metadata=combined_metadata,
            )

            # Push to context stack
            self.context.push_trace(
                trace_id=trace_id,
                trace_type="chain",
                name=f"chain_{chain_name}",
                start_time=start_time,
                metadata=combined_metadata,
            )

            # Log chain start event
            self.event_repo.create_event(
                trace_id=trace_id,
                event_type="chain_start",
                event_name=chain_name,
                timestamp=start_time,
                data={"inputs": inputs} if ENABLE_PROMPT_LOGGING else {},
            )

        self._safe_execute(_on_chain_start)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain completes."""

        def _on_chain_end():
            trace_id = self._get_or_create_trace_id(run_id)
            end_time = get_current_timestamp()
            start_time = self._run_start_times.get(str(run_id), end_time)

            # Pop from context stack
            self.context.pop_trace()

            # Update trace with completion
            self.trace_repo.update_trace_completion(
                trace_id=trace_id,
                end_time=end_time,
                start_time=start_time,
                status="success",
            )

            # Log chain end event
            self.event_repo.create_event(
                trace_id=trace_id,
                event_type="chain_end",
                event_name="chain_completed",
                timestamp=end_time,
                data={"outputs": outputs} if ENABLE_RESPONSE_LOGGING else {},
            )

            # Cleanup
            if str(run_id) in self._run_id_to_trace_id:
                del self._run_id_to_trace_id[str(run_id)]
            if str(run_id) in self._run_start_times:
                del self._run_start_times[str(run_id)]

        self._safe_execute(_on_chain_end)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain encounters an error."""

        def _on_chain_error():
            trace_id = self._get_or_create_trace_id(run_id)
            end_time = get_current_timestamp()
            start_time = self._run_start_times.get(str(run_id), end_time)

            # Pop from context stack
            self.context.pop_trace()

            # Update trace with error
            self.trace_repo.update_trace_completion(
                trace_id=trace_id,
                end_time=end_time,
                start_time=start_time,
                status="error",
                error_message=str(error),
            )

            # Cleanup
            if str(run_id) in self._run_id_to_trace_id:
                del self._run_id_to_trace_id[str(run_id)]
            if str(run_id) in self._run_start_times:
                del self._run_start_times[str(run_id)]

        self._safe_execute(_on_chain_error)

    # Tool Callbacks

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""

        def _on_tool_start():
            trace_id = self._get_or_create_trace_id(run_id)
            start_time = get_current_timestamp()
            self._run_start_times[str(run_id)] = start_time

            # Get parent trace ID (usually from agent)
            parent_trace_id = (
                self._get_or_create_trace_id(parent_run_id)
                if parent_run_id
                else self.context.get_current_trace_id()
            )

            # Get tool name
            tool_name = serialized.get("name", "unknown_tool")

            # Combine metadata
            combined_metadata = {**self.global_metadata}
            if metadata:
                combined_metadata.update(metadata)
            if tags:
                combined_metadata["tags"] = tags

            # Create trace
            self.trace_repo.create_trace(
                trace_id=trace_id,
                trace_type="tool",
                name=f"tool_{tool_name}",
                start_time=start_time,
                session_id=self.context.get_session_id(),
                parent_trace_id=parent_trace_id,
                metadata=combined_metadata,
            )

            # Log tool start event
            self.event_repo.create_event(
                trace_id=trace_id,
                event_type="tool_start",
                event_name=tool_name,
                timestamp=start_time,
                data={"input": input_str} if ENABLE_PROMPT_LOGGING else {},
            )

        self._safe_execute(_on_tool_start)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool completes."""

        def _on_tool_end():
            trace_id = self._get_or_create_trace_id(run_id)
            end_time = get_current_timestamp()
            start_time = self._run_start_times.get(str(run_id), end_time)

            # Update trace with completion
            self.trace_repo.update_trace_completion(
                trace_id=trace_id,
                end_time=end_time,
                start_time=start_time,
                status="success",
            )

            # Log tool end event
            self.event_repo.create_event(
                trace_id=trace_id,
                event_type="tool_end",
                event_name="tool_completed",
                timestamp=end_time,
                data={"output": output} if ENABLE_RESPONSE_LOGGING else {},
            )

            # Cleanup
            if str(run_id) in self._run_id_to_trace_id:
                del self._run_id_to_trace_id[str(run_id)]
            if str(run_id) in self._run_start_times:
                del self._run_start_times[str(run_id)]

        self._safe_execute(_on_tool_end)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool encounters an error."""

        def _on_tool_error():
            trace_id = self._get_or_create_trace_id(run_id)
            end_time = get_current_timestamp()
            start_time = self._run_start_times.get(str(run_id), end_time)

            # Update trace with error
            self.trace_repo.update_trace_completion(
                trace_id=trace_id,
                end_time=end_time,
                start_time=start_time,
                status="error",
                error_message=str(error),
            )

            # Cleanup
            if str(run_id) in self._run_id_to_trace_id:
                del self._run_id_to_trace_id[str(run_id)]
            if str(run_id) in self._run_start_times:
                del self._run_start_times[str(run_id)]

        self._safe_execute(_on_tool_error)

    # Agent Callbacks

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""

        def _on_agent_action():
            # Log agent action as an event under the current trace
            current_trace_id = (
                self._get_or_create_trace_id(parent_run_id)
                if parent_run_id
                else self.context.get_current_trace_id()
            )

            if current_trace_id:
                self.event_repo.create_event(
                    trace_id=current_trace_id,
                    event_type="agent_action",
                    event_name=action.tool,
                    timestamp=get_current_timestamp(),
                    data={
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                        "log": action.log,
                    }
                    if ENABLE_PROMPT_LOGGING
                    else {},
                )

        self._safe_execute(_on_agent_action)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""

        def _on_agent_finish():
            # Log agent finish as an event under the current trace
            current_trace_id = (
                self._get_or_create_trace_id(parent_run_id)
                if parent_run_id
                else self.context.get_current_trace_id()
            )

            if current_trace_id:
                self.event_repo.create_event(
                    trace_id=current_trace_id,
                    event_type="agent_finish",
                    event_name="agent_completed",
                    timestamp=get_current_timestamp(),
                    data={"return_values": finish.return_values, "log": finish.log}
                    if ENABLE_RESPONSE_LOGGING
                    else {},
                )

        self._safe_execute(_on_agent_finish)

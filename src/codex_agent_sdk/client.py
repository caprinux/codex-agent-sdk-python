"""Public ``CodexSDKClient`` — multi-turn, interactive client."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from codex_agent_sdk._internal.client import InternalClient
from codex_agent_sdk.types import (
    AgentMessageItem,
    CodexAgentOptions,
    ItemCompletedEvent,
    ThreadEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnResult,
    UserInput,
)


class CodexSDKClient:
    """Multi-turn client for interactive Codex conversations.

    Each call to :meth:`run` or :meth:`run_streamed` executes one turn.
    The client automatically captures the ``thread_id`` from the first turn
    and passes it to subsequent turns via ``codex exec resume <thread_id>``.

    Usage::

        async with CodexSDKClient(options) as client:
            # First turn
            result = await client.run("Set up a new Python project")
            print(result.final_response)

            # Follow-up turn (automatically resumes the thread)
            result = await client.run("Now add a test suite")
            print(result.final_response)

    The client can also be used without ``async with``::

        client = CodexSDKClient(options)
        result = await client.run("Hello")
    """

    def __init__(self, options: Optional[CodexAgentOptions] = None) -> None:
        self._options = options or CodexAgentOptions()
        self._internal = InternalClient(self._options)
        self._thread_id: Optional[str] = None

    @property
    def thread_id(self) -> Optional[str]:
        """The thread ID from the most recent session, if any."""
        return self._thread_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> CodexSDKClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass  # No persistent process to tear down — each turn is a new subprocess

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    async def run_streamed(
        self,
        prompt: UserInput,
        *,
        resume_thread_id: Optional[str] = None,
    ) -> AsyncIterator[ThreadEvent]:
        """Execute one turn and yield events as they arrive.

        Parameters
        ----------
        prompt:
            The user prompt for this turn.
        resume_thread_id:
            Explicit thread ID to resume. If ``None``, uses the thread ID
            captured from the previous turn (if any).
        """
        tid = resume_thread_id or self._thread_id
        async for event in self._internal.process_query(prompt, resume_thread_id=tid):
            # Capture the thread ID for subsequent turns
            if isinstance(event, ThreadStartedEvent) and event.thread_id:
                self._thread_id = event.thread_id
            yield event

    # ------------------------------------------------------------------
    # Blocking (buffered) interface
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: UserInput,
        *,
        resume_thread_id: Optional[str] = None,
    ) -> TurnResult:
        """Execute one turn and return an aggregated :class:`TurnResult`.

        This buffers all events internally and returns when the turn
        completes (or fails).
        """
        result = TurnResult()
        async for event in self.run_streamed(prompt, resume_thread_id=resume_thread_id):
            result.events.append(event)

            if isinstance(event, ThreadStartedEvent):
                result.thread_id = event.thread_id

            elif isinstance(event, ItemCompletedEvent):
                result.items.append(event.item)
                # Capture the final agent message
                if isinstance(event.item, AgentMessageItem):
                    result.final_response = event.item.text

            elif isinstance(event, TurnCompletedEvent):
                result.usage = event.usage

            elif isinstance(event, TurnFailedEvent):
                from codex_agent_sdk._errors import CodexSDKError

                raise CodexSDKError(f"Turn failed: {event.error.message}")

        return result

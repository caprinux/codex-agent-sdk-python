"""Public ``query()`` function — the simplest way to use the Codex SDK."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from codex_agent_sdk._internal.client import InternalClient
from codex_agent_sdk.types import (
    CodexAgentOptions,
    ThreadEvent,
    UserInput,
)


async def query(
    prompt: UserInput,
    *,
    options: Optional[CodexAgentOptions] = None,
    resume_thread_id: Optional[str] = None,
) -> AsyncIterator[ThreadEvent]:
    """Run a single Codex turn and yield events.

    This is the simplest entry point into the SDK. It spawns a ``codex exec``
    process, sends *prompt*, and yields :class:`ThreadEvent` objects as they
    arrive.

    Parameters
    ----------
    prompt:
        The user prompt — either a plain string or a sequence of
        :class:`TextInput` / :class:`ImageInput` objects.
    options:
        Configuration for the Codex CLI. ``None`` uses all defaults.
    resume_thread_id:
        If provided, resumes an existing thread instead of starting a new one.

    Yields
    ------
    ThreadEvent
        Events emitted by the Codex CLI (thread lifecycle, items, errors).

    Example
    -------
    ::

        from codex_agent_sdk import query, ItemCompletedEvent, AgentMessageItem

        async for event in query("What files are in this directory?"):
            if isinstance(event, ItemCompletedEvent):
                if isinstance(event.item, AgentMessageItem):
                    print(event.item.text)
    """
    opts = options or CodexAgentOptions()
    client = InternalClient(opts)
    async for event in client.process_query(
        prompt,
        resume_thread_id=resume_thread_id,
    ):
        yield event

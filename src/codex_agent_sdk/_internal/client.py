"""Internal client used by the public ``query()`` function."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from codex_agent_sdk._internal.query import Query
from codex_agent_sdk.types import (
    CodexAgentOptions,
    ThreadEvent,
    UserInput,
)


class InternalClient:
    """Low-level client that creates a :class:`Query` and yields events.

    This is the bridge between the public ``query()`` async generator and
    the internal transport machinery.
    """

    def __init__(self, options: CodexAgentOptions) -> None:
        self._options = options

    async def process_query(
        self,
        prompt: UserInput,
        *,
        resume_thread_id: Optional[str] = None,
    ) -> AsyncIterator[ThreadEvent]:
        """Run a single turn and yield events."""
        query = Query(
            self._options,
            prompt,
            resume_thread_id=resume_thread_id,
        )
        try:
            await query.start()
            await query.send_prompt()
            async for event in query.receive_events():
                yield event
        finally:
            await query.close()

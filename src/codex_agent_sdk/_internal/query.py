"""Query manages a single Codex CLI turn — transport lifecycle + event streaming."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from codex_agent_sdk._internal.message_parser import parse_event
from codex_agent_sdk._internal.transport._base import Transport
from codex_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from codex_agent_sdk.types import (
    CodexAgentOptions,
    ThreadEvent,
    UserInput,
)


class Query:
    """Drives a single turn against the Codex CLI.

    Lifecycle:
    1. ``start()``   — spawn the CLI process
    2. ``send_prompt()`` — write the prompt to stdin and close it
    3. ``receive_events()`` — async-iterate over parsed events
    4. ``close()``   — tear down
    """

    def __init__(
        self,
        options: CodexAgentOptions,
        prompt: UserInput,
        *,
        resume_thread_id: Optional[str] = None,
    ) -> None:
        self._transport: Transport = SubprocessCLITransport(
            options,
            prompt,
            resume_thread_id=resume_thread_id,
        )
        self._prompt = prompt
        self._started = False

    async def start(self) -> None:
        """Spawn the CLI process."""
        await self._transport.connect()
        self._started = True

    async def send_prompt(self) -> None:
        """Write the prompt text to stdin and close it."""
        if not self._started:
            raise RuntimeError("Query has not been started")

        # Codex CLI reads the prompt from stdin, then we close stdin
        # to signal that input is complete.
        prompt_text = self._get_prompt_text()
        await self._transport.write(prompt_text)
        await self._transport.end_input()

    async def receive_events(self) -> AsyncIterator[ThreadEvent]:
        """Yield typed events from the CLI's stdout."""
        async for raw in self._transport.read_messages():
            yield parse_event(raw)

    async def close(self) -> None:
        """Shut down the transport."""
        await self._transport.close()

    def _get_prompt_text(self) -> str:
        if isinstance(self._prompt, str):
            return self._prompt
        from codex_agent_sdk.types import TextInput

        parts: list[str] = []
        for part in self._prompt:
            if isinstance(part, TextInput):
                parts.append(part.text)
        return "\n\n".join(parts)

"""Tests for the Query class lifecycle."""

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, Mock

import anyio

from codex_agent_sdk._internal.query import Query
from codex_agent_sdk._internal.transport._base import Transport
from codex_agent_sdk.types import CodexAgentOptions


class MockTransport(Transport):
    """Concrete transport that records writes and plays back canned messages."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages_to_read = messages or []
        self.written_data: list[str] = []
        self._connected = False
        self._input_ended = False
        self._closed = False

    async def connect(self) -> None:
        self._connected = True

    async def write(self, data: str) -> None:
        self.written_data.append(data)

    async def end_input(self) -> None:
        self._input_ended = True

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        for msg in self.messages_to_read:
            yield msg

    async def close(self) -> None:
        self._closed = True

    def is_ready(self) -> bool:
        return self._connected


class TestQueryLifecycle:
    def test_start_connects_transport(self) -> None:
        async def _test() -> None:
            transport = MockTransport()
            query = Query.__new__(Query)
            query._transport = transport
            query._prompt = "test"
            query._started = False

            await query.start()
            assert transport._connected is True
            assert query._started is True

        anyio.run(_test)

    def test_send_prompt_writes_and_closes(self) -> None:
        async def _test() -> None:
            transport = MockTransport()
            query = Query.__new__(Query)
            query._transport = transport
            query._prompt = "Hello, Codex!"
            query._started = True

            await query.send_prompt()
            assert transport.written_data == ["Hello, Codex!"]
            assert transport._input_ended is True

        anyio.run(_test)

    def test_send_prompt_raises_if_not_started(self) -> None:
        async def _test() -> None:
            query = Query.__new__(Query)
            query._started = False
            try:
                await query.send_prompt()
                assert False, "Should have raised"
            except RuntimeError as e:
                assert "not been started" in str(e)

        anyio.run(_test)

    def test_receive_events_yields_parsed_events(self) -> None:
        async def _test() -> None:
            messages = [
                {"type": "thread.started", "thread_id": "tid-1"},
                {"type": "turn.started"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "id": "m1", "text": "Done"},
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5},
                },
            ]
            transport = MockTransport(messages)
            query = Query.__new__(Query)
            query._transport = transport
            query._prompt = "test"
            query._started = True

            events = []
            async for event in query.receive_events():
                events.append(event)

            assert len(events) == 4
            assert events[0].type == "thread.started"
            assert events[1].type == "turn.started"
            assert events[2].type == "item.completed"
            assert events[3].type == "turn.completed"

        anyio.run(_test)

    def test_close_closes_transport(self) -> None:
        async def _test() -> None:
            transport = MockTransport()
            query = Query.__new__(Query)
            query._transport = transport

            await query.close()
            assert transport._closed is True

        anyio.run(_test)

    def test_prompt_text_from_string(self) -> None:
        query = Query.__new__(Query)
        query._prompt = "Hello world"
        assert query._get_prompt_text() == "Hello world"

    def test_prompt_text_from_structured(self) -> None:
        from codex_agent_sdk.types import ImageInput, TextInput

        query = Query.__new__(Query)
        query._prompt = [TextInput(text="Part 1"), ImageInput(path="/img.png"), TextInput(text="Part 2")]
        assert query._get_prompt_text() == "Part 1\n\nPart 2"

    def test_empty_messages(self) -> None:
        async def _test() -> None:
            transport = MockTransport([])
            query = Query.__new__(Query)
            query._transport = transport
            query._prompt = "test"
            query._started = True

            events = []
            async for event in query.receive_events():
                events.append(event)

            assert len(events) == 0

        anyio.run(_test)

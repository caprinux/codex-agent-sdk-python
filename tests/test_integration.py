"""Integration tests with mocked transport — end-to-end through query() and InternalClient."""

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import anyio

from codex_agent_sdk._internal.client import InternalClient
from codex_agent_sdk.query import query
from codex_agent_sdk.types import (
    AgentMessageItem,
    CodexAgentOptions,
    CommandExecutionItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    ThreadEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)


def _make_mock_transport(messages: list[dict[str, Any]]) -> AsyncMock:
    """Create a mock transport that yields the given messages."""
    mock_transport = AsyncMock()

    async def mock_read() -> AsyncIterator[dict[str, Any]]:
        for msg in messages:
            yield msg

    mock_transport.read_messages = mock_read
    mock_transport.connect = AsyncMock()
    mock_transport.close = AsyncMock()
    mock_transport.end_input = AsyncMock()
    mock_transport.write = AsyncMock()
    mock_transport.is_ready = Mock(return_value=True)
    return mock_transport


# A typical successful turn's JSONL output
SIMPLE_TURN_MESSAGES: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "session-001"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {"type": "agent_message", "id": "msg_1", "text": "Hello! I can help with that."},
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 500, "cached_input_tokens": 200, "output_tokens": 50},
    },
]

TOOL_USE_MESSAGES: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "session-002"},
    {"type": "turn.started"},
    {
        "type": "item.started",
        "item": {
            "type": "command_execution",
            "id": "cmd_1",
            "command": "ls -la",
            "aggregated_output": "",
            "status": "in_progress",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "id": "cmd_1",
            "command": "ls -la",
            "aggregated_output": "total 42\n-rw-r--r-- 1 user user 100 main.py\n",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "id": "msg_1",
            "text": "The directory contains main.py.",
        },
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 100},
    },
]


class TestQueryFunction:
    def test_simple_query(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(SIMPLE_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                events: list[ThreadEvent] = []
                async for event in query("Hello"):
                    events.append(event)

                assert len(events) == 4
                assert isinstance(events[0], ThreadStartedEvent)
                assert events[0].thread_id == "session-001"
                assert isinstance(events[1], TurnStartedEvent)
                assert isinstance(events[2], ItemCompletedEvent)
                assert isinstance(events[2].item, AgentMessageItem)
                assert events[2].item.text == "Hello! I can help with that."
                assert isinstance(events[3], TurnCompletedEvent)
                assert events[3].usage.input_tokens == 500

        anyio.run(_test)

    def test_query_with_tool_use(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TOOL_USE_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                events: list[ThreadEvent] = []
                async for event in query("What files are here?"):
                    events.append(event)

                assert len(events) == 6

                # Check command execution events
                assert isinstance(events[2], ItemStartedEvent)
                assert isinstance(events[2].item, CommandExecutionItem)
                assert events[2].item.command == "ls -la"

                assert isinstance(events[3], ItemCompletedEvent)
                assert isinstance(events[3].item, CommandExecutionItem)
                assert events[3].item.exit_code == 0

                # Check agent message
                assert isinstance(events[4], ItemCompletedEvent)
                assert isinstance(events[4].item, AgentMessageItem)

        anyio.run(_test)

    def test_query_passes_options(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(SIMPLE_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ) as mock_cls:
                options = CodexAgentOptions(model="o4-mini", cwd="/my/project")
                events = []
                async for event in query("test", options=options):
                    events.append(event)

                # Verify transport was constructed with the right options
                call_args = mock_cls.call_args
                assert call_args[0][0].model == "o4-mini"
                assert call_args[0][0].cwd == "/my/project"

        anyio.run(_test)

    def test_query_with_resume(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(SIMPLE_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ) as mock_cls:
                events = []
                async for event in query("follow up", resume_thread_id="session-001"):
                    events.append(event)

                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["resume_thread_id"] == "session-001"

        anyio.run(_test)

    def test_query_transport_closed_on_success(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(SIMPLE_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                async for _ in query("test"):
                    pass

                mock_transport.close.assert_awaited_once()

        anyio.run(_test)

    def test_query_transport_closed_on_error(self) -> None:
        async def _test() -> None:
            mock_transport = AsyncMock()

            async def mock_read() -> AsyncIterator[dict[str, Any]]:
                yield {"type": "thread.started", "thread_id": "x"}
                raise RuntimeError("boom")

            mock_transport.read_messages = mock_read
            mock_transport.connect = AsyncMock()
            mock_transport.close = AsyncMock()
            mock_transport.end_input = AsyncMock()
            mock_transport.write = AsyncMock()

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                try:
                    async for _ in query("test"):
                        pass
                except RuntimeError:
                    pass

                mock_transport.close.assert_awaited_once()

        anyio.run(_test)


class TestInternalClient:
    def test_process_query_yields_events(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(SIMPLE_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = InternalClient(CodexAgentOptions())
                events = []
                async for event in client.process_query("Hello"):
                    events.append(event)

                assert len(events) == 4

        anyio.run(_test)

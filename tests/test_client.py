"""Tests for CodexSDKClient — multi-turn, streaming, TurnResult aggregation."""

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest

from codex_agent_sdk._errors import CodexSDKError
from codex_agent_sdk.client import CodexSDKClient
from codex_agent_sdk.types import (
    AgentMessageItem,
    CodexAgentOptions,
    CommandExecutionItem,
    ItemCompletedEvent,
    SandboxMode,
    ThreadStartedEvent,
    TurnCompletedEvent,
)


def _make_mock_transport(messages: list[dict[str, Any]]) -> AsyncMock:
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


TURN_1_MESSAGES: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "tid-100"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {"type": "agent_message", "id": "msg_1", "text": "Created the file."},
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 300, "cached_input_tokens": 100, "output_tokens": 30},
    },
]

TURN_2_MESSAGES: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "tid-100"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {"type": "agent_message", "id": "msg_2", "text": "Added type hints."},
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 600, "cached_input_tokens": 400, "output_tokens": 40},
    },
]

FAILED_TURN_MESSAGES: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "tid-fail"},
    {"type": "turn.started"},
    {"type": "turn.failed", "error": {"message": "Rate limit exceeded"}},
]

TOOL_USE_TURN_MESSAGES: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "tid-tool"},
    {"type": "turn.started"},
    {
        "type": "item.started",
        "item": {
            "type": "command_execution",
            "id": "cmd_1",
            "command": "cat main.py",
            "aggregated_output": "",
            "status": "in_progress",
        },
    },
    {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "id": "cmd_1",
            "command": "cat main.py",
            "aggregated_output": "print('hello')\n",
            "exit_code": 0,
            "status": "completed",
        },
    },
    {
        "type": "item.completed",
        "item": {"type": "agent_message", "id": "msg_1", "text": "The file prints hello."},
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 800, "cached_input_tokens": 0, "output_tokens": 80},
    },
]


class TestCodexSDKClientRun:
    def test_basic_run(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                result = await client.run("Create a file")

                assert result.thread_id == "tid-100"
                assert result.final_response == "Created the file."
                assert result.usage is not None
                assert result.usage.input_tokens == 300
                assert result.usage.output_tokens == 30
                assert len(result.items) == 1
                assert isinstance(result.items[0], AgentMessageItem)
                assert len(result.events) == 4

        anyio.run(_test)

    def test_run_captures_thread_id(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                assert client.thread_id is None
                await client.run("test")
                assert client.thread_id == "tid-100"

        anyio.run(_test)

    def test_run_with_tool_use(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TOOL_USE_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                result = await client.run("Read main.py")

                # items only contains completed items
                assert len(result.items) == 2
                assert isinstance(result.items[0], CommandExecutionItem)
                assert result.items[0].exit_code == 0
                assert isinstance(result.items[1], AgentMessageItem)
                assert result.final_response == "The file prints hello."

        anyio.run(_test)

    def test_run_failed_turn_raises(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(FAILED_TURN_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                try:
                    await client.run("test")
                    assert False, "Should have raised"
                except CodexSDKError as e:
                    assert "Rate limit exceeded" in str(e)

        anyio.run(_test)


class TestCodexSDKClientMultiTurn:
    def test_multi_turn_passes_thread_id(self) -> None:
        async def _test() -> None:
            call_count = 0

            def make_transport(*args: Any, **kwargs: Any) -> AsyncMock:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _make_mock_transport(TURN_1_MESSAGES)
                else:
                    return _make_mock_transport(TURN_2_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                side_effect=make_transport,
            ) as mock_cls:
                client = CodexSDKClient()

                result1 = await client.run("Create file")
                assert result1.final_response == "Created the file."
                assert client.thread_id == "tid-100"

                result2 = await client.run("Add types")
                assert result2.final_response == "Added type hints."

                # Second call should have resume_thread_id
                second_call = mock_cls.call_args_list[1]
                assert second_call[1]["resume_thread_id"] == "tid-100"

        anyio.run(_test)

    def test_explicit_resume_thread_id(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ) as mock_cls:
                client = CodexSDKClient()
                await client.run("test", resume_thread_id="explicit-tid")

                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["resume_thread_id"] == "explicit-tid"

        anyio.run(_test)


class TestCodexSDKClientStreaming:
    def test_run_streamed_yields_events(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                events = []
                async for event in client.run_streamed("test"):
                    events.append(event)

                assert len(events) == 4
                assert isinstance(events[0], ThreadStartedEvent)
                assert isinstance(events[-1], TurnCompletedEvent)

        anyio.run(_test)

    def test_run_streamed_captures_thread_id(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                assert client.thread_id is None
                async for _ in client.run_streamed("test"):
                    pass
                assert client.thread_id == "tid-100"

        anyio.run(_test)


class TestCodexSDKClientContextManager:
    def test_async_context_manager(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                async with CodexSDKClient() as client:
                    result = await client.run("test")
                    assert result.final_response == "Created the file."

        anyio.run(_test)

    def test_options_passed_through(self) -> None:
        async def _test() -> None:
            mock_transport = _make_mock_transport(TURN_1_MESSAGES)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ) as mock_cls:
                options = CodexAgentOptions(
                    model="o4-mini",
                    sandbox=SandboxMode.WORKSPACE_WRITE,
                )
                async with CodexSDKClient(options) as client:
                    await client.run("test")

                call_args = mock_cls.call_args
                assert call_args[0][0].model == "o4-mini"
                assert call_args[0][0].sandbox == SandboxMode.WORKSPACE_WRITE

        anyio.run(_test)


class TestCodexSDKClientEdgeCases:
    def test_no_agent_message_in_turn(self) -> None:
        """Turn with only a command execution and no agent message."""

        async def _test() -> None:
            messages: list[dict[str, Any]] = [
                {"type": "thread.started", "thread_id": "tid"},
                {"type": "turn.started"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "id": "cmd_1",
                        "command": "echo done",
                        "aggregated_output": "done\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5},
                },
            ]
            mock_transport = _make_mock_transport(messages)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                result = await client.run("test")
                assert result.final_response is None
                assert len(result.items) == 1
                assert isinstance(result.items[0], CommandExecutionItem)

        anyio.run(_test)

    def test_multiple_agent_messages_uses_last(self) -> None:
        async def _test() -> None:
            messages: list[dict[str, Any]] = [
                {"type": "thread.started", "thread_id": "tid"},
                {"type": "turn.started"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "id": "m1", "text": "First"},
                },
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "id": "m2", "text": "Second"},
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5},
                },
            ]
            mock_transport = _make_mock_transport(messages)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                result = await client.run("test")
                # final_response should be the last agent message
                assert result.final_response == "Second"
                assert len(result.items) == 2

        anyio.run(_test)

    def test_empty_turn(self) -> None:
        async def _test() -> None:
            messages: list[dict[str, Any]] = [
                {"type": "thread.started", "thread_id": "tid"},
                {"type": "turn.started"},
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 5, "cached_input_tokens": 0, "output_tokens": 0},
                },
            ]
            mock_transport = _make_mock_transport(messages)

            with patch(
                "codex_agent_sdk._internal.query.SubprocessCLITransport",
                return_value=mock_transport,
            ):
                client = CodexSDKClient()
                result = await client.run("test")
                assert result.items == []
                assert result.final_response is None
                assert result.usage is not None
                assert result.usage.output_tokens == 0

        anyio.run(_test)

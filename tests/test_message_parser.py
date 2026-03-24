"""Tests for the JSONL message parser."""

from codex_agent_sdk._internal.message_parser import _parse_item, parse_event
from codex_agent_sdk.types import (
    AgentMessageItem,
    CommandExecutionItem,
    ErrorItem,
    FileChangeItem,
    FileChangeKind,
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemStatus,
    ItemUpdatedEvent,
    McpToolCallItem,
    ReasoningItem,
    StreamErrorEvent,
    ThreadStartedEvent,
    TodoListItem,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
    WebSearchItem,
)


# ---------------------------------------------------------------------------
# Item parsing
# ---------------------------------------------------------------------------


class TestParseItem:
    def test_agent_message(self) -> None:
        item = _parse_item({"type": "agent_message", "id": "msg_1", "text": "Hello world"})
        assert isinstance(item, AgentMessageItem)
        assert item.id == "msg_1"
        assert item.text == "Hello world"

    def test_agent_message_missing_text(self) -> None:
        item = _parse_item({"type": "agent_message", "id": "msg_1"})
        assert isinstance(item, AgentMessageItem)
        assert item.text == ""

    def test_reasoning(self) -> None:
        item = _parse_item({"type": "reasoning", "id": "r_1", "text": "Thinking about it..."})
        assert isinstance(item, ReasoningItem)
        assert item.text == "Thinking about it..."

    def test_command_execution_in_progress(self) -> None:
        item = _parse_item({
            "type": "command_execution",
            "id": "cmd_1",
            "command": "ls -la",
            "aggregated_output": "",
            "status": "in_progress",
        })
        assert isinstance(item, CommandExecutionItem)
        assert item.command == "ls -la"
        assert item.exit_code is None
        assert item.status == ItemStatus.IN_PROGRESS

    def test_command_execution_completed(self) -> None:
        item = _parse_item({
            "type": "command_execution",
            "id": "cmd_1",
            "command": "echo hello",
            "aggregated_output": "hello\n",
            "exit_code": 0,
            "status": "completed",
        })
        assert isinstance(item, CommandExecutionItem)
        assert item.exit_code == 0
        assert item.aggregated_output == "hello\n"
        assert item.status == ItemStatus.COMPLETED

    def test_command_execution_declined(self) -> None:
        item = _parse_item({
            "type": "command_execution",
            "id": "cmd_1",
            "command": "rm -rf /",
            "aggregated_output": "",
            "status": "declined",
        })
        assert isinstance(item, CommandExecutionItem)
        assert item.status == ItemStatus.DECLINED

    def test_file_change(self) -> None:
        item = _parse_item({
            "type": "file_change",
            "id": "fc_1",
            "changes": [
                {"path": "src/main.py", "kind": "add"},
                {"path": "old.py", "kind": "delete"},
                {"path": "README.md", "kind": "update"},
            ],
            "status": "completed",
        })
        assert isinstance(item, FileChangeItem)
        assert len(item.changes) == 3
        assert item.changes[0].path == "src/main.py"
        assert item.changes[0].kind == FileChangeKind.ADD
        assert item.changes[1].kind == FileChangeKind.DELETE
        assert item.changes[2].kind == FileChangeKind.UPDATE

    def test_file_change_empty_changes(self) -> None:
        item = _parse_item({"type": "file_change", "id": "fc_1", "status": "completed"})
        assert isinstance(item, FileChangeItem)
        assert item.changes == []

    def test_mcp_tool_call(self) -> None:
        item = _parse_item({
            "type": "mcp_tool_call",
            "id": "mcp_1",
            "server": "my-server",
            "tool": "read_file",
            "arguments": '{"path": "/tmp/x"}',
            "result": "file contents",
            "status": "completed",
        })
        assert isinstance(item, McpToolCallItem)
        assert item.server == "my-server"
        assert item.tool == "read_file"
        assert item.result == "file contents"
        assert item.error is None

    def test_mcp_tool_call_with_error(self) -> None:
        item = _parse_item({
            "type": "mcp_tool_call",
            "id": "mcp_1",
            "server": "s",
            "tool": "t",
            "arguments": "{}",
            "error": "permission denied",
            "status": "failed",
        })
        assert isinstance(item, McpToolCallItem)
        assert item.error == "permission denied"
        assert item.result is None
        assert item.status == ItemStatus.FAILED

    def test_web_search(self) -> None:
        item = _parse_item({"type": "web_search", "id": "ws_1", "query": "python asyncio"})
        assert isinstance(item, WebSearchItem)
        assert item.query == "python asyncio"

    def test_todo_list(self) -> None:
        item = _parse_item({
            "type": "todo_list",
            "id": "todo_1",
            "items": [
                {"text": "Step 1", "completed": True},
                {"text": "Step 2", "completed": False},
            ],
        })
        assert isinstance(item, TodoListItem)
        assert len(item.items) == 2
        assert item.items[0].text == "Step 1"
        assert item.items[0].completed is True
        assert item.items[1].completed is False

    def test_todo_list_empty(self) -> None:
        item = _parse_item({"type": "todo_list", "id": "todo_1"})
        assert isinstance(item, TodoListItem)
        assert item.items == []

    def test_error_item(self) -> None:
        item = _parse_item({"type": "error", "id": "err_1", "message": "something broke"})
        assert isinstance(item, ErrorItem)
        assert item.message == "something broke"

    def test_unknown_item_type(self) -> None:
        item = _parse_item({"type": "future_type", "id": "x_1"})
        assert isinstance(item, ErrorItem)
        assert "Unknown item type" in item.message

    def test_missing_type(self) -> None:
        item = _parse_item({"id": "no_type"})
        assert isinstance(item, ErrorItem)

    def test_unknown_status_falls_back(self) -> None:
        item = _parse_item({
            "type": "command_execution",
            "id": "cmd_1",
            "command": "x",
            "status": "some_future_status",
        })
        assert isinstance(item, CommandExecutionItem)
        assert item.status == ItemStatus.IN_PROGRESS

    def test_unknown_file_change_kind_falls_back(self) -> None:
        item = _parse_item({
            "type": "file_change",
            "id": "fc_1",
            "changes": [{"path": "x.py", "kind": "rename"}],
            "status": "completed",
        })
        assert isinstance(item, FileChangeItem)
        assert item.changes[0].kind == FileChangeKind.UPDATE


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


class TestParseEvent:
    def test_thread_started(self) -> None:
        event = parse_event({"type": "thread.started", "thread_id": "tid-abc"})
        assert isinstance(event, ThreadStartedEvent)
        assert event.thread_id == "tid-abc"

    def test_thread_started_missing_id(self) -> None:
        event = parse_event({"type": "thread.started"})
        assert isinstance(event, ThreadStartedEvent)
        assert event.thread_id == ""

    def test_turn_started(self) -> None:
        event = parse_event({"type": "turn.started"})
        assert isinstance(event, TurnStartedEvent)

    def test_turn_completed(self) -> None:
        event = parse_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 200,
            },
        })
        assert isinstance(event, TurnCompletedEvent)
        assert event.usage.input_tokens == 1000
        assert event.usage.cached_input_tokens == 500
        assert event.usage.output_tokens == 200

    def test_turn_completed_missing_usage(self) -> None:
        event = parse_event({"type": "turn.completed"})
        assert isinstance(event, TurnCompletedEvent)
        assert event.usage.input_tokens == 0

    def test_turn_failed(self) -> None:
        event = parse_event({
            "type": "turn.failed",
            "error": {"message": "Rate limit exceeded"},
        })
        assert isinstance(event, TurnFailedEvent)
        assert event.error.message == "Rate limit exceeded"

    def test_turn_failed_missing_error(self) -> None:
        event = parse_event({"type": "turn.failed"})
        assert isinstance(event, TurnFailedEvent)
        assert event.error.message == ""

    def test_item_started(self) -> None:
        event = parse_event({
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "id": "cmd_1",
                "command": "pwd",
                "aggregated_output": "",
                "status": "in_progress",
            },
        })
        assert isinstance(event, ItemStartedEvent)
        assert isinstance(event.item, CommandExecutionItem)
        assert event.item.command == "pwd"

    def test_item_updated(self) -> None:
        event = parse_event({
            "type": "item.updated",
            "item": {
                "type": "command_execution",
                "id": "cmd_1",
                "command": "long-running",
                "aggregated_output": "partial output...",
                "status": "in_progress",
            },
        })
        assert isinstance(event, ItemUpdatedEvent)
        assert isinstance(event.item, CommandExecutionItem)
        assert event.item.aggregated_output == "partial output..."

    def test_item_completed(self) -> None:
        event = parse_event({
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "id": "msg_1",
                "text": "All done!",
            },
        })
        assert isinstance(event, ItemCompletedEvent)
        assert isinstance(event.item, AgentMessageItem)
        assert event.item.text == "All done!"

    def test_stream_error(self) -> None:
        event = parse_event({"type": "error", "message": "connection lost"})
        assert isinstance(event, StreamErrorEvent)
        assert event.message == "connection lost"

    def test_unknown_event_type(self) -> None:
        event = parse_event({"type": "future.event.type"})
        assert isinstance(event, StreamErrorEvent)
        assert "Unknown event type" in event.message

    def test_empty_dict(self) -> None:
        event = parse_event({})
        assert isinstance(event, StreamErrorEvent)

    def test_item_completed_with_nested_item(self) -> None:
        """Full round-trip: event wrapping a file_change item."""
        event = parse_event({
            "type": "item.completed",
            "item": {
                "type": "file_change",
                "id": "fc_1",
                "changes": [{"path": "new.py", "kind": "add"}],
                "status": "completed",
            },
        })
        assert isinstance(event, ItemCompletedEvent)
        assert isinstance(event.item, FileChangeItem)
        assert event.item.changes[0].path == "new.py"
        assert event.item.changes[0].kind == FileChangeKind.ADD

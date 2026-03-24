"""Parse raw JSONL dicts from the Codex CLI into typed event objects."""

from __future__ import annotations

from typing import Any

from codex_agent_sdk.types import (
    AgentMessageItem,
    CommandExecutionItem,
    ErrorItem,
    FileChange,
    FileChangeItem,
    FileChangeKind,
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemStatus,
    ItemUpdatedEvent,
    McpToolCallItem,
    ReasoningItem,
    StreamErrorEvent,
    ThreadEvent,
    ThreadItem,
    ThreadStartedEvent,
    TodoItem,
    TodoListItem,
    TurnCompletedEvent,
    TurnError,
    TurnFailedEvent,
    TurnStartedEvent,
    Usage,
    WebSearchItem,
)


def _parse_status(raw: Any) -> ItemStatus:
    if isinstance(raw, str):
        try:
            return ItemStatus(raw)
        except ValueError:
            return ItemStatus.IN_PROGRESS
    return ItemStatus.IN_PROGRESS


def _parse_file_change_kind(raw: Any) -> FileChangeKind:
    if isinstance(raw, str):
        try:
            return FileChangeKind(raw)
        except ValueError:
            return FileChangeKind.UPDATE
    return FileChangeKind.UPDATE


def _parse_item(raw: dict[str, Any]) -> ThreadItem:
    """Convert a raw item dict into a typed :class:`ThreadItem`."""
    item_type = raw.get("type", "")
    item_id = raw.get("id", "")

    match item_type:
        case "agent_message":
            return AgentMessageItem(
                type=item_type,
                id=item_id,
                text=raw.get("text", ""),
            )

        case "reasoning":
            return ReasoningItem(
                type=item_type,
                id=item_id,
                text=raw.get("text", ""),
            )

        case "command_execution":
            return CommandExecutionItem(
                type=item_type,
                id=item_id,
                command=raw.get("command", ""),
                aggregated_output=raw.get("aggregated_output", ""),
                exit_code=raw.get("exit_code"),
                status=_parse_status(raw.get("status")),
            )

        case "file_change":
            changes: list[FileChange] = []
            for c in raw.get("changes", []):
                changes.append(
                    FileChange(
                        path=c.get("path", ""),
                        kind=_parse_file_change_kind(c.get("kind")),
                    )
                )
            return FileChangeItem(
                type=item_type,
                id=item_id,
                changes=changes,
                status=_parse_status(raw.get("status")),
            )

        case "mcp_tool_call":
            return McpToolCallItem(
                type=item_type,
                id=item_id,
                server=raw.get("server", ""),
                tool=raw.get("tool", ""),
                arguments=raw.get("arguments", ""),
                result=raw.get("result"),
                error=raw.get("error"),
                status=_parse_status(raw.get("status")),
            )

        case "web_search":
            return WebSearchItem(
                type=item_type,
                id=item_id,
                query=raw.get("query", ""),
            )

        case "todo_list":
            items: list[TodoItem] = []
            for t in raw.get("items", []):
                items.append(
                    TodoItem(
                        text=t.get("text", ""),
                        completed=t.get("completed", False),
                    )
                )
            return TodoListItem(
                type=item_type,
                id=item_id,
                items=items,
            )

        case "error":
            return ErrorItem(
                type=item_type,
                id=item_id,
                message=raw.get("message", ""),
            )

        case _:
            # Unknown item type — preserve as an ErrorItem so consumers
            # can still inspect it.
            return ErrorItem(
                type=item_type,
                id=item_id,
                message=f"Unknown item type: {item_type}",
            )


def parse_event(raw: dict[str, Any]) -> ThreadEvent:
    """Convert a raw JSONL dict into a typed :class:`ThreadEvent`."""
    event_type = raw.get("type", "")

    match event_type:
        case "thread.started":
            return ThreadStartedEvent(
                type=event_type,
                thread_id=raw.get("thread_id", ""),
            )

        case "turn.started":
            return TurnStartedEvent(type=event_type)

        case "turn.completed":
            usage_raw = raw.get("usage", {})
            return TurnCompletedEvent(
                type=event_type,
                usage=Usage(
                    input_tokens=usage_raw.get("input_tokens", 0),
                    cached_input_tokens=usage_raw.get("cached_input_tokens", 0),
                    output_tokens=usage_raw.get("output_tokens", 0),
                ),
            )

        case "turn.failed":
            err_raw = raw.get("error", {})
            return TurnFailedEvent(
                type=event_type,
                error=TurnError(message=err_raw.get("message", "")),
            )

        case "item.started":
            return ItemStartedEvent(
                type=event_type,
                item=_parse_item(raw.get("item", {})),
            )

        case "item.updated":
            return ItemUpdatedEvent(
                type=event_type,
                item=_parse_item(raw.get("item", {})),
            )

        case "item.completed":
            return ItemCompletedEvent(
                type=event_type,
                item=_parse_item(raw.get("item", {})),
            )

        case "error":
            return StreamErrorEvent(
                type=event_type,
                message=raw.get("message", ""),
            )

        case _:
            return StreamErrorEvent(
                type=event_type,
                message=f"Unknown event type: {event_type}",
            )

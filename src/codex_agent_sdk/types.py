"""Type definitions for the Codex Agent SDK.

Maps to the JSONL event protocol emitted by `codex exec --experimental-json`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Union

if sys.version_info >= (3, 11):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SandboxMode(str, Enum):
    """Sandbox policy for the Codex CLI."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    FULL_ACCESS = "danger-full-access"


class ApprovalPolicy(str, Enum):
    """Approval policy for tool use."""

    NEVER = "never"
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    UNTRUSTED = "untrusted"


class ReasoningEffort(str, Enum):
    """Model reasoning effort level."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class WebSearch(str, Enum):
    """Web search mode."""

    DISABLED = "disabled"
    CACHED = "cached"
    LIVE = "live"


class ItemStatus(str, Enum):
    """Status of a thread item."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DECLINED = "declined"


class FileChangeKind(str, Enum):
    """Kind of file change."""

    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class CodexAgentOptions:
    """Configuration options for a Codex agent session.

    These map to CLI flags passed to ``codex exec``.
    """

    # Model configuration
    model: Optional[str] = None
    reasoning_effort: Optional[ReasoningEffort] = None

    # Sandbox / approval
    sandbox: Optional[SandboxMode] = None
    approval_policy: Optional[ApprovalPolicy] = None
    full_auto: bool = False

    # Working directory
    cwd: Optional[str] = None
    additional_writable_dirs: Sequence[str] = field(default_factory=list)

    # Session
    ephemeral: bool = False

    # Images
    images: Sequence[str] = field(default_factory=list)

    # Structured output
    output_schema: Optional[str] = None
    output_schema_file: Optional[str] = None

    # Web search
    web_search: Optional[WebSearch] = None

    # Network
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    # Misc
    profile: Optional[str] = None
    skip_git_repo_check: bool = False
    config_overrides: dict[str, str] = field(default_factory=dict)

    # CLI binary
    cli_path: Optional[str] = None

    # Environment variables forwarded to the CLI process
    env: Optional[dict[str, str]] = None


# ---------------------------------------------------------------------------
# Thread items
# ---------------------------------------------------------------------------


@dataclass
class AgentMessageItem:
    """Agent's natural-language (or structured) response."""

    type: str  # "agent_message"
    id: str
    text: str = ""


@dataclass
class ReasoningItem:
    """Agent's reasoning summary."""

    type: str  # "reasoning"
    id: str
    text: str = ""


@dataclass
class FileChange:
    """A single file change within a FileChangeItem."""

    path: str
    kind: FileChangeKind


@dataclass
class FileChangeItem:
    """One or more file modifications."""

    type: str  # "file_change"
    id: str
    changes: list[FileChange] = field(default_factory=list)
    status: ItemStatus = ItemStatus.IN_PROGRESS


@dataclass
class CommandExecutionItem:
    """Shell command execution."""

    type: str  # "command_execution"
    id: str
    command: str = ""
    aggregated_output: str = ""
    exit_code: Optional[int] = None
    status: ItemStatus = ItemStatus.IN_PROGRESS


@dataclass
class McpToolCallItem:
    """MCP tool invocation."""

    type: str  # "mcp_tool_call"
    id: str
    server: str = ""
    tool: str = ""
    arguments: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    status: ItemStatus = ItemStatus.IN_PROGRESS


@dataclass
class WebSearchItem:
    """Web search request."""

    type: str  # "web_search"
    id: str
    query: str = ""


@dataclass
class TodoItem:
    """A single entry in a TodoListItem."""

    text: str
    completed: bool = False


@dataclass
class TodoListItem:
    """Agent's running to-do list."""

    type: str  # "todo_list"
    id: str
    items: list[TodoItem] = field(default_factory=list)


@dataclass
class ErrorItem:
    """Non-fatal error surfaced as a thread item."""

    type: str  # "error"
    id: str
    message: str = ""


ThreadItem: TypeAlias = Union[
    AgentMessageItem,
    ReasoningItem,
    FileChangeItem,
    CommandExecutionItem,
    McpToolCallItem,
    WebSearchItem,
    TodoListItem,
    ErrorItem,
]


# ---------------------------------------------------------------------------
# Thread events (top-level JSONL messages)
# ---------------------------------------------------------------------------


@dataclass
class ThreadStartedEvent:
    """Emitted once at the beginning; carries the thread ID for resumption."""

    type: str  # "thread.started"
    thread_id: str


@dataclass
class TurnStartedEvent:
    """Agent begins processing the prompt."""

    type: str  # "turn.started"


@dataclass
class Usage:
    """Token usage for a completed turn."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TurnCompletedEvent:
    """Turn finished successfully."""

    type: str  # "turn.completed"
    usage: Usage = field(default_factory=Usage)


@dataclass
class TurnError:
    """Error detail inside a TurnFailedEvent."""

    message: str


@dataclass
class TurnFailedEvent:
    """Turn failed."""

    type: str  # "turn.failed"
    error: TurnError = field(default_factory=lambda: TurnError(message=""))


@dataclass
class ItemStartedEvent:
    """A new item begins (typically ``in_progress``)."""

    type: str  # "item.started"
    item: ThreadItem


@dataclass
class ItemUpdatedEvent:
    """An existing item's state was updated."""

    type: str  # "item.updated"
    item: ThreadItem


@dataclass
class ItemCompletedEvent:
    """An item reached its terminal state."""

    type: str  # "item.completed"
    item: ThreadItem


@dataclass
class StreamErrorEvent:
    """Unrecoverable stream-level error."""

    type: str  # "error"
    message: str


ThreadEvent: TypeAlias = Union[
    ThreadStartedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    ItemStartedEvent,
    ItemUpdatedEvent,
    ItemCompletedEvent,
    StreamErrorEvent,
]


# ---------------------------------------------------------------------------
# Result container (returned by non-streaming ``run()``)
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Aggregated result of a single turn."""

    thread_id: Optional[str] = None
    items: list[ThreadItem] = field(default_factory=list)
    final_response: Optional[str] = None
    usage: Optional[Usage] = None
    events: list[ThreadEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------


@dataclass
class ImageInput:
    """An image to attach to the prompt."""

    path: str


@dataclass
class TextInput:
    """Text portion of a user prompt."""

    text: str


UserInput: TypeAlias = Union[str, Sequence[Union[TextInput, ImageInput]]]


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "SandboxMode",
    "ApprovalPolicy",
    "ReasoningEffort",
    "WebSearch",
    "ItemStatus",
    "FileChangeKind",
    # Options
    "CodexAgentOptions",
    # Items
    "AgentMessageItem",
    "ReasoningItem",
    "FileChange",
    "FileChangeItem",
    "CommandExecutionItem",
    "McpToolCallItem",
    "WebSearchItem",
    "TodoItem",
    "TodoListItem",
    "ErrorItem",
    "ThreadItem",
    # Events
    "ThreadStartedEvent",
    "TurnStartedEvent",
    "Usage",
    "TurnCompletedEvent",
    "TurnError",
    "TurnFailedEvent",
    "ItemStartedEvent",
    "ItemUpdatedEvent",
    "ItemCompletedEvent",
    "StreamErrorEvent",
    "ThreadEvent",
    # Result
    "TurnResult",
    # Input
    "ImageInput",
    "TextInput",
    "UserInput",
]

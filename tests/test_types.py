"""Tests for type construction and defaults."""

from codex_agent_sdk.types import (
    AgentMessageItem,
    ApprovalPolicy,
    CodexAgentOptions,
    CommandExecutionItem,
    ErrorItem,
    FileChange,
    FileChangeItem,
    FileChangeKind,
    ImageInput,
    ItemStatus,
    McpToolCallItem,
    ReasoningEffort,
    ReasoningItem,
    SandboxMode,
    TextInput,
    ThreadStartedEvent,
    TodoItem,
    TodoListItem,
    TurnCompletedEvent,
    TurnError,
    TurnFailedEvent,
    TurnResult,
    TurnStartedEvent,
    Usage,
    WebSearch,
    WebSearchItem,
)


class TestEnums:
    def test_sandbox_mode_values(self) -> None:
        assert SandboxMode.READ_ONLY.value == "read-only"
        assert SandboxMode.WORKSPACE_WRITE.value == "workspace-write"
        assert SandboxMode.FULL_ACCESS.value == "danger-full-access"

    def test_approval_policy_values(self) -> None:
        assert ApprovalPolicy.NEVER.value == "never"
        assert ApprovalPolicy.ON_REQUEST.value == "on-request"
        assert ApprovalPolicy.ON_FAILURE.value == "on-failure"
        assert ApprovalPolicy.UNTRUSTED.value == "untrusted"

    def test_reasoning_effort_values(self) -> None:
        assert ReasoningEffort.MINIMAL.value == "minimal"
        assert ReasoningEffort.LOW.value == "low"
        assert ReasoningEffort.MEDIUM.value == "medium"
        assert ReasoningEffort.HIGH.value == "high"
        assert ReasoningEffort.XHIGH.value == "xhigh"

    def test_web_search_values(self) -> None:
        assert WebSearch.DISABLED.value == "disabled"
        assert WebSearch.CACHED.value == "cached"
        assert WebSearch.LIVE.value == "live"

    def test_item_status_values(self) -> None:
        assert ItemStatus.IN_PROGRESS.value == "in_progress"
        assert ItemStatus.COMPLETED.value == "completed"
        assert ItemStatus.FAILED.value == "failed"
        assert ItemStatus.DECLINED.value == "declined"

    def test_file_change_kind_values(self) -> None:
        assert FileChangeKind.ADD.value == "add"
        assert FileChangeKind.DELETE.value == "delete"
        assert FileChangeKind.UPDATE.value == "update"


class TestCodexAgentOptions:
    def test_defaults(self) -> None:
        opts = CodexAgentOptions()
        assert opts.model is None
        assert opts.sandbox is None
        assert opts.approval_policy is None
        assert opts.full_auto is False
        assert opts.cwd is None
        assert opts.additional_writable_dirs == []
        assert opts.ephemeral is False
        assert opts.images == []
        assert opts.output_schema is None
        assert opts.output_schema_file is None
        assert opts.web_search is None
        assert opts.base_url is None
        assert opts.api_key is None
        assert opts.profile is None
        assert opts.skip_git_repo_check is False
        assert opts.config_overrides == {}
        assert opts.cli_path is None
        assert opts.env is None
        assert opts.reasoning_effort is None

    def test_with_values(self) -> None:
        opts = CodexAgentOptions(
            model="o4-mini",
            sandbox=SandboxMode.WORKSPACE_WRITE,
            full_auto=True,
            cwd="/tmp/test",
            api_key="sk-test",
        )
        assert opts.model == "o4-mini"
        assert opts.sandbox == SandboxMode.WORKSPACE_WRITE
        assert opts.full_auto is True
        assert opts.cwd == "/tmp/test"
        assert opts.api_key == "sk-test"

    def test_mutable_defaults_are_independent(self) -> None:
        opts1 = CodexAgentOptions()
        opts2 = CodexAgentOptions()
        opts1.images.append("img.png")  # type: ignore[union-attr]
        assert len(opts2.images) == 0


class TestThreadItems:
    def test_agent_message(self) -> None:
        item = AgentMessageItem(type="agent_message", id="msg_1", text="Hello")
        assert item.type == "agent_message"
        assert item.id == "msg_1"
        assert item.text == "Hello"

    def test_agent_message_defaults(self) -> None:
        item = AgentMessageItem(type="agent_message", id="msg_1")
        assert item.text == ""

    def test_reasoning_item(self) -> None:
        item = ReasoningItem(type="reasoning", id="r_1", text="Thinking...")
        assert item.text == "Thinking..."

    def test_command_execution_item(self) -> None:
        item = CommandExecutionItem(
            type="command_execution",
            id="cmd_1",
            command="ls -la",
            aggregated_output="total 42\n",
            exit_code=0,
            status=ItemStatus.COMPLETED,
        )
        assert item.command == "ls -la"
        assert item.exit_code == 0
        assert item.status == ItemStatus.COMPLETED

    def test_command_execution_defaults(self) -> None:
        item = CommandExecutionItem(type="command_execution", id="cmd_1")
        assert item.command == ""
        assert item.aggregated_output == ""
        assert item.exit_code is None
        assert item.status == ItemStatus.IN_PROGRESS

    def test_file_change_item(self) -> None:
        changes = [
            FileChange(path="src/main.py", kind=FileChangeKind.ADD),
            FileChange(path="old.py", kind=FileChangeKind.DELETE),
        ]
        item = FileChangeItem(
            type="file_change",
            id="fc_1",
            changes=changes,
            status=ItemStatus.COMPLETED,
        )
        assert len(item.changes) == 2
        assert item.changes[0].path == "src/main.py"
        assert item.changes[0].kind == FileChangeKind.ADD

    def test_mcp_tool_call_item(self) -> None:
        item = McpToolCallItem(
            type="mcp_tool_call",
            id="mcp_1",
            server="my-server",
            tool="read_file",
            arguments='{"path": "/tmp/test"}',
            result="file contents",
            status=ItemStatus.COMPLETED,
        )
        assert item.server == "my-server"
        assert item.tool == "read_file"
        assert item.result == "file contents"

    def test_web_search_item(self) -> None:
        item = WebSearchItem(type="web_search", id="ws_1", query="python async")
        assert item.query == "python async"

    def test_todo_list_item(self) -> None:
        items = [
            TodoItem(text="Step 1", completed=True),
            TodoItem(text="Step 2", completed=False),
        ]
        item = TodoListItem(type="todo_list", id="todo_1", items=items)
        assert len(item.items) == 2
        assert item.items[0].completed is True
        assert item.items[1].completed is False

    def test_error_item(self) -> None:
        item = ErrorItem(type="error", id="err_1", message="Something went wrong")
        assert item.message == "Something went wrong"


class TestThreadEvents:
    def test_thread_started(self) -> None:
        event = ThreadStartedEvent(type="thread.started", thread_id="abc-123")
        assert event.thread_id == "abc-123"

    def test_turn_started(self) -> None:
        event = TurnStartedEvent(type="turn.started")
        assert event.type == "turn.started"

    def test_turn_completed(self) -> None:
        usage = Usage(input_tokens=100, cached_input_tokens=50, output_tokens=25)
        event = TurnCompletedEvent(type="turn.completed", usage=usage)
        assert event.usage.input_tokens == 100
        assert event.usage.cached_input_tokens == 50
        assert event.usage.output_tokens == 25

    def test_usage_defaults(self) -> None:
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.cached_input_tokens == 0
        assert usage.output_tokens == 0

    def test_turn_failed(self) -> None:
        event = TurnFailedEvent(
            type="turn.failed",
            error=TurnError(message="API error"),
        )
        assert event.error.message == "API error"


class TestTurnResult:
    def test_defaults(self) -> None:
        result = TurnResult()
        assert result.thread_id is None
        assert result.items == []
        assert result.final_response is None
        assert result.usage is None
        assert result.events == []

    def test_with_data(self) -> None:
        item = AgentMessageItem(type="agent_message", id="msg_1", text="Done")
        result = TurnResult(
            thread_id="tid",
            items=[item],
            final_response="Done",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        assert result.thread_id == "tid"
        assert len(result.items) == 1
        assert result.final_response == "Done"


class TestUserInput:
    def test_text_input(self) -> None:
        ti = TextInput(text="Hello")
        assert ti.text == "Hello"

    def test_image_input(self) -> None:
        ii = ImageInput(path="/tmp/img.png")
        assert ii.path == "/tmp/img.png"

"""E2e tests for tool use — verify the agent can execute commands."""

import anyio

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    CommandExecutionItem,
    ItemCompletedEvent,
    SandboxMode,
    ThreadEvent,
    query,
)


class TestToolUse:
    def test_command_execution(self) -> None:
        """Verify the agent can run a shell command and return the output."""

        async def _test() -> None:
            events: list[ThreadEvent] = []
            async for event in query(
                "Run `echo SDK_TOOL_TEST` and tell me the output",
                options=CodexAgentOptions(
                    sandbox=SandboxMode.READ_ONLY,
                    full_auto=True,
                    ephemeral=True,
                ),
            ):
                events.append(event)

            # Should have at least one command execution
            cmd_items = [
                e
                for e in events
                if isinstance(e, ItemCompletedEvent)
                and isinstance(e.item, CommandExecutionItem)
            ]
            assert len(cmd_items) >= 1

            # At least one command should have completed successfully
            completed = [e for e in cmd_items if e.item.exit_code == 0]
            assert len(completed) >= 1

            # Should have an agent message
            messages = [
                e
                for e in events
                if isinstance(e, ItemCompletedEvent)
                and isinstance(e.item, AgentMessageItem)
            ]
            assert len(messages) >= 1

        anyio.run(_test)

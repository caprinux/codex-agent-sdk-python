"""Basic e2e tests — verify the SDK can talk to the real Codex API."""

import anyio

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    ItemCompletedEvent,
    SandboxMode,
    ThreadEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
    query,
)


class TestBasicQuery:
    def test_simple_prompt_returns_events(self) -> None:
        """Send a simple prompt and verify we get the expected event sequence."""

        async def _test() -> None:
            events: list[ThreadEvent] = []
            async for event in query(
                "Reply with exactly: HELLO_SDK_TEST",
                options=CodexAgentOptions(
                    sandbox=SandboxMode.READ_ONLY,
                    full_auto=True,
                    ephemeral=True,
                ),
            ):
                events.append(event)

            # Must have at least thread.started, turn.started, an item, turn.completed
            assert len(events) >= 3

            # First event should be thread.started with a non-empty ID
            assert isinstance(events[0], ThreadStartedEvent)
            assert events[0].thread_id != ""

            # Should contain a turn.started
            assert any(isinstance(e, TurnStartedEvent) for e in events)

            # Should contain a turn.completed with usage
            completed = [e for e in events if isinstance(e, TurnCompletedEvent)]
            assert len(completed) == 1
            assert completed[0].usage.input_tokens > 0
            assert completed[0].usage.output_tokens > 0

            # Should have an agent message containing our test string
            messages = [
                e
                for e in events
                if isinstance(e, ItemCompletedEvent)
                and isinstance(e.item, AgentMessageItem)
            ]
            assert len(messages) >= 1
            assert "HELLO_SDK_TEST" in messages[-1].item.text

        anyio.run(_test)

    def test_ephemeral_query(self) -> None:
        """Verify ephemeral sessions work (no persistent state)."""

        async def _test() -> None:
            events: list[ThreadEvent] = []
            async for event in query(
                "Reply with exactly: EPHEMERAL_TEST",
                options=CodexAgentOptions(
                    full_auto=True,
                    ephemeral=True,
                ),
            ):
                events.append(event)

            messages = [
                e
                for e in events
                if isinstance(e, ItemCompletedEvent)
                and isinstance(e.item, AgentMessageItem)
            ]
            assert len(messages) >= 1
            assert "EPHEMERAL_TEST" in messages[-1].item.text

        anyio.run(_test)

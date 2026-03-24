"""E2e tests for CodexSDKClient — multi-turn and streaming."""

import anyio

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    CodexSDKClient,
    ItemCompletedEvent,
    SandboxMode,
    ThreadStartedEvent,
    TurnCompletedEvent,
)


class TestClientRun:
    def test_run_returns_turn_result(self) -> None:
        """Verify client.run() returns a populated TurnResult."""

        async def _test() -> None:
            client = CodexSDKClient(
                CodexAgentOptions(
                    sandbox=SandboxMode.READ_ONLY,
                    full_auto=True,
                    ephemeral=True,
                )
            )

            result = await client.run("Reply with exactly: CLIENT_TEST")

            assert result.thread_id is not None
            assert result.thread_id != ""
            assert result.usage is not None
            assert result.usage.input_tokens > 0
            assert result.final_response is not None
            assert "CLIENT_TEST" in result.final_response
            assert client.thread_id == result.thread_id

        anyio.run(_test)


class TestClientStreaming:
    def test_run_streamed_yields_events(self) -> None:
        """Verify client.run_streamed() yields real events."""

        async def _test() -> None:
            client = CodexSDKClient(
                CodexAgentOptions(
                    sandbox=SandboxMode.READ_ONLY,
                    full_auto=True,
                    ephemeral=True,
                )
            )

            saw_thread_started = False
            saw_turn_completed = False
            saw_message = False

            async for event in client.run_streamed("Reply with exactly: STREAM_TEST"):
                if isinstance(event, ThreadStartedEvent):
                    saw_thread_started = True
                elif isinstance(event, TurnCompletedEvent):
                    saw_turn_completed = True
                elif (
                    isinstance(event, ItemCompletedEvent)
                    and isinstance(event.item, AgentMessageItem)
                ):
                    if "STREAM_TEST" in event.item.text:
                        saw_message = True

            assert saw_thread_started
            assert saw_turn_completed
            assert saw_message

        anyio.run(_test)


class TestClientMultiTurn:
    def test_multi_turn_resumes_thread(self) -> None:
        """Verify the client can execute a second turn via resume."""

        async def _test() -> None:
            client = CodexSDKClient(
                CodexAgentOptions(
                    sandbox=SandboxMode.READ_ONLY,
                    full_auto=True,
                    ephemeral=True,
                )
            )

            result1 = await client.run("Remember the word BANANA")
            assert client.thread_id is not None
            assert result1.final_response is not None

            # Second turn should succeed (resumes the session)
            result2 = await client.run("What word did I ask you to remember?")
            assert result2.final_response is not None
            assert result2.thread_id is not None

        anyio.run(_test)

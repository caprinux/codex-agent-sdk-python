"""Streaming example — process events as they arrive.

Shows how to use ``run_streamed()`` to react to events in real time,
including command executions, file changes, and reasoning.

Usage:
    export CODEX_API_KEY="sk-..."
    python examples/streaming.py
"""

import asyncio

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    CodexSDKClient,
    CommandExecutionItem,
    FileChangeItem,
    ItemCompletedEvent,
    ItemStartedEvent,
    ReasoningItem,
    SandboxMode,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)


async def main() -> None:
    options = CodexAgentOptions(
        sandbox=SandboxMode.READ_ONLY,
    )

    client = CodexSDKClient(options)

    async for event in client.run_streamed(
        "Explain the structure of this project directory"
    ):
        match event:
            case ThreadStartedEvent(thread_id=tid):
                print(f"--- Thread started: {tid} ---")

            case TurnStartedEvent():
                print("--- Turn started ---")

            case ItemStartedEvent(item=item):
                if isinstance(item, CommandExecutionItem):
                    print(f"  > Running: {item.command}")

            case ItemCompletedEvent(item=item):
                if isinstance(item, CommandExecutionItem):
                    print(f"  > Exit code: {item.exit_code}")
                    if item.aggregated_output:
                        for line in item.aggregated_output.splitlines()[:10]:
                            print(f"    {line}")
                elif isinstance(item, AgentMessageItem):
                    print(f"\n{item.text}")
                elif isinstance(item, ReasoningItem):
                    print(f"  [reasoning] {item.text[:100]}...")
                elif isinstance(item, FileChangeItem):
                    for change in item.changes:
                        print(f"  [{change.kind.value}] {change.path}")

            case TurnCompletedEvent(usage=usage):
                print(f"\n--- Turn completed (tokens: {usage.input_tokens} in, {usage.output_tokens} out) ---")


if __name__ == "__main__":
    asyncio.run(main())

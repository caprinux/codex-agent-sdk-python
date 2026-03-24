"""Simple proof-of-concept: send "hello" to Codex and print the response."""

import asyncio

from codex_agent_sdk import (
    AgentMessageItem,
    CodexAgentOptions,
    CodexSDKClient,
    CommandExecutionItem,
    ItemCompletedEvent,
    SandboxMode,
    TurnCompletedEvent,
)


async def main() -> None:
    client = CodexSDKClient(
        CodexAgentOptions(
            model="o4-mini",
            sandbox=SandboxMode.READ_ONLY,
            full_auto=True,
        )
    )

    print("Sending 'hello' to Codex...\n")

    async for event in client.run_streamed("Say hello and tell me what directory I'm in"):
        if isinstance(event, ItemCompletedEvent):
            if isinstance(event.item, CommandExecutionItem):
                print(f"$ {event.item.command}")
                print(event.item.aggregated_output)
            elif isinstance(event.item, AgentMessageItem):
                print(event.item.text)
        elif isinstance(event, TurnCompletedEvent):
            print(f"\n[tokens: {event.usage.input_tokens} in, {event.usage.output_tokens} out]")


if __name__ == "__main__":
    asyncio.run(main())

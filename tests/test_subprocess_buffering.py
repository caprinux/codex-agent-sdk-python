"""Tests for JSONL stream parsing edge cases in SubprocessCLITransport."""

import json
from unittest.mock import MagicMock

import anyio
from anyio import EndOfStream

from codex_agent_sdk._errors import CLIJSONDecodeError
from codex_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from codex_agent_sdk.types import CodexAgentOptions


class MockByteReceiveStream:
    """Simulates a raw stdout byte stream conforming to anyio's ByteReceiveStream protocol.

    ``TextReceiveStream`` calls ``await self.transport_stream.receive()``
    which must return ``bytes`` and raise ``EndOfStream`` when exhausted.
    """

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [c.encode() for c in chunks]
        self._index = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._index >= len(self._chunks):
            raise EndOfStream
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self) -> None:
        pass


def _make_transport_with_mock_stdout(chunks: list[str]) -> SubprocessCLITransport:
    """Create a transport with a mocked process that emits the given stdout chunks."""
    options = CodexAgentOptions(cli_path="/fake/codex")
    transport = SubprocessCLITransport(options, "test")

    mock_process = MagicMock()
    mock_process.returncode = None
    mock_process.stdout = MockByteReceiveStream(chunks)
    transport._process = mock_process

    return transport


async def _collect_messages(transport: SubprocessCLITransport) -> list[dict]:
    messages = []
    async for msg in transport.read_messages():
        messages.append(msg)
    return messages


class TestJsonlParsing:
    def test_single_complete_line(self) -> None:
        async def _test() -> None:
            data = {"type": "thread.started", "thread_id": "abc"}
            transport = _make_transport_with_mock_stdout([json.dumps(data) + "\n"])
            messages = await _collect_messages(transport)
            assert len(messages) == 1
            assert messages[0] == data

        anyio.run(_test)

    def test_multiple_lines_in_one_chunk(self) -> None:
        async def _test() -> None:
            line1 = json.dumps({"type": "turn.started"})
            line2 = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}})
            transport = _make_transport_with_mock_stdout([f"{line1}\n{line2}\n"])
            messages = await _collect_messages(transport)
            assert len(messages) == 2
            assert messages[0]["type"] == "turn.started"
            assert messages[1]["type"] == "turn.completed"

        anyio.run(_test)

    def test_json_split_across_chunks(self) -> None:
        async def _test() -> None:
            full_line = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "id": "m1", "text": "hello"}})
            # Split in the middle
            mid = len(full_line) // 2
            chunk1 = full_line[:mid]
            chunk2 = full_line[mid:] + "\n"
            transport = _make_transport_with_mock_stdout([chunk1, chunk2])
            messages = await _collect_messages(transport)
            assert len(messages) == 1
            assert messages[0]["type"] == "item.completed"

        anyio.run(_test)

    def test_empty_lines_ignored(self) -> None:
        async def _test() -> None:
            data = json.dumps({"type": "turn.started"})
            transport = _make_transport_with_mock_stdout([f"\n\n{data}\n\n"])
            messages = await _collect_messages(transport)
            assert len(messages) == 1

        anyio.run(_test)

    def test_whitespace_only_lines_ignored(self) -> None:
        async def _test() -> None:
            data = json.dumps({"type": "turn.started"})
            transport = _make_transport_with_mock_stdout([f"   \n  \n{data}\n  \n"])
            messages = await _collect_messages(transport)
            assert len(messages) == 1

        anyio.run(_test)

    def test_trailing_partial_json_at_eof(self) -> None:
        async def _test() -> None:
            complete = json.dumps({"type": "turn.started"})
            # Partial JSON at end without newline — should be discarded
            transport = _make_transport_with_mock_stdout([f"{complete}\n{{\"type\": \"inc"])
            messages = await _collect_messages(transport)
            assert len(messages) == 1
            assert messages[0]["type"] == "turn.started"

        anyio.run(_test)

    def test_complete_json_at_eof_without_newline(self) -> None:
        async def _test() -> None:
            data = {"type": "turn.started"}
            # No trailing newline
            transport = _make_transport_with_mock_stdout([json.dumps(data)])
            messages = await _collect_messages(transport)
            assert len(messages) == 1
            assert messages[0] == data

        anyio.run(_test)

    def test_invalid_json_raises(self) -> None:
        async def _test() -> None:
            transport = _make_transport_with_mock_stdout(["not valid json\n"])
            try:
                await _collect_messages(transport)
                assert False, "Should have raised"
            except CLIJSONDecodeError as e:
                assert "not valid json" in e.line

        anyio.run(_test)

    def test_many_small_chunks(self) -> None:
        async def _test() -> None:
            full = json.dumps({"type": "thread.started", "thread_id": "x"}) + "\n"
            # Send one character at a time
            chunks = [c for c in full]
            transport = _make_transport_with_mock_stdout(chunks)
            messages = await _collect_messages(transport)
            assert len(messages) == 1
            assert messages[0]["thread_id"] == "x"

        anyio.run(_test)

    def test_multiple_events_interleaved_chunks(self) -> None:
        async def _test() -> None:
            event1 = json.dumps({"type": "turn.started"}) + "\n"
            event2 = json.dumps({"type": "item.started", "item": {"type": "agent_message", "id": "m1", "text": ""}}) + "\n"
            event3 = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}}) + "\n"

            # Chunk boundaries don't align with event boundaries
            combined = event1 + event2 + event3
            chunks = [combined[:20], combined[20:60], combined[60:]]
            transport = _make_transport_with_mock_stdout(chunks)
            messages = await _collect_messages(transport)
            assert len(messages) == 3

        anyio.run(_test)

    def test_empty_stream(self) -> None:
        async def _test() -> None:
            transport = _make_transport_with_mock_stdout([])
            messages = await _collect_messages(transport)
            assert len(messages) == 0

        anyio.run(_test)

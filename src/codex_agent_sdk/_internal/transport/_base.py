"""Abstract base transport for communicating with the Codex CLI."""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator


class Transport(abc.ABC):
    """Abstract interface for a Codex CLI transport.

    A transport is responsible for:
    1. Spawning / connecting to the CLI process
    2. Writing the user prompt to stdin
    3. Reading JSONL events from stdout
    4. Tearing down the process
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the connection (spawn the process)."""

    @abc.abstractmethod
    async def write(self, data: str) -> None:
        """Write *data* to the CLI's stdin."""

    @abc.abstractmethod
    async def end_input(self) -> None:
        """Signal that no more input will be sent (close stdin)."""

    @abc.abstractmethod
    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed JSON objects from the CLI's stdout."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Shut down the transport and clean up resources."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return ``True`` if the transport is connected and the process is alive."""

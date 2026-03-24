"""E2E test configuration — requires a real Codex CLI and API key."""

import os
import shutil

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end tests that call the real Codex API")


def _has_codex_auth() -> bool:
    """Check if codex has an auth config (logged in via `codex auth`)."""
    auth_path = os.path.expanduser("~/.codex/auth.json")
    return os.path.isfile(auth_path)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip e2e tests if prerequisites are missing."""
    codex_available = shutil.which("codex") is not None
    api_key = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    has_auth = _has_codex_auth()

    for item in items:
        item.add_marker(pytest.mark.e2e)

        if not codex_available:
            item.add_marker(pytest.mark.skip(reason="codex CLI not found on PATH"))
        elif not api_key and not has_auth:
            item.add_marker(
                pytest.mark.skip(
                    reason="No API key: set CODEX_API_KEY/OPENAI_API_KEY or run `codex auth`"
                )
            )

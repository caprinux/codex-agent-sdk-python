"""Tests for SubprocessCLITransport: command building, env, CLI resolution."""

import os
import stat
import tempfile
from unittest.mock import patch

import pytest

from codex_agent_sdk._errors import CLINotFoundError
from codex_agent_sdk._internal.transport.subprocess_cli import (
    SubprocessCLITransport,
    _find_cli,
)
from codex_agent_sdk.types import (
    ApprovalPolicy,
    CodexAgentOptions,
    ImageInput,
    ReasoningEffort,
    SandboxMode,
    TextInput,
    WebSearch,
)


def _make_options(**kwargs: object) -> CodexAgentOptions:
    """Helper to build options with a fake CLI path."""
    defaults = {"cli_path": "/fake/codex"}
    defaults.update(kwargs)
    return CodexAgentOptions(**defaults)  # type: ignore[arg-type]


def _make_transport(
    prompt: object = "test prompt",
    resume_thread_id: str | None = None,
    **opts_kwargs: object,
) -> SubprocessCLITransport:
    options = _make_options(**opts_kwargs)
    return SubprocessCLITransport(options, prompt, resume_thread_id=resume_thread_id)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI resolution
# ---------------------------------------------------------------------------


class TestFindCli:
    def test_explicit_path(self, tmp_path: object) -> None:
        # Create a fake executable
        fake_cli = tempfile.NamedTemporaryFile(delete=False, suffix="_codex")
        os.chmod(fake_cli.name, stat.S_IRWXU)
        try:
            result = _find_cli(fake_cli.name)
            assert result == fake_cli.name
        finally:
            os.unlink(fake_cli.name)

    def test_env_var(self) -> None:
        fake_path = tempfile.NamedTemporaryFile(delete=False, suffix="_codex")
        os.chmod(fake_path.name, stat.S_IRWXU)
        try:
            with patch.dict(os.environ, {"CODEX_CLI_PATH": fake_path.name}):
                result = _find_cli()
                assert result == fake_path.name
        finally:
            os.unlink(fake_path.name)

    def test_not_found_raises(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("shutil.which", return_value=None),
            patch("os.path.isfile", return_value=False),
        ):
            with pytest.raises(CLINotFoundError) as exc_info:
                _find_cli("/nonexistent/codex")
            assert exc_info.value.searched_paths is not None
            assert "/nonexistent/codex" in exc_info.value.searched_paths


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


@patch(
    "codex_agent_sdk._internal.transport.subprocess_cli._find_cli",
    return_value="/fake/codex",
)
class TestBuildCommand:
    def test_minimal(self, _mock_find: object) -> None:
        t = _make_transport()
        cmd = t._build_command()
        assert cmd[0] == "/fake/codex"
        assert cmd[1] == "exec"
        assert "--experimental-json" in cmd
        # No extra flags for defaults
        assert "--model" not in cmd
        assert "--sandbox" not in cmd

    def test_model(self, _mock_find: object) -> None:
        t = _make_transport(model="o4-mini")
        cmd = t._build_command()
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o4-mini"

    def test_sandbox_modes(self, _mock_find: object) -> None:
        for mode in SandboxMode:
            t = _make_transport(sandbox=mode)
            cmd = t._build_command()
            idx = cmd.index("--sandbox")
            assert cmd[idx + 1] == mode.value

    def test_cwd(self, _mock_find: object) -> None:
        t = _make_transport(cwd="/my/project")
        cmd = t._build_command()
        idx = cmd.index("-C")
        assert cmd[idx + 1] == "/my/project"

    def test_additional_writable_dirs(self, _mock_find: object) -> None:
        t = _make_transport(additional_writable_dirs=["/tmp/a", "/tmp/b"])
        cmd = t._build_command()
        add_dir_indices = [i for i, v in enumerate(cmd) if v == "--add-dir"]
        assert len(add_dir_indices) == 2
        assert cmd[add_dir_indices[0] + 1] == "/tmp/a"
        assert cmd[add_dir_indices[1] + 1] == "/tmp/b"

    def test_full_auto(self, _mock_find: object) -> None:
        t = _make_transport(full_auto=True)
        cmd = t._build_command()
        assert "--full-auto" in cmd

    def test_skip_git_repo_check(self, _mock_find: object) -> None:
        t = _make_transport(skip_git_repo_check=True)
        cmd = t._build_command()
        assert "--skip-git-repo-check" in cmd

    def test_ephemeral(self, _mock_find: object) -> None:
        t = _make_transport(ephemeral=True)
        cmd = t._build_command()
        assert "--ephemeral" in cmd

    def test_profile(self, _mock_find: object) -> None:
        t = _make_transport(profile="myprofile")
        cmd = t._build_command()
        idx = cmd.index("--profile")
        assert cmd[idx + 1] == "myprofile"

    def test_images_from_options(self, _mock_find: object) -> None:
        t = _make_transport(images=["/tmp/img1.png", "/tmp/img2.jpg"])
        cmd = t._build_command()
        image_indices = [i for i, v in enumerate(cmd) if v == "--image"]
        assert len(image_indices) == 2
        assert cmd[image_indices[0] + 1] == "/tmp/img1.png"
        assert cmd[image_indices[1] + 1] == "/tmp/img2.jpg"

    def test_images_from_structured_input(self, _mock_find: object) -> None:
        prompt = [
            TextInput(text="Describe this"),
            ImageInput(path="/tmp/screenshot.png"),
        ]
        t = _make_transport(prompt=prompt)
        cmd = t._build_command()
        assert "--image" in cmd
        idx = cmd.index("--image")
        assert cmd[idx + 1] == "/tmp/screenshot.png"

    def test_output_schema_file(self, _mock_find: object) -> None:
        t = _make_transport(output_schema_file="/tmp/schema.json")
        cmd = t._build_command()
        idx = cmd.index("--output-schema")
        assert cmd[idx + 1] == "/tmp/schema.json"

    def test_base_url_as_config(self, _mock_find: object) -> None:
        t = _make_transport(base_url="https://custom.api.com")
        cmd = t._build_command()
        config_indices = [i for i, v in enumerate(cmd) if v == "--config"]
        config_values = [cmd[i + 1] for i in config_indices]
        assert "openai_base_url=https://custom.api.com" in config_values

    def test_reasoning_effort_as_config(self, _mock_find: object) -> None:
        t = _make_transport(reasoning_effort=ReasoningEffort.HIGH)
        cmd = t._build_command()
        config_indices = [i for i, v in enumerate(cmd) if v == "--config"]
        config_values = [cmd[i + 1] for i in config_indices]
        assert any("model_reasoning_effort" in v and "high" in v for v in config_values)

    def test_web_search_as_config(self, _mock_find: object) -> None:
        t = _make_transport(web_search=WebSearch.LIVE)
        cmd = t._build_command()
        config_indices = [i for i, v in enumerate(cmd) if v == "--config"]
        config_values = [cmd[i + 1] for i in config_indices]
        assert any("web_search" in v and "live" in v for v in config_values)

    def test_approval_policy_as_config(self, _mock_find: object) -> None:
        t = _make_transport(approval_policy=ApprovalPolicy.ON_REQUEST)
        cmd = t._build_command()
        config_indices = [i for i, v in enumerate(cmd) if v == "--config"]
        config_values = [cmd[i + 1] for i in config_indices]
        assert any("approval_policy" in v and "on-request" in v for v in config_values)

    def test_custom_config_overrides(self, _mock_find: object) -> None:
        t = _make_transport(config_overrides={"key1": "val1", "key2": "val2"})
        cmd = t._build_command()
        config_indices = [i for i, v in enumerate(cmd) if v == "--config"]
        config_values = [cmd[i + 1] for i in config_indices]
        assert "key1=val1" in config_values
        assert "key2=val2" in config_values

    def test_resume_thread_id(self, _mock_find: object) -> None:
        t = _make_transport(resume_thread_id="thread-abc-123")
        cmd = t._build_command()
        assert cmd[-2] == "resume"
        assert cmd[-1] == "thread-abc-123"

    def test_resume_after_other_flags(self, _mock_find: object) -> None:
        t = _make_transport(model="o4-mini", resume_thread_id="tid")
        cmd = t._build_command()
        resume_idx = cmd.index("resume")
        model_idx = cmd.index("--model")
        assert resume_idx > model_idx

    def test_all_options_together(self, _mock_find: object) -> None:
        t = _make_transport(
            model="o4-mini",
            sandbox=SandboxMode.WORKSPACE_WRITE,
            cwd="/proj",
            full_auto=True,
            ephemeral=True,
            skip_git_repo_check=True,
            profile="test",
            images=["/img.png"],
            base_url="https://api.example.com",
            reasoning_effort=ReasoningEffort.MEDIUM,
        )
        cmd = t._build_command()
        assert "--model" in cmd
        assert "--sandbox" in cmd
        assert "-C" in cmd
        assert "--full-auto" in cmd
        assert "--ephemeral" in cmd
        assert "--skip-git-repo-check" in cmd
        assert "--profile" in cmd
        assert "--image" in cmd
        assert "--config" in cmd


# ---------------------------------------------------------------------------
# Environment building
# ---------------------------------------------------------------------------


class TestBuildEnv:
    def test_default_env_includes_originator(self) -> None:
        t = _make_transport()
        env = t._build_env()
        assert env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] == "codex_sdk_py"

    def test_api_key_set(self) -> None:
        t = _make_transport(api_key="sk-test-123")
        env = t._build_env()
        assert env["CODEX_API_KEY"] == "sk-test-123"

    def test_custom_env_merged(self) -> None:
        t = _make_transport(env={"MY_VAR": "my_value"})
        env = t._build_env()
        assert env["MY_VAR"] == "my_value"
        assert "PATH" in env  # OS env is preserved

    def test_custom_env_overrides_os_env(self) -> None:
        t = _make_transport(env={"PATH": "/custom/path"})
        env = t._build_env()
        assert env["PATH"] == "/custom/path"


# ---------------------------------------------------------------------------
# Prompt text extraction
# ---------------------------------------------------------------------------


class TestGetPromptText:
    def test_string_prompt(self) -> None:
        t = _make_transport(prompt="Hello world")
        assert t._get_prompt_text() == "Hello world"

    def test_structured_text_only(self) -> None:
        prompt = [TextInput(text="Part 1"), TextInput(text="Part 2")]
        t = _make_transport(prompt=prompt)
        assert t._get_prompt_text() == "Part 1\n\nPart 2"

    def test_structured_mixed_ignores_images(self) -> None:
        prompt = [
            TextInput(text="Describe this"),
            ImageInput(path="/img.png"),
            TextInput(text="And this"),
        ]
        t = _make_transport(prompt=prompt)
        assert t._get_prompt_text() == "Describe this\n\nAnd this"

    def test_structured_images_only(self) -> None:
        prompt = [ImageInput(path="/a.png"), ImageInput(path="/b.png")]
        t = _make_transport(prompt=prompt)
        assert t._get_prompt_text() == ""


# ---------------------------------------------------------------------------
# Transport state
# ---------------------------------------------------------------------------


class TestTransportState:
    def test_not_ready_before_connect(self) -> None:
        t = _make_transport()
        assert t.is_ready() is False

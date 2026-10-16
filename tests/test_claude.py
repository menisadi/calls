from __future__ import annotations

import subprocess
from typing import Any

import pytest

from calls import claude, cli
from calls.config import Config


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch):
    """Pretend `claude` is installed; record each subprocess call."""
    calls: list[dict[str, Any]] = []

    def run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="  hello \n", stderr="")

    monkeypatch.setattr(claude.shutil, "which", lambda name: "/bin/claude")
    monkeypatch.setattr(claude.subprocess, "run", run)
    return calls


class TestBackend:
    def test_runs_headless_without_tools_and_api_key(
        self, fake_claude, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

        out = claude.ClaudeBackend("sonnet").translate("shalom")

        assert out == "hello"
        call = fake_claude[0]
        assert call["input"] == "shalom"
        assert "ANTHROPIC_API_KEY" not in call["env"]
        assert call["command"][:2] == ["claude", "-p"]
        assert "--no-session-persistence" in call["command"]
        assert call["command"][call["command"].index("--model") + 1] == "sonnet"

    def test_nonzero_exit_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            claude.subprocess,
            "run",
            lambda command, **kw: subprocess.CompletedProcess(command, 1, "", "boom"),
        )
        with pytest.raises(claude.ClaudeError, match="boom"):
            claude.ClaudeBackend("sonnet").run("sys", "text")

    def test_missing_cli_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(claude.shutil, "which", lambda name: None)
        with pytest.raises(claude.ClaudeError):
            claude.ClaudeBackend.load("sonnet")


def _translate(config: Config, argv: list[str]) -> int:
    args = cli.build_parser(config).parse_args(["translate", *argv])
    return args.handler(args, config)


class TestTranslateWithClaude:
    def test_translates_and_records_the_engine(
        self, config: Config, make_call, fake_claude
    ):
        make_call("[A]_[0501111111]_2026-08-30_20-22-21", transcript="שלום")

        result = _translate(config, ["--engine", "claude", "--last"])

        assert result == 0
        assert len(fake_claude) == 1

    def test_over_the_limit_asks_and_declining_runs_nothing(
        self, config: Config, make_call, fake_claude, monkeypatch: pytest.MonkeyPatch
    ):
        config.claude_confirm_above = 1
        make_call("[A]_[0501111111]_2026-08-30_20-22-21", transcript="a")
        make_call("[B]_[0502222222]_2026-08-31_20-22-21", transcript="b")
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")

        result = _translate(config, ["--engine", "claude", "--untranslated"])

        assert result == 1
        assert fake_claude == []

    def test_yes_skips_the_prompt(
        self, config: Config, make_call, fake_claude, monkeypatch: pytest.MonkeyPatch
    ):
        config.claude_confirm_above = 1
        make_call("[A]_[0501111111]_2026-08-30_20-22-21", transcript="a")
        make_call("[B]_[0502222222]_2026-08-31_20-22-21", transcript="b")
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": pytest.fail("should not prompt")
        )

        result = _translate(config, ["--engine", "claude", "--untranslated", "-y"])

        assert result == 0
        assert len(fake_claude) == 2

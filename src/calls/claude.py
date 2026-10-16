"""Opt-in translation and summaries through the `claude` CLI.

Runs `claude -p` headless, so it uses the Claude Code subscription login
rather than API billing. Transcripts leave the machine, which is why this is
never the default engine.

The subprocess is configured as a plain text-in/text-out call: no tools, no
saved session, no user/project settings, and a throwaway working directory so
no CLAUDE.md or hooks get picked up. `--bare` is deliberately not used, as it
skips the OAuth login that subscription auth depends on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from . import summarize, translate

TIMEOUT_SECONDS = 300


class ClaudeError(Exception):
    """`claude` is missing, failed, or produced nothing usable."""


class ClaudeBackend:
    """Drop-in for Translator/Summarizer, backed by the `claude` CLI."""

    def __init__(self, model: str):
        self.model = model
        self.model_path = f"claude:{model}"

    @classmethod
    def load(cls, model: str) -> ClaudeBackend:
        if shutil.which("claude") is None:
            raise ClaudeError("`claude` is not on PATH (is Claude Code installed?)")
        return cls(model)

    def run(self, system: str, text: str) -> str:
        # An API key in the environment would switch `claude` from the
        # subscription to API billing.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        command = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--no-session-persistence",
            "--setting-sources",
            "",
        ]
        try:
            with tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run(
                    command,
                    input=text,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=cwd,
                    timeout=TIMEOUT_SECONDS,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClaudeError(f"claude failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ClaudeError(f"claude exited {result.returncode}: {detail}")
        return result.stdout.strip()

    def translate(self, transcript: str) -> str:
        return self.run(translate.SYSTEM, transcript)

    def summarize(self, translation: str) -> str:
        return self.run(summarize.SYSTEM, translation)

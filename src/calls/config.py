"""Paths, models and tunables, all overridable by environment variable.

Collected here rather than spread across modules so that a test (or a second
archive) can point the whole program somewhere else by setting one thing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default region for bare local numbers. Israel, matching the recorder's own
# locale; override if the phone's address book is elsewhere.
DEFAULT_COUNTRY_CODE = "972"

AUDIO_SUFFIXES = (
    ".opus",
    ".m4a",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    ".amr",
)


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass
class Config:
    """Resolved configuration for one run."""

    calls_dir: Path
    phonerec_dir: Path
    db_path: Path
    markdown_path: Path
    scribe_command: str
    model_he: str
    model_en: str
    lang: str
    max_tokens: int
    country_code: str
    audio_suffixes: tuple[str, ...] = field(default=AUDIO_SUFFIXES)

    @classmethod
    def from_env(cls) -> Config:
        home = Path.home()
        calls_dir = _path_env("CALLS_DIR", home / "Recordings" / "calls")
        return cls(
            calls_dir=calls_dir,
            # Originals are left in place by `call import`, so for a recording
            # that is still there the incoming/outgoing split recovers the call
            # direction, which the flattened filename doesn't carry.
            phonerec_dir=_path_env(
                "CALLS_PHONEREC_DIR", home / "Recordings" / "PhoneRec"
            ),
            db_path=_path_env("CALLS_DB", calls_dir / "calls.db"),
            markdown_path=_path_env("CALLS_MARKDOWN", calls_dir / "index.md"),
            # Transcription stays an external shell tool: it is an
            # ffmpeg/whisper-cli pipeline, and is useful on arbitrary audio
            # with no knowledge of this archive.
            scribe_command=os.environ.get("CALLS_SCRIBE", "callscribe"),
            model_he=os.environ.get(
                "CALLS_MODEL_HE",
                str(home / ".local/share/mlx/DictaLM-3.0-1.7B-Instruct-bf16"),
            ),
            model_en=os.environ.get(
                "CALLS_MODEL_EN", "mlx-community/Qwen3-4B-4bit-DWQ-053125"
            ),
            lang=os.environ.get("CALLS_LANG", "he"),
            max_tokens=int(os.environ.get("CALLS_MAX_TOKENS", "300")),
            country_code=os.environ.get("CALLS_COUNTRY_CODE", DEFAULT_COUNTRY_CODE),
        )

    def model_for_lang(self, lang: str) -> str:
        return self.model_en if lang == "en" else self.model_he

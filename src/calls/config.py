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
    scribe_command: str
    model_he: str
    model_en: str
    model_translate: str
    lang: str
    max_tokens: int
    translate_max_tokens: int
    country_code: str
    audio_suffixes: tuple[str, ...] = field(default=AUDIO_SUFFIXES)

    @classmethod
    def from_env(cls) -> Config:
        home = Path.home()
        calls_dir = _path_env("CALLS_DIR", home / "Recordings" / "calls")
        model_he = os.environ.get(
            "CALLS_MODEL_HE",
            str(home / ".local/share/mlx/DictaLM-3.0-1.7B-Instruct-bf16"),
        )
        return cls(
            calls_dir=calls_dir,
            # Originals are left in place by `call import`, so for a recording
            # that is still there the incoming/outgoing split recovers the call
            # direction, which the flattened filename doesn't carry.
            phonerec_dir=_path_env(
                "CALLS_PHONEREC_DIR", home / "Recordings" / "PhoneRec"
            ),
            db_path=_path_env("CALLS_DB", calls_dir / "calls.db"),
            # Transcription stays an external shell tool: it is an
            # ffmpeg/whisper-cli pipeline, and is useful on arbitrary audio
            # with no knowledge of this archive.
            scribe_command=os.environ.get("CALLS_SCRIBE", "callscribe"),
            model_he=model_he,
            model_en=os.environ.get(
                "CALLS_MODEL_EN", "mlx-community/Qwen3-4B-4bit-DWQ-053125"
            ),
            # Hebrew-to-English translation defaults to the same model as
            # Hebrew tagging: DictaLM's Hebrew tuning produced more faithful
            # translations than Qwen3-4B in testing (e.g. it didn't mistake
            # "brownies" for "bronzes").
            model_translate=os.environ.get("CALLS_MODEL_TRANSLATE", model_he),
            lang=os.environ.get("CALLS_LANG", "he"),
            max_tokens=int(os.environ.get("CALLS_MAX_TOKENS", "300")),
            # A ceiling, not a target: generation stops at the model's own end
            # token well before this in practice. It exists so a long call's
            # translation isn't silently truncated mid-sentence by a budget
            # sized for a short one - see translate.py's per-call scaling.
            translate_max_tokens=int(
                os.environ.get("CALLS_TRANSLATE_MAX_TOKENS", "8000")
            ),
            country_code=os.environ.get("CALLS_COUNTRY_CODE", DEFAULT_COUNTRY_CODE),
        )

    def model_for_lang(self, lang: str) -> str:
        return self.model_en if lang == "en" else self.model_he

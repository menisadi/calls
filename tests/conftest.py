"""Fixtures building a throwaway archive on disk.

The recordings are empty files, not real audio: ffprobe fails on them and the
probe degrades to blank duration/channels, which is exactly the behaviour worth
testing anyway. Nothing here touches the real archive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calls.config import AUDIO_SUFFIXES, Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    phonerec_dir = tmp_path / "PhoneRec"
    (phonerec_dir / "incoming").mkdir(parents=True)
    (phonerec_dir / "outgoing").mkdir(parents=True)
    return Config(
        calls_dir=calls_dir,
        phonerec_dir=phonerec_dir,
        db_path=calls_dir / "calls.db",
        scribe_command="callscribe-not-installed",
        model_he="model-he",
        model_en="model-en",
        model_translate="model-translate",
        lang="he",
        max_tokens=300,
        translate_max_tokens=8000,
        country_code="972",
        audio_suffixes=AUDIO_SUFFIXES,
    )


@pytest.fixture
def make_call(config: Config):
    """Create a recording, and optionally its transcript, in the archive."""

    def _make(base: str, transcript: str | None = None, suffix: str = ".opus") -> Path:
        recording = config.calls_dir / f"{base}{suffix}"
        recording.write_bytes(b"")
        if transcript is not None:
            recording.with_suffix(".txt").write_text(transcript, encoding="utf-8")
        return recording

    return _make


@pytest.fixture
def make_original(config: Config):
    """Create a recording in the recorder's incoming/outgoing folder."""

    def _make(base: str, folder: str, suffix: str = ".opus") -> Path:
        original = config.phonerec_dir / folder / f"{base}{suffix}"
        original.write_bytes(b"")
        return original

    return _make

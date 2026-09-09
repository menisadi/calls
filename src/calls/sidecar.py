"""The JSON sidecar: one call's metadata and per-stage pipeline status.

The sidecar sits next to the recording as "<base>.json" and is the source of
truth for everything about a call that isn't the audio or the transcript text.
Every field except ``tag`` and ``direction`` is derived from observable facts
(the filename, ffprobe, whether a transcript exists), so a refresh is
idempotent and self-healing: it can never disagree with what's on disk.

The pipeline stages are derived the same way rather than being flags someone
has to remember to set. That is the point: the importer used to treat "a .txt
exists" as "this call is fully handled", so a call whose transcription
succeeded but whose tagging failed was skipped forever and never made it into
the index. Deriving the transcript and the tag stages separately means a
half-finished call is always visible as such.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AUDIO_SUFFIXES, Config

# Bump when the on-disk shape changes in a way older readers can't handle.
SCHEMA_VERSION = 1

# Recorder-produced names look like "[Contact]_[Phone]_YYYY-MM-DD_HH-MM-SS".
# The phone field is whatever the recorder had on hand: a local number, an
# E.164 number, a short code, or a plain name (e.g. "[שופרסל]_[Shufersal]").
FILENAME_PATTERN = re.compile(
    r"^\[(?P<contact>[^\]]*)\]_\[(?P<phone>[^\]]*)\]_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
)

UNKNOWN_PHONES = frozenset({"", "null", "none", "unknown", "private", "anonymous"})

DIRECTIONS = ("in", "out", "unknown")

# Folder name in the recorder's tree -> direction it implies.
SOURCE_DIRECTIONS = (("incoming", "in"), ("outgoing", "out"))


class SidecarError(Exception):
    """A sidecar exists but cannot be used."""


@dataclass
class Tag:
    """A topic tag: one general category plus one or two specific subjects.

    Deliberately short, and deliberately not a prose summary: with a
    "summarize" framing the small local models ignore length instructions and
    produce sentences or bullet lists, and the index exists to be scanned.
    """

    general: str = ""
    specific: list[str] = field(default_factory=list)
    source: str | None = None

    def render(self) -> str:
        parts = [self.general, *self.specific]
        return ", ".join(part.strip() for part in parts if part.strip())

    def is_present(self) -> bool:
        return bool(self.general.strip() or self.specific)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "general": self.general,
            "specific": self.specific,
        }
        if self.source:
            payload["source"] = self.source
        return payload

    @classmethod
    def from_json(cls, data: Any) -> Tag:
        if not isinstance(data, dict):
            return cls()
        specific_raw = data.get("specific") or []
        if not isinstance(specific_raw, list):
            specific_raw = [specific_raw]
        source = data.get("source")
        return cls(
            general=str(data.get("general", "")).strip(),
            specific=[str(item).strip() for item in specific_raw if str(item).strip()],
            source=str(source) if source else None,
        )


@dataclass
class CallName:
    """What the recorder's filename tells us about a call."""

    contact: str
    phone_raw: str
    date: str
    time: str

    @classmethod
    def parse(cls, base_name: str) -> CallName | None:
        match = FILENAME_PATTERN.match(base_name)
        if match is None:
            return None
        date = match.group("date")
        time = match.group("time").replace("-", ":")
        # The regex only checks shape, so validate the instant itself: this
        # date is what orders the index, and a nonsense
        # value there is worse than falling back to the file's mtime.
        try:
            datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        return cls(
            contact=match.group("contact").strip(),
            phone_raw=match.group("phone").strip(),
            date=date,
            time=time,
        )


def normalize_phone(raw: str, country_code: str) -> str | None:
    """Best-effort E.164 for a recorder-supplied phone field.

    Returns None when the field plainly isn't a number, so that grouping by
    phone never silently merges distinct non-numeric labels. Deliberately not
    the `phonenumbers` library: this only has to collapse the one real
    ambiguity seen in practice, the same contact appearing as both
    "0547602488" and "+972547602488".
    """
    cleaned = re.sub(r"[\s\-().]", "", raw.strip())
    if cleaned.lower() in UNKNOWN_PHONES:
        return None
    if cleaned.startswith("+"):
        return cleaned if cleaned[1:].isdigit() else None
    if not cleaned.isdigit():
        return None
    if cleaned.startswith("00"):
        return f"+{cleaned[2:]}"
    # Short codes (service numbers like 1455) have no country context and are
    # meaningless with a prefix bolted on, so they stay as dialled.
    if len(cleaned) <= 6:
        return cleaned
    if cleaned.startswith("0"):
        return f"+{country_code}{cleaned[1:]}"
    return f"+{cleaned}"


@dataclass
class Probe:
    duration_seconds: float | None = None
    channels: int | None = None


def probe_audio(path: Path) -> Probe:
    """Duration and channel count via ffprobe; blank if it can't be read."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=channels",
                "-select_streams",
                "a:0",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(completed.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return Probe()

    duration_raw = data.get("format", {}).get("duration")
    try:
        duration = round(float(duration_raw), 1) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None

    streams = data.get("streams") or [{}]
    return Probe(duration_seconds=duration, channels=streams[0].get("channels"))


def find_recording(
    base: Path, suffixes: tuple[str, ...] = AUDIO_SUFFIXES
) -> Path | None:
    """The audio file for a sidecar/transcript base path, if one exists."""
    for suffix in suffixes:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def sidecar_path_for(recording: Path) -> Path:
    return recording.with_suffix(".json")


def translation_path_for(recording: Path) -> Path:
    return recording.with_suffix(".en.txt")


def read(sidecar_path: Path) -> dict[str, Any]:
    """The sidecar as raw JSON, or an empty dict if there isn't one."""
    if not sidecar_path.is_file():
        return {}
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SidecarError(f"malformed sidecar {sidecar_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SidecarError(f"malformed sidecar {sidecar_path}: expected an object")
    return data


def write(sidecar_path: Path, sidecar: dict[str, Any]) -> None:
    """Atomic replace, so an interrupted write can't truncate the sidecar."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(sidecar_path.parent), prefix=f".{sidecar_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, sidecar_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def stage_status(sidecar: dict[str, Any], stage: str) -> str:
    """Status of one pipeline stage.

    A missing sidecar, missing key or unreadable value all mean "not done",
    which is the safe answer: the stage simply runs again.
    """
    stages = sidecar.get("stages")
    if not isinstance(stages, dict):
        return "missing"
    entry = stages.get(stage)
    if not isinstance(entry, dict):
        return "missing"
    return str(entry.get("status") or "missing")


def now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def infer_direction(recording: Path, phonerec_dir: Path) -> str:
    """Direction from the surviving original, then from the contact field."""
    for folder, direction in SOURCE_DIRECTIONS:
        if (phonerec_dir / folder / recording.name).exists():
            return direction
    parsed = CallName.parse(recording.stem)
    contact = (parsed.contact if parsed else "").lower()
    if contact == "outgoing":
        return "out"
    if contact == "incoming":
        return "in"
    return "unknown"


def refresh(
    recording: Path,
    config: Config,
    direction: str | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild the sidecar's derived fields, preserving what we don't own.

    ``tag`` and anything a future version adds survive untouched; everything
    else is recomputed from disk.
    """
    if existing is None:
        existing = read(sidecar_path_for(recording))

    sidecar = dict(existing)
    sidecar["schema"] = SCHEMA_VERSION
    sidecar["base"] = recording.stem
    sidecar["recording"] = recording.name

    parsed = CallName.parse(recording.stem)
    if parsed is not None:
        sidecar["contact"] = parsed.contact
        sidecar["phone_raw"] = parsed.phone_raw
        sidecar["phone"] = normalize_phone(parsed.phone_raw, config.country_code)
        sidecar["date"] = parsed.date
        sidecar["time"] = parsed.time
        sidecar["started_at"] = f"{parsed.date}T{parsed.time}"
    else:
        # Unparseable name: keep whatever a previous run recorded rather than
        # blanking it, and fall back to the file's own mtime for ordering.
        stamp = datetime.fromtimestamp(recording.stat().st_mtime).astimezone()
        sidecar.setdefault("contact", "")
        sidecar.setdefault("phone_raw", "")
        sidecar.setdefault("phone", None)
        sidecar["date"] = stamp.date().isoformat()
        sidecar["time"] = stamp.time().isoformat(timespec="seconds")
        sidecar["started_at"] = stamp.isoformat(timespec="seconds")

    # An explicit direction wins; otherwise never downgrade a known direction
    # to "unknown" just because the original has since been pruned from the
    # recorder's folder.
    if direction:
        sidecar["direction"] = direction
    else:
        inferred = infer_direction(recording, config.phonerec_dir)
        if inferred != "unknown" or sidecar.get("direction") in (None, "unknown"):
            sidecar["direction"] = inferred

    probe = probe_audio(recording)
    sidecar["duration_seconds"] = probe.duration_seconds
    sidecar["channels"] = probe.channels

    stages = dict(sidecar.get("stages") or {})

    transcript = recording.with_suffix(".txt")
    scribe = dict(stages.get("scribe") or {})
    if transcript.is_file() and transcript.stat().st_size > 0:
        words = len(transcript.read_text(encoding="utf-8").split())
        scribe["status"] = "ok" if words else "empty"
        scribe["words"] = words
        scribe.setdefault("at", now_stamp())
        sidecar["transcript"] = transcript.name
    else:
        scribe["status"] = "missing"
        scribe.pop("words", None)
        sidecar["transcript"] = None
    stages["scribe"] = scribe

    # The tagging step owns its stage's provenance (model, language); a refresh
    # only ever re-derives whether a usable tag is actually present.
    topic = dict(stages.get("topic") or {})
    topic["status"] = (
        "ok" if Tag.from_json(sidecar.get("tag")).is_present() else "missing"
    )
    stages["topic"] = topic

    # Mirrors the scribe stage: derived from the file on disk, so a deleted or
    # regenerated translation is reflected without re-running anything.
    # record_translation()'s "model" and "at" survive untouched, since only
    # "status" and "words" are set here.
    translation = translation_path_for(recording)
    translate = dict(stages.get("translate") or {})
    if translation.is_file() and translation.stat().st_size > 0:
        words = len(translation.read_text(encoding="utf-8").split())
        translate["status"] = "ok" if words else "empty"
        translate["words"] = words
        sidecar["transcript_en"] = translation.name
    else:
        translate["status"] = "missing"
        translate.pop("words", None)
        sidecar["transcript_en"] = None
    stages["translate"] = translate

    # Like topic: the summary lives in the sidecar itself (it's short, same as
    # a tag, not a full-length artifact), so a refresh only re-derives whether
    # one is present, preserving record_summary()'s model/at.
    summarize = dict(stages.get("summarize") or {})
    summarize["status"] = (
        "ok" if str(sidecar.get("summary") or "").strip() else "missing"
    )
    stages["summarize"] = summarize

    sidecar["stages"] = stages
    return sidecar


def record_tag(recording: Path, tag: Tag, model: str, lang: str) -> dict[str, Any]:
    """Merge a freshly generated tag into the sidecar, and return it.

    Only the fields the tagging step owns are written. If no sidecar exists yet
    the bare minimum is created from the filename, and a later refresh fills in
    duration, channels and direction - so the steps can run in either order
    without one clobbering the other's work.
    """
    path = sidecar_path_for(recording)
    sidecar = read(path)
    sidecar.setdefault("schema", SCHEMA_VERSION)
    sidecar.setdefault("base", recording.stem)

    parsed = CallName.parse(recording.stem)
    if parsed is not None:
        sidecar.setdefault("contact", parsed.contact)
        sidecar.setdefault("phone_raw", parsed.phone_raw)
        sidecar.setdefault("date", parsed.date)
        sidecar.setdefault("time", parsed.time)

    sidecar["tag"] = tag.to_json()

    stages = dict(sidecar.get("stages") or {})
    stages["topic"] = {
        "status": "ok",
        "at": now_stamp(),
        "model": model,
        "lang": lang,
    }
    sidecar["stages"] = stages

    write(path, sidecar)
    return sidecar


def record_translation(recording: Path, text: str, model: str) -> dict[str, Any]:
    """Write the English translation next to the transcript, and record it.

    Mirrors record_tag: only the fields this step owns are written, and a
    missing sidecar is created from the filename so translate can run before
    or after scribe/topic without clobbering their work.
    """
    translation = translation_path_for(recording)
    translation.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(translation.parent), prefix=f".{translation.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, translation)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    path = sidecar_path_for(recording)
    sidecar = read(path)
    sidecar.setdefault("schema", SCHEMA_VERSION)
    sidecar.setdefault("base", recording.stem)

    parsed = CallName.parse(recording.stem)
    if parsed is not None:
        sidecar.setdefault("contact", parsed.contact)
        sidecar.setdefault("phone_raw", parsed.phone_raw)
        sidecar.setdefault("date", parsed.date)
        sidecar.setdefault("time", parsed.time)

    sidecar["transcript_en"] = translation.name

    stages = dict(sidecar.get("stages") or {})
    stages["translate"] = {
        "status": "ok",
        "at": now_stamp(),
        "model": model,
    }
    sidecar["stages"] = stages

    write(path, sidecar)
    return sidecar


def record_summary(recording: Path, summary: str, model: str) -> dict[str, Any]:
    """Merge a freshly generated summary into the sidecar, and return it.

    Mirrors record_tag: only the fields this step owns are written, and a
    missing sidecar is created from the filename so summarize can run
    independently of scribe/topic/translate.
    """
    path = sidecar_path_for(recording)
    sidecar = read(path)
    sidecar.setdefault("schema", SCHEMA_VERSION)
    sidecar.setdefault("base", recording.stem)

    parsed = CallName.parse(recording.stem)
    if parsed is not None:
        sidecar.setdefault("contact", parsed.contact)
        sidecar.setdefault("phone_raw", parsed.phone_raw)
        sidecar.setdefault("date", parsed.date)
        sidecar.setdefault("time", parsed.time)

    sidecar["summary"] = summary

    stages = dict(sidecar.get("stages") or {})
    stages["summarize"] = {
        "status": "ok",
        "at": now_stamp(),
        "model": model,
    }
    sidecar["stages"] = stages

    write(path, sidecar)
    return sidecar


def _recording_sort_key(recording: Path) -> tuple[str, str, str]:
    """(date, time, path) for chronological ordering, oldest first.

    Sorting `Path`s directly sorts by filename, which starts with the contact
    name - not the date - so "the last one" was really "whichever contact
    name sorts last" (Hebrew names, being higher Unicode codepoints, sort
    after Latin ones regardless of when the call happened). Falls back to the
    file's own mtime for a name CallName.parse() can't read, exactly like
    refresh() does; the path is a tiebreaker so ordering stays deterministic
    when two recordings share a timestamp.
    """
    parsed = CallName.parse(recording.stem)
    if parsed is not None:
        date, time = parsed.date, parsed.time
    else:
        stamp = datetime.fromtimestamp(recording.stat().st_mtime).astimezone()
        date = stamp.date().isoformat()
        time = stamp.time().isoformat(timespec="seconds")
    return (date, time, str(recording))


def all_recordings(config: Config) -> list[Path]:
    """Every recording in the archive, oldest first."""
    if not config.calls_dir.is_dir():
        raise SidecarError(f"calls directory not found: {config.calls_dir}")
    return sorted(
        (
            path
            for path in config.calls_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in config.audio_suffixes
        ),
        key=_recording_sort_key,
    )


def resolve_input(raw: str, config: Config) -> Path:
    """Accept a recording, transcript or sidecar path interchangeably."""
    path = Path(raw).expanduser()
    if path.suffix.lower() in config.audio_suffixes:
        return path
    recording = find_recording(path.with_suffix(""), config.audio_suffixes)
    return recording if recording is not None else path

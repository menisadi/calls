"""Importing new recordings, and removing calls.

Replaces what used to be two bash scripts. They reached into the sidecar JSON
with `jq` to decide what to do, which is the wrong side of the boundary: the
schema is defined here, so the decisions belong here too.

What still needs doing is read from each call's sidecar rather than guessed
from which files exist. The old rule - "a transcript exists, so this call is
done" - meant a call whose transcription succeeded but whose tagging failed was
skipped on every later run and never reached the index. Per-stage status makes
each stage independently resumable, and the whole archive is swept every run,
so a call left half-finished gets picked up next time regardless of whether its
original is still in the recorder's folder.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import sidecar
from .config import Config
from .sidecar import SOURCE_DIRECTIONS, Tag

Reporter = Callable[[str], None]


class PipelineError(Exception):
    """The pipeline cannot proceed."""


@dataclass
class Plan:
    """What one call still needs."""

    recording: Path
    scribe: str
    topic: str

    @property
    def base(self) -> str:
        return self.recording.stem

    @property
    def is_complete(self) -> bool:
        return self.scribe == "ok" and self.topic == "ok"


@dataclass
class ImportReport:
    copied: list[str] = field(default_factory=list)
    would_copy: list[str] = field(default_factory=list)
    transcribed: list[str] = field(default_factory=list)
    tagged: list[str] = field(default_factory=list)
    pending: list[Plan] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _new_originals(config: Config) -> Iterator[tuple[Path, str]]:
    """Every recording in the recorder's folders, with its direction."""
    for folder, direction in SOURCE_DIRECTIONS:
        source_dir = config.phonerec_dir / folder
        if not source_dir.is_dir():
            continue
        for original in sorted(source_dir.iterdir()):
            if original.is_file() and original.suffix.lower() in config.audio_suffixes:
                yield original, direction


def copy_new(
    config: Config, report: ImportReport, dry_run: bool, notify: Reporter
) -> None:
    """Copy anything new out of the recorder's folder, recording direction.

    Originals are left in place: that folder is a receive-only Syncthing share,
    so deleting from it risks Syncthing either reintroducing the file or losing
    the recording on the phone.
    """
    config.calls_dir.mkdir(parents=True, exist_ok=True)

    for original, direction in _new_originals(config):
        destination = config.calls_dir / original.name
        if destination.exists():
            # Already imported. Refresh the sidecar anyway: while the original
            # is still here, this is the last chance to record the direction.
            if not dry_run:
                sidecar.write(
                    sidecar.sidecar_path_for(destination),
                    sidecar.refresh(destination, config, direction=direction),
                )
            continue
        if dry_run:
            notify(f"would copy ({direction}): {original.name}")
            report.would_copy.append(original.name)
            continue
        notify(f"copying ({direction}): {original.name}")
        shutil.copy2(original, destination)
        sidecar.write(
            sidecar.sidecar_path_for(destination),
            sidecar.refresh(destination, config, direction=direction),
        )
        report.copied.append(original.name)


def plan_for(recording: Path, config: Config, refresh: bool = True) -> Plan:
    """Refresh the sidecar, then report which stages are still outstanding."""
    if refresh:
        data = sidecar.refresh(recording, config)
        sidecar.write(sidecar.sidecar_path_for(recording), data)
    else:
        data = sidecar.read(sidecar.sidecar_path_for(recording))
    return Plan(
        recording=recording,
        scribe=sidecar.stage_status(data, "scribe"),
        topic=sidecar.stage_status(data, "topic"),
    )


def transcribe(recording: Path, config: Config, quiet: bool) -> None:
    """Run the external transcription tool on one recording."""
    command = [config.scribe_command]
    if quiet:
        command.append("-q")
    command.append(str(recording))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(
            f"transcription tool not found: {config.scribe_command}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(
            f"{config.scribe_command} failed on {recording.name}"
        ) from exc


def _read_transcript(recording: Path) -> str:
    transcript = recording.with_suffix(".txt")
    if not transcript.is_file():
        raise PipelineError(f"no transcript for {recording.name}")
    return transcript.read_text(encoding="utf-8")


def finish(
    plans: list[Plan],
    config: Config,
    lang: str,
    model_path: str | None,
    dry_run: bool,
    quiet: bool,
    notify: Reporter,
    report: ImportReport,
) -> None:
    """Bring every outstanding call up to date.

    The model is loaded once, and only if something actually needs tagging -
    loading it costs seconds, so a run with nothing to tag should not pay for
    it. A single bad recording (e.g. no speech detected) is recorded as a
    failure and the batch continues.
    """
    outstanding = [plan for plan in plans if not plan.is_complete]
    if dry_run:
        for plan in outstanding:
            notify(f"would run (scribe={plan.scribe} topic={plan.topic}): {plan.base}")
        report.pending = outstanding
        return

    tagger = None
    total = len(outstanding)
    for position, plan in enumerate(outstanding, start=1):
        prefix = f"[{position}/{total}]"
        try:
            if plan.scribe != "ok":
                notify(f"{prefix} transcribing: {plan.base}")
                transcribe(plan.recording, config, quiet)
                # Re-derive so the sidecar reflects the transcript that now
                # exists; without this the call still looks untranscribed.
                refreshed = sidecar.refresh(plan.recording, config)
                sidecar.write(sidecar.sidecar_path_for(plan.recording), refreshed)
                if sidecar.stage_status(refreshed, "scribe") != "ok":
                    raise PipelineError(f"no usable transcript for {plan.base}")
                report.transcribed.append(plan.base)

            if plan.topic != "ok":
                if tagger is None:
                    from .topic import Tagger

                    notify(
                        f"loading model: {model_path or config.model_for_lang(lang)}"
                    )
                    tagger = Tagger.load(config, lang, model_path)
                notify(f"{prefix} tagging: {plan.base}")
                tag = tagger.tag(_read_transcript(plan.recording), lang)
                if not tag.is_present():
                    raise PipelineError(f"model produced no tag for {plan.base}")
                sidecar.record_tag(plan.recording, tag, tagger.model_path, lang)
                notify(f"{prefix} {plan.base}: {tag.render()}")
                report.tagged.append(plan.base)
        except Exception as exc:
            notify(f"failed on {plan.base}: {exc}")
            report.failures.append(plan.base)


def matching_files(base_name: str, config: Config) -> list[Path]:
    """Every file in the archive whose name, minus its extension, matches.

    An exact string comparison rather than a glob: recorder names are bracketed
    ("[Contact]_[Phone]_..."), and both shell globs and `find -name` read
    "[1455]" as a character class matching a single one of "1", "4" or "5" -
    which silently matched nothing at all.
    """
    if not config.calls_dir.is_dir():
        raise PipelineError(f"calls directory not found: {config.calls_dir}")
    return sorted(
        path
        for path in config.calls_dir.iterdir()
        if path.is_file() and path.name.rsplit(".", 1)[0] == base_name
    )


def call_tag(base_name: str, config: Config) -> str:
    """The call's topic tag, for a removal preview."""
    path = config.calls_dir / f"{base_name}.json"
    try:
        return Tag.from_json(sidecar.read(path).get("tag")).render()
    except Exception:
        return ""


def trash(path: Path) -> None:
    """Move a file to the macOS trash, so a removal stays recoverable."""
    try:
        subprocess.run(["trash", str(path)], check=True)
    except FileNotFoundError as exc:
        raise PipelineError(
            "the `trash` tool is not installed; use --permanent to delete outright"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"failed to trash {path.name}") from exc


def remove(paths: list[Path], permanent: bool) -> None:
    for path in paths:
        if permanent:
            path.unlink(missing_ok=True)
        else:
            trash(path)

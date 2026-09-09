"""The `call` command: subcommands over one call archive.

Transcription is deliberately not a subcommand here. It stays an external shell
tool (`callscribe`): it is an ffmpeg/whisper-cli/awk pipeline with real
portability workarounds, and it is useful on arbitrary audio with no knowledge
of this archive. `call import` invokes it.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from . import index, pipeline, sidecar
from .config import Config
from .index import Filters, format_duration
from .pipeline import ImportReport

PROGRAM_NAME = "call"

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_MAGENTA = "\033[35m"

# The snippet() call in index.search() marks matches with these control
# characters rather than literal brackets, so they can't collide with
# punctuation that actually appears in a transcript.
_MATCH_START = "\x01"
_MATCH_END = "\x02"


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


_COLOR_MODE = "auto"


def _use_color() -> bool:
    if _COLOR_MODE == "always":
        return True
    if _COLOR_MODE == "never":
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _style(text: str, *codes: str) -> str:
    if not text or not _use_color():
        return text
    return "".join(codes) + text + _RESET


def _render_snippet(snippet: str) -> str:
    text = " ".join(snippet.split())
    highlight = (_BOLD, _YELLOW) if _use_color() else ()
    start = "".join(highlight)
    end = _RESET if highlight else ""
    return text.replace(_MATCH_START, start).replace(_MATCH_END, end)


def _reporter(quiet: bool):
    def notify(message: str) -> None:
        if not quiet:
            _warn(message)

    return notify


_ARROW_STYLE = {"in": (_GREEN, _BOLD), "out": (_MAGENTA, _BOLD)}


def _print_rows(
    rows: list[sqlite3.Row], show_snippet: bool, verbose: bool = False
) -> None:
    if not rows:
        _warn("no matching calls")
        return
    for row in rows:
        arrow = {"in": "<-", "out": "->"}.get(row["direction"], " ?")
        contact = row["contact"] or row["phone_raw"] or "?"
        print(
            f"{_style(row['date'] + ' ' + row['time'][:5], _DIM)} "
            f"{_style(arrow, *_ARROW_STYLE.get(row['direction'], (_DIM,)))} "
            f"{_style(contact, _BOLD, _CYAN)} "
            f"({format_duration(row['duration_seconds'])})  "
            f"{_style(row['tag'], _YELLOW)}  "
            f"{_style('#' + row['hash'], _DIM)}"
        )
        if verbose:
            print(_style(f"    {row['base']}", _DIM))
        if show_snippet and row["snippet"]:
            print(f"    {_render_snippet(row['snippet'])}")


def _rebuild(config: Config, notify) -> None:
    report = index.build(config)
    notify(f"indexed {report.indexed} calls into {config.db_path}")
    if report.without_transcript:
        notify(f"  {report.without_transcript} without a transcript")
    if report.without_tag:
        notify(f"  {report.without_tag} without a topic tag")
    for name in report.skipped:
        _warn(f"{PROGRAM_NAME}: skipped {name}: not a call sidecar")


def _report_import(report: ImportReport, notify) -> int:
    notify(f"copied {len(report.copied)} new recording(s)")
    if report.transcribed:
        notify(f"transcribed {len(report.transcribed)}")
    if report.tagged:
        notify(f"tagged {len(report.tagged)}")
    if report.failures:
        _warn(f"failed ({len(report.failures)}):")
        for name in report.failures:
            _warn(f"  {name}")
        return 1
    return 0


def cmd_import(args: argparse.Namespace, config: Config) -> int:
    notify = _reporter(args.quiet)
    report = ImportReport()

    pipeline.copy_new(config, report, args.dry_run, notify)

    recordings = sidecar.all_recordings(config)
    plans = [
        pipeline.plan_for(recording, config, refresh=not args.dry_run)
        for recording in recordings
    ]
    complete = sum(1 for plan in plans if plan.is_complete)
    notify(f"{complete}/{len(plans)} calls already up to date")

    pipeline.finish(
        plans,
        config,
        args.lang,
        args.model,
        args.dry_run,
        args.quiet,
        notify,
        report,
    )

    if args.dry_run:
        return 0
    if not args.no_index:
        _rebuild(config, notify)
    return _report_import(report, notify)


def cmd_meta(args: argparse.Namespace, config: Config) -> int:
    notify = _reporter(args.quiet)

    recordings = [sidecar.resolve_input(raw, config) for raw in args.inputs]
    if args.all:
        recordings.extend(sidecar.all_recordings(config))
    if not recordings:
        _warn(f"{PROGRAM_NAME} meta: give a recording, or --all")
        return 1

    failures = 0
    for recording in recordings:
        if not recording.is_file():
            _warn(f"{PROGRAM_NAME}: no such recording: {recording}")
            failures += 1
            continue
        path = sidecar.sidecar_path_for(recording)
        existing = sidecar.read(path)
        data = sidecar.refresh(recording, config, args.direction, existing)
        if args.dry_run:
            print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
            continue
        if data == existing:
            notify(f"{recording.stem}: unchanged")
            continue
        sidecar.write(path, data)
        stages = data["stages"]
        notify(
            f"{recording.stem}: scribe={stages['scribe']['status']} "
            f"topic={stages['topic']['status']} wrote {path.name}"
        )
    return 1 if failures else 0


def cmd_topic(args: argparse.Namespace, config: Config) -> int:
    from .topic import Tagger

    notify = _reporter(args.quiet)

    recordings = [sidecar.resolve_input(raw, config) for raw in args.inputs]
    if args.last:
        recordings.extend(sidecar.all_recordings(config)[-1:])
    if args.untagged:
        recordings.extend(
            recording
            for recording in sidecar.all_recordings(config)
            if sidecar.stage_status(
                sidecar.read(sidecar.sidecar_path_for(recording)), "topic"
            )
            != "ok"
        )
    if not recordings:
        _warn(f"{PROGRAM_NAME} topic: give a transcript, or --last / --untagged")
        return 1

    notify(f"loading model: {args.model or config.model_for_lang(args.lang)}")
    tagger = Tagger.load(config, args.lang, args.model)

    failures = 0
    for recording in recordings:
        transcript = recording.with_suffix(".txt")
        if not transcript.is_file():
            _warn(f"{PROGRAM_NAME}: no transcript for {recording.name}")
            failures += 1
            continue
        notify(f"{recording.stem}: generating topic tag")
        tag = tagger.tag(transcript.read_text(encoding="utf-8"), args.lang)
        if not tag.is_present():
            _warn(f"{PROGRAM_NAME}: model produced no output for {transcript.name}")
            failures += 1
            continue
        if args.dry_run:
            print(tag.render())
            continue
        sidecar.record_tag(recording, tag, tagger.model_path, args.lang)
        notify(f"{recording.stem}: {tag.render()}")
    return 1 if failures else 0


def _resolve_hash(raw: str, config: Config) -> list[str]:
    """Base names matching `raw` as a hash prefix, or [] if the index can't say."""
    try:
        return index.resolve_hash(config, raw)
    except index.ArchiveIndexError:
        return []


def _resolve_recording(raw: str, config: Config) -> Path | None:
    """A transcript/sidecar/recording path, or a hash prefix from `call index -l`.

    Shared by translate/summarize, and resolves the same way `call rm`/`call
    show` do: `raw` as a path first, then as a hash prefix. Returns None
    (having already warned) only when `raw` looks like a hash and matches
    more than one call; a path that simply doesn't exist yet is still
    returned as-is, so the caller's own "no transcript for X" error names the
    right call instead of a resolver-internal one.
    """
    resolved = sidecar.resolve_input(raw, config)
    if resolved.is_file():
        return resolved
    matches = _resolve_hash(raw, config)
    if len(matches) > 1:
        _warn(f"{PROGRAM_NAME}: '{raw}' matches multiple calls:")
        for match in matches:
            _warn(f"  {match}")
        return None
    if matches:
        found = sidecar.find_recording(
            config.calls_dir / matches[0], config.audio_suffixes
        )
        if found is not None:
            return found
    return resolved


def _contact_recordings(contact: str, config: Config) -> list[Path]:
    """Every recording matching a --contact filter, resolved to its audio file.

    Shared by translate/summarize (and mirrors `call rm --contact`): relies on
    the SQLite index for the contact/phone matching, so - like `rm --contact`
    - it needs a reasonably fresh `call index` to find everything.
    """
    filters = Filters(contact=contact, country_code=config.country_code)
    matches = index.listing(config, filters, limit=sys.maxsize)
    recordings = []
    for row in matches:
        recording = sidecar.find_recording(
            config.calls_dir / row["base"], config.audio_suffixes
        )
        if recording is not None:
            recordings.append(recording)
    return recordings


def cmd_translate(args: argparse.Namespace, config: Config) -> int:
    from .translate import Translator

    notify = _reporter(args.quiet)

    recordings = []
    for raw in args.inputs:
        resolved = _resolve_recording(raw, config)
        if resolved is None:
            return 1
        recordings.append(resolved)
    if args.last:
        recordings.extend(sidecar.all_recordings(config)[-1:])
    if args.contact:
        matches = _contact_recordings(args.contact, config)
        if not matches:
            _warn(f"{PROGRAM_NAME}: no calls match contact '{args.contact}'")
            return 1
        if args.untranslated:
            matches = [
                recording
                for recording in matches
                if sidecar.stage_status(
                    sidecar.read(sidecar.sidecar_path_for(recording)), "translate"
                )
                != "ok"
            ]
        recordings.extend(matches)
    elif args.untranslated:
        recordings.extend(
            recording
            for recording in sidecar.all_recordings(config)
            if sidecar.stage_status(
                sidecar.read(sidecar.sidecar_path_for(recording)), "translate"
            )
            != "ok"
        )
    if not recordings:
        _warn(
            f"{PROGRAM_NAME} translate: give a transcript, or "
            "--last / --untranslated / --contact"
        )
        return 1
    if args.list:
        for recording in recordings:
            status = sidecar.stage_status(
                sidecar.read(sidecar.sidecar_path_for(recording)), "translate"
            )
            print(f"{recording.stem}  [translate: {status}]")
        return 0

    notify(f"loading model: {args.model or config.model_translate}")
    translator = Translator.load(config, args.model)

    failures = 0
    for recording in recordings:
        transcript = recording.with_suffix(".txt")
        if not transcript.is_file():
            _warn(f"{PROGRAM_NAME}: no transcript for {recording.name}")
            failures += 1
            continue
        notify(f"{recording.stem}: translating")
        text = translator.translate(transcript.read_text(encoding="utf-8"))
        if not text:
            _warn(f"{PROGRAM_NAME}: model produced no output for {transcript.name}")
            failures += 1
            continue
        if args.dry_run:
            print(text)
            continue
        sidecar.record_translation(recording, text, translator.model_path)
        notify(
            f"{recording.stem}: wrote {sidecar.translation_path_for(recording).name}"
        )
    return 1 if failures else 0


def cmd_summarize(args: argparse.Namespace, config: Config) -> int:
    from .summarize import Summarizer

    notify = _reporter(args.quiet)

    recordings = []
    for raw in args.inputs:
        resolved = _resolve_recording(raw, config)
        if resolved is None:
            return 1
        recordings.append(resolved)
    if args.last:
        recordings.extend(sidecar.all_recordings(config)[-1:])
    if args.contact:
        matches = _contact_recordings(args.contact, config)
        if not matches:
            _warn(f"{PROGRAM_NAME}: no calls match contact '{args.contact}'")
            return 1
        if args.unsummarized:
            matches = [
                recording
                for recording in matches
                if sidecar.stage_status(
                    sidecar.read(sidecar.sidecar_path_for(recording)), "summarize"
                )
                != "ok"
            ]
        recordings.extend(matches)
    elif args.unsummarized:
        recordings.extend(
            recording
            for recording in sidecar.all_recordings(config)
            if sidecar.stage_status(
                sidecar.read(sidecar.sidecar_path_for(recording)), "summarize"
            )
            != "ok"
        )
    if not recordings:
        _warn(
            f"{PROGRAM_NAME} summarize: give a transcript, or "
            "--last / --unsummarized / --contact"
        )
        return 1
    if args.list:
        for recording in recordings:
            status = sidecar.stage_status(
                sidecar.read(sidecar.sidecar_path_for(recording)), "summarize"
            )
            print(f"{recording.stem}  [summarize: {status}]")
        return 0

    notify(f"loading model: {args.model or config.model_summarize}")
    summarizer = Summarizer.load(config, args.model)

    failures = 0
    for recording in recordings:
        translation = sidecar.translation_path_for(recording)
        if not translation.is_file():
            _warn(
                f"{PROGRAM_NAME}: no English translation for {recording.name} "
                f"(run `{PROGRAM_NAME} translate` first)"
            )
            failures += 1
            continue
        notify(f"{recording.stem}: summarizing")
        summary = summarizer.summarize(translation.read_text(encoding="utf-8"))
        if not summary:
            _warn(f"{PROGRAM_NAME}: model produced no output for {translation.name}")
            failures += 1
            continue
        if args.dry_run:
            print(summary)
            continue
        sidecar.record_summary(recording, summary, summarizer.model_path)
        notify(f"{recording.stem}: {summary}")
    return 1 if failures else 0


def cmd_index(args: argparse.Namespace, config: Config) -> int:
    global _COLOR_MODE
    _COLOR_MODE = args.color
    notify = _reporter(args.quiet)
    filters = Filters(
        contact=args.contact,
        direction=args.direction,
        since=args.since,
        until=args.until,
        untagged=args.untagged,
        country_code=config.country_code,
    )

    if args.search:
        _print_rows(
            index.search(config, args.search, filters, args.limit),
            show_snippet=True,
            verbose=args.verbose,
        )
        return 0
    if args.list:
        _print_rows(
            index.listing(config, filters, args.limit),
            show_snippet=False,
            verbose=args.verbose,
        )
        return 0
    _rebuild(config, notify)
    return 0


def cmd_rm(args: argparse.Namespace, config: Config) -> int:
    notify = _reporter(args.quiet)

    names = list(args.names)
    if args.contact:
        filters = Filters(contact=args.contact, country_code=config.country_code)
        matches = index.listing(config, filters, limit=sys.maxsize)
        if not matches:
            _warn(f"{PROGRAM_NAME}: no calls match contact '{args.contact}'")
            return 1
        names.extend(row["base"] for row in matches)
    if not names:
        _warn(f"{PROGRAM_NAME}: nothing to remove")
        return 1

    removed = 0
    for raw in names:
        base_name = Path(raw).name.rsplit(".", 1)[0]
        files = pipeline.matching_files(base_name, config)
        if not files:
            matches = _resolve_hash(raw, config)
            if len(matches) > 1:
                _warn(f"{PROGRAM_NAME}: '{raw}' matches multiple calls:")
                for match in matches:
                    _warn(f"  {match}")
                return 1
            if matches:
                base_name = matches[0]
                files = pipeline.matching_files(base_name, config)
        if not files:
            _warn(f"{PROGRAM_NAME}: nothing found for '{base_name}'")
            return 1

        tag = pipeline.call_tag(base_name, config)
        notify(f"{base_name}:")
        if tag:
            notify(f"  topic: {tag}")
        for path in files:
            notify(f"  file:  {path}")

        if args.dry_run:
            continue
        if not args.force:
            reply = input("Remove the above? [y/N] ").strip().lower()
            if reply != "y":
                notify(f"skipped {base_name}")
                continue
        pipeline.remove(files, args.permanent)
        notify(f"removed {base_name}")
        removed += 1

    if removed and not args.no_index:
        _rebuild(config, notify)
    return 0


def cmd_show(args: argparse.Namespace, config: Config) -> int:
    global _COLOR_MODE
    _COLOR_MODE = args.color

    raw = args.name
    base_name = Path(raw).name.rsplit(".", 1)[0]
    if not pipeline.matching_files(base_name, config):
        matches = _resolve_hash(raw, config)
        if len(matches) > 1:
            _warn(f"{PROGRAM_NAME}: '{raw}' matches multiple calls:")
            for match in matches:
                _warn(f"  {match}")
            return 1
        if matches:
            base_name = matches[0]

    transcript = config.calls_dir / f"{base_name}.txt"
    translation = config.calls_dir / f"{base_name}.en.txt"

    # "auto" prefers the English translation - the whole point of translating
    # is to make a call easier to read - but only where one actually exists,
    # so most of the archive (nothing translated yet) is unaffected.
    if args.lang == "en" or (args.lang == "auto" and translation.is_file()):
        path, showing_en = translation, True
    else:
        path, showing_en = transcript, False

    if not path.is_file():
        kind = "translation" if showing_en else "transcript"
        _warn(f"{PROGRAM_NAME}: no {kind} for '{raw}'")
        return 1

    if not args.quiet:
        tag = pipeline.call_tag(base_name, config)
        header = _style(base_name, _BOLD, _CYAN)
        if showing_en:
            header += _style(" [en]", _DIM)
        print(header)
        if tag:
            print(_style(tag, _YELLOW))
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def _add_lang_options(parser: argparse.ArgumentParser, config: Config) -> None:
    parser.add_argument(
        "-l",
        "--lang",
        choices=["he", "en"],
        default=config.lang,
        help="transcript/tag language (default: %(default)s)",
    )
    parser.add_argument(
        "-m", "--model", help="MLX model to use (overrides the language's default)"
    )


def build_parser(config: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Import, transcribe, tag and search a phone call archive.",
        epilog=(
            "Environment:\n"
            f"  CALLS_DIR            Archive directory (default: {config.calls_dir})\n"
            f"  CALLS_PHONEREC_DIR   Recorder folder (default: {config.phonerec_dir})\n"
            f"  CALLS_DB             Index database (default: {config.db_path})\n"
            "  CALLS_SCRIBE         Transcription tool "
            f"(default: {config.scribe_command})\n"
            "  CALLS_MODEL_HE       MLX model for --lang he\n"
            "  CALLS_MODEL_EN       MLX model for --lang en\n"
            "  CALLS_MODEL_TRANSLATE  MLX model for `translate` "
            f"(default: CALLS_MODEL_HE)\n"
            "  CALLS_MODEL_SUMMARIZE  MLX model for `summarize` "
            f"(default: CALLS_MODEL_EN)\n"
            f"  CALLS_LANG           Default language (default: {config.lang})\n"
            f"  CALLS_MAX_TOKENS     Generation budget (default: {config.max_tokens})\n"
            "  CALLS_TRANSLATE_MAX_TOKENS  Translation generation ceiling "
            f"(default: {config.translate_max_tokens})\n"
            "  CALLS_SUMMARIZE_MAX_TOKENS  Summary generation ceiling "
            f"(default: {config.summarize_max_tokens})\n"
            f"  CALLS_COUNTRY_CODE   Country code for local numbers "
            f"(default: {config.country_code})"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress progress messages"
    )

    # Accept -q on either side of the subcommand, since both read naturally.
    # The subparser copy defaults to SUPPRESS rather than False: a plain
    # default would overwrite `call -q index` back to False, which is the
    # standard argparse trap with shared parent options.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="suppress progress messages",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser(
        "import",
        parents=[common],
        help="copy new recordings, then transcribe, tag and index what's outstanding",
        description=(
            "Copy each new recording out of the recorder's incoming/outgoing "
            "folders, then bring every call in the archive up to date. Each "
            "stage is skipped only when the call's sidecar says it already "
            "succeeded, so re-running is safe and a call left half-finished by "
            "an earlier failure is retried rather than abandoned."
        ),
    )
    importer.add_argument(
        "--dry-run", action="store_true", help="show what would be copied and run"
    )
    importer.add_argument(
        "--no-index", action="store_true", help="skip the final index rebuild"
    )
    _add_lang_options(importer, config)
    importer.set_defaults(handler=cmd_import)

    meta = subparsers.add_parser(
        "meta",
        parents=[common],
        help="create or refresh call metadata sidecars",
        description=(
            "Create or refresh the JSON sidecar holding a call's metadata and "
            "pipeline stage status. Safe to re-run: every derived field is "
            "recomputed from the filename, ffprobe and the transcript on disk, "
            "while the topic tag is preserved."
        ),
    )
    meta.add_argument("inputs", nargs="*", help="recording, transcript or sidecar path")
    meta.add_argument(
        "--all", action="store_true", help="refresh every recording in the archive"
    )
    meta.add_argument(
        "--direction",
        choices=["in", "out", "unknown"],
        help="record this direction instead of inferring it",
    )
    meta.add_argument(
        "--dry-run", action="store_true", help="print the sidecar without writing it"
    )
    meta.set_defaults(handler=cmd_meta)

    topic = subparsers.add_parser(
        "topic",
        parents=[common],
        help="generate a topic tag for a transcript",
        description=(
            "Generate a short topic tag (one general category plus one or two "
            "specific subjects) with a local model, and record it in the "
            "call's sidecar."
        ),
    )
    topic.add_argument("inputs", nargs="*", help="transcript or recording path")
    topic.add_argument(
        "--last", action="store_true", help="use the newest recording in the archive"
    )
    topic.add_argument(
        "--untagged", action="store_true", help="tag every call that has no tag yet"
    )
    topic.add_argument(
        "--dry-run", action="store_true", help="print the tag without writing it"
    )
    _add_lang_options(topic, config)
    topic.set_defaults(handler=cmd_topic)

    translator = subparsers.add_parser(
        "translate",
        parents=[common],
        help="translate a transcript to English",
        description=(
            "Translate a Hebrew call transcript to English with a local "
            "model, writing it as '<base>.en.txt' next to the transcript."
        ),
    )
    translator.add_argument(
        "inputs",
        nargs="*",
        help=(
            "transcript or recording path, or the hash id shown by "
            "`call index -l` (a unique prefix of it is enough)"
        ),
    )
    translator.add_argument(
        "--last", action="store_true", help="use the newest recording in the archive"
    )
    translator.add_argument(
        "--untranslated",
        action="store_true",
        help="translate every call with no English translation yet",
    )
    translator.add_argument(
        "--contact",
        help=(
            "translate every call matching this contact name or phone "
            "substring, regardless of whether it's already translated "
            "(combine with --untranslated to only pick up its pending calls)"
        ),
    )
    translator.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "generate the translation - same cost as a real run - and print "
            "it instead of writing it"
        ),
    )
    translator.add_argument(
        "--list",
        action="store_true",
        help=(
            "print which calls would be translated and their current "
            "status, without loading the model"
        ),
    )
    translator.add_argument(
        "-m", "--model", help="MLX model to use (overrides the default)"
    )
    translator.set_defaults(handler=cmd_translate)

    summarizer = subparsers.add_parser(
        "summarize",
        parents=[common],
        help="generate a short English summary of a call",
        description=(
            "Generate a short (a few sentences) English summary of a call "
            "from its translation, and record it in the call's sidecar. "
            "Requires `call translate` to have already produced an English "
            "version."
        ),
    )
    summarizer.add_argument(
        "inputs",
        nargs="*",
        help=(
            "transcript or recording path, or the hash id shown by "
            "`call index -l` (a unique prefix of it is enough)"
        ),
    )
    summarizer.add_argument(
        "--last", action="store_true", help="use the newest recording in the archive"
    )
    summarizer.add_argument(
        "--unsummarized",
        action="store_true",
        help="summarize every call with no summary yet",
    )
    summarizer.add_argument(
        "--contact",
        help=(
            "summarize every call matching this contact name or phone "
            "substring, regardless of whether it's already summarized "
            "(combine with --unsummarized to only pick up its pending calls)"
        ),
    )
    summarizer.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "generate the summary - same cost as a real run - and print it "
            "instead of writing it"
        ),
    )
    summarizer.add_argument(
        "--list",
        action="store_true",
        help=(
            "print which calls would be summarized and their current "
            "status, without loading the model"
        ),
    )
    summarizer.add_argument(
        "-m", "--model", help="MLX model to use (overrides the default)"
    )
    summarizer.set_defaults(handler=cmd_summarize)

    idx = subparsers.add_parser(
        "index",
        parents=[common],
        help="rebuild, search or list the index",
        description=(
            "With no options, rebuilds the SQLite index from the sidecars. "
            "Search is substring-based "
            "(FTS5 trigram), so queries need at least 3 characters and Hebrew "
            "prefixes do not hide matches."
        ),
        epilog=(
            "Examples:\n"
            "  call index                          rebuild\n"
            "  call index -s 'תקלה במערכת'          search the transcripts\n"
            "  call index -l --contact 0525252145  list one contact's calls\n"
            "  call index -l --untagged            find calls with no tag\n"
            "  call index -l --color=always | less -RS\n"
            "                                       page with color, no wrapping"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    idx.add_argument("-s", "--search", help="full-text search across transcripts")
    idx.add_argument(
        "-l", "--list", action="store_true", help="list calls instead of rebuilding"
    )
    idx.add_argument("--contact", help="filter by contact name or phone substring")
    idx.add_argument(
        "--direction", choices=["in", "out", "unknown"], help="filter by direction"
    )
    idx.add_argument("--since", help="only calls on or after YYYY-MM-DD")
    idx.add_argument("--until", help="only calls on or before YYYY-MM-DD")
    idx.add_argument(
        "--untagged", action="store_true", help="only calls with no topic tag"
    )
    idx.add_argument(
        "--limit",
        type=int,
        default=50,
        help="maximum rows to show (default: %(default)s)",
    )
    idx.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show each call's base name",
    )
    idx.add_argument(
        "-C",
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="colorize output (default: %(default)s; use 'always' with less -R)",
    )
    idx.set_defaults(handler=cmd_index)

    remover = subparsers.add_parser(
        "rm",
        parents=[common],
        help="remove a call's recording, transcript and sidecar together",
        description=(
            "Remove a call's recording, transcript, .srt (if any) and sidecar "
            "together, then rebuild the index. Files are moved to the trash "
            "unless --permanent is given."
        ),
    )
    remover.add_argument(
        "names",
        nargs="*",
        help=(
            "call base name, any of its files, or the hash id shown by "
            "`call index -l` (a unique prefix of it is enough)"
        ),
    )
    remover.add_argument(
        "--contact",
        help="remove every call matching this contact name or phone substring",
    )
    remover.add_argument(
        "--permanent", action="store_true", help="delete outright instead of trashing"
    )
    remover.add_argument(
        "-f", "--force", action="store_true", help="do not ask for confirmation"
    )
    remover.add_argument(
        "--dry-run", action="store_true", help="show what would be removed"
    )
    remover.add_argument(
        "--no-index", action="store_true", help="skip the index rebuild"
    )
    remover.set_defaults(handler=cmd_rm)

    shower = subparsers.add_parser(
        "show",
        parents=[common],
        help="print a call's transcript",
        description=(
            "Print the transcript for one call, resolved the same way `call rm` "
            "resolves its argument: a base name, any of its files, or a unique "
            "hash prefix from `call index -l`. Prints the English translation "
            "instead of the Hebrew transcript when one exists, unless --lang "
            "says otherwise."
        ),
    )
    shower.add_argument(
        "name",
        help=(
            "call base name, any of its files, or the hash id shown by "
            "`call index -l` (a unique prefix of it is enough)"
        ),
    )
    shower.add_argument(
        "-l",
        "--lang",
        choices=["auto", "he", "en"],
        default="auto",
        help=(
            "which version to print: auto prefers the English translation "
            "if one exists (default: %(default)s)"
        ),
    )
    shower.add_argument(
        "-C",
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="colorize the header (default: %(default)s)",
    )
    shower.set_defaults(handler=cmd_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    config = Config.from_env()
    parser = build_parser(config)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return int(args.handler(args, config))
    except KeyboardInterrupt:
        _warn("interrupted")
        return 130
    except (
        sidecar.SidecarError,
        index.ArchiveIndexError,
        pipeline.PipelineError,
    ) as exc:
        _warn(f"{PROGRAM_NAME}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

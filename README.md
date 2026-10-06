# calls

A CLI for a phone call recording archive: import recordings off the phone,
transcribe them, tag them by topic with a local model, translate them to
English, summarize them, and search them.

```
call import                          # copy new recordings, then finish what's outstanding
call index -s 'תקלה במערכת'           # full-text search the transcripts
call index -l --contact 0525252145   # list one contact's calls
call index -l --untagged             # find calls that never got tagged
call transcribe a1b2c3d -l en -m MODEL.bin   # retranscribe in another language
call translate --untranslated        # translate every Hebrew call to English
call summarize --unsummarized        # summarize every translated call
call rm a1b2c3d                      # remove by the hash id shown in `call index -l`
call rm --older-than 90 --recording-only   # reclaim space, keep transcripts forever
```

## How the archive is laid out

Two layers, and the split is the whole design:

**Files are the source of truth.** Each call is three files sharing a base name:

```
[Contact]_[Phone]_2026-08-27_14-44-31.opus     the recording
[Contact]_[Phone]_2026-08-27_14-44-31.txt      the transcript
[Contact]_[Phone]_2026-08-27_14-44-31.en.txt   the English translation (optional)
[Contact]_[Phone]_2026-08-27_14-44-31.json     the sidecar
```

The sidecar holds contact, phone (as recorded plus an E.164-normalized form),
direction, duration, channel count, the topic tag, and per-stage pipeline
status. Everything in it except the tag and the direction is derived from
observable facts, so refreshing it is idempotent and self-healing — it can
never claim more than what is actually on disk.

**The index is derived and disposable.** `call index` rebuilds `calls.db`
(SQLite + FTS5) from the sidecars. Delete it and rebuild; nothing is lost.
That is what makes the index safe to change — a new column, a different
tokenizer, a different format — without any risk to the archive itself.

There is deliberately only one index. An earlier `index.md` table was dropped
once `calls.db` existed: it was a third copy of data already held in two
places, rewritten on every build, and the only artifact whose format could
break on an odd tag. `call index -l` scans better than a table did, and this
archive is local-only, so nothing else was ever going to read the file.

## Why it works this way

**Per-stage status, not "does a file exist".** The pipeline used to treat "a
`.txt` exists" as "this call is fully handled". A call whose transcription
succeeded but whose tagging failed was therefore skipped on every later run and
never reached the index — silently, forever. Transcript and tag are now
separate stages, each independently resumable, and every run sweeps the whole
archive rather than only what is still on the phone.

**Tags, not summaries.** The tag is one general category plus one or two
specific subjects, and it is deliberately short. Prose summaries were tried and
rejected: under a "summarize" framing, models this size (DictaLM 1.7B,
Qwen3-4B) ignore length instructions and produce full sentences or markdown
bullet lists. "Topic tag" framing keeps them terse. English output is also
markedly more verbose than Hebrew for the same model. The index exists to be
*scanned* — to find the right recording, and to spot recordings worth deleting
— so the fix for a vague tag is more structure, not a longer field.

**Trigram full-text search.** FTS5's default `unicode61` tokenizer matches
whole tokens, and Hebrew glues prefixes onto words (ה/ו/ב/ל/מ/ש) with no
stemmer available in SQLite — a search for `תקלה` would not find `בתקלה`. The
trigram tokenizer matches substrings instead. The trade-off is that queries
need at least 3 characters.

**Phone numbers are the identity, not names.** The contact name in a filename
comes from the address book at record time, so it drifts; the same person
appeared as both `[דור אקוקה]_[0547602488]` and
`[דור אקוקה]_[+972547602488]`. The sidecar normalizes to E.164 so those
collapse into one identity, and `--contact` normalizes the needle too, so
either form finds every call.

**Direction is captured at import.** The recorder splits calls into
`incoming/` and `outgoing/`, and the flattened filename does not carry that.
Import is the only moment it is knowable, so it is recorded then; calls whose
originals were pruned before this existed are `unknown` and cannot be
recovered.

**Bracketed filenames are matched literally.** Recorder names look like
`[1455]_[1455]_…`, and both shell globs and `find -name` read `[1455]` as a
character class matching a single one of `1`, `4` or `5`. Matching by glob
found nothing at all. Basenames are compared as strings.

## Transcription stays a shell script

`callscribe` (in `~/bin`) is not a subcommand here, on purpose. It is an
ffmpeg/whisper-cli/awk pipeline: it detects whether a recording's two channels
carry genuinely different audio, transcribes each channel separately and merges
them on a shared clock for real speaker attribution, and carries portability
workarounds for macOS `tr` locale behaviour and BSD/GNU `stat`. Reimplementing
that in Python would mean shelling out to the same binaries and rewriting the
awk state machines worse. It is also useful on arbitrary audio, with no
knowledge of this archive. `call import` invokes it; `CALLS_SCRIBE` points
somewhere else if needed.

`call transcribe` re-runs it on existing calls, e.g. one that was transcribed
in the default Hebrew but is spoken in English. `-l` is the spoken language and
`-m` the whisper model path (the default `ivrit-turbo` model is Hebrew-tuned, so
pass an English or multilingual model for `-l en`). The tag, translation and
summary of the old transcript are cleared; `call import` then tags again, and
needs `-l en` too, since tagging otherwise uses `CALLS_LANG`.

## Install

```sh
uv tool install ~/Code/personal/calls     # puts `call` on PATH
uv tool install --force --reinstall .     # after making changes
```

Requires `ffprobe` (duration/channels), `callscribe` (transcription) and
`trash` (recoverable removal; `call rm --permanent` skips it).

## Configuration

Every path and model is an environment variable:

| Variable | Default |
|---|---|
| `CALLS_DIR` | `~/Recordings/calls` |
| `CALLS_PHONEREC_DIR` | `~/Recordings/PhoneRec` |
| `CALLS_DB` | `$CALLS_DIR/calls.db` |
| `CALLS_SCRIBE` | `callscribe` |
| `CALLS_MODEL_HE` | `~/.local/share/mlx/DictaLM-3.0-1.7B-Instruct-bf16` |
| `CALLS_MODEL_EN` | `mlx-community/Qwen3-4B-4bit-DWQ-053125` |
| `CALLS_MODEL_TRANSLATE` | same as `CALLS_MODEL_HE` |
| `CALLS_MODEL_SUMMARIZE` | same as `CALLS_MODEL_EN` |
| `CALLS_LANG` | `he` |
| `CALLS_MAX_TOKENS` | `300` |
| `CALLS_TRANSLATE_MAX_TOKENS` | `8000` |
| `CALLS_SUMMARIZE_MAX_TOKENS` | `500` |
| `CALLS_COUNTRY_CODE` | `972` |

Use `-l en` for mostly-English calls: it transcribes them in English (with the
multilingual whisper model) and tags them with Qwen3-4B, since DictaLM is
Hebrew-tuned and gives weaker, more generic tags on English-heavy content.

`call import` takes optional call ids (or paths) to act on only those calls,
copying nothing new from the recorder, so a mixed archive can be done in two
passes:

```
call import <english ids...> -l en
call import -l he                    # everything else
```

`call translate` is a separate command, not a stage of `call import`: unlike
transcription and tagging, not every call needs an English copy, so it isn't
run automatically. DictaLM is the default translator too - in testing it was
more faithful to specifics (foreign loanwords, mid-sentence topic shifts) than
Qwen3-4B, despite being the smaller model.

`call show` (and anything built on it, like an fzf browser) prints the English
translation instead of the Hebrew transcript whenever one exists - `-l he`
forces the original. This only changes behaviour for calls translated on
purpose, since nothing is translated automatically.

`call summarize` requires `call translate` to have already run: it summarizes
the English translation, not the Hebrew transcript, and needs Qwen3-4B rather
than DictaLM to do it. A head-to-head across all four combinations of
{Hebrew, English} x {DictaLM, Qwen3-4B} found the other three each broken in
a different way - Qwen3-4B writing Hebrew directly mixed in stray Chinese
characters mid-sentence, and DictaLM summarizing its own English translation
stayed accurate on short calls but hallucinated plausible-sounding fake
entity names once a call got long and technical. Only Qwen3-4B on the English
translation held up across a short, a dense, and a long call.

## Development

```sh
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check
uv run -m pytest
```

## Not done yet

Filenames are still the recorder's, in one flat directory. Renaming to a
sortable ASCII scheme under `YYYY/MM/` was considered and deliberately
deferred — the sidecar already carries stable identity, so the layout change
can happen whenever, independently.

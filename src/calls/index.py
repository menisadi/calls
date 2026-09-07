"""The SQLite index over the archive: build, search, list, render.

The index is entirely derived from the JSON sidecars plus the transcript text,
so it is disposable: delete the database and rebuild it, and nothing is lost.
That is the whole point of keeping the files as the source of truth - the index
can be changed, re-tokenized or thrown away without ever risking the archive.

Full-text search uses FTS5 with the trigram tokenizer rather than the default
unicode61. Hebrew glues prefixes onto words (ה/ו/ב/ל/מ/ש) and has no stemmer in
SQLite, so whole-token matching misses far too much: a search for "תקלה" would
not find "בתקלה". Trigram indexing matches substrings, at the cost of needing
at least 3 characters per query term.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_COUNTRY_CODE, Config
from .sidecar import Tag, normalize_phone, read

# Trigram indexing cannot answer a query shorter than one trigram.
MIN_SEARCH_LENGTH = 3

SCHEMA = """
CREATE TABLE calls (
    base             TEXT PRIMARY KEY,
    contact          TEXT NOT NULL DEFAULT '',
    phone            TEXT,
    phone_raw        TEXT,
    direction        TEXT NOT NULL DEFAULT 'unknown',
    started_at       TEXT NOT NULL DEFAULT '',
    date             TEXT NOT NULL DEFAULT '',
    time             TEXT NOT NULL DEFAULT '',
    duration_seconds REAL,
    channels         INTEGER,
    tag              TEXT NOT NULL DEFAULT '',
    tag_general      TEXT NOT NULL DEFAULT '',
    tag_specific     TEXT NOT NULL DEFAULT '[]',
    scribe_status    TEXT NOT NULL DEFAULT 'missing',
    scribe_words     INTEGER,
    topic_status     TEXT NOT NULL DEFAULT 'missing',
    recording        TEXT,
    transcript       TEXT,
    path             TEXT NOT NULL
);
CREATE INDEX calls_date ON calls(date DESC);
CREATE INDEX calls_phone ON calls(phone);
CREATE INDEX calls_contact ON calls(contact);

CREATE VIRTUAL TABLE search USING fts5(
    base UNINDEXED,
    contact,
    tag,
    body,
    tokenize='trigram'
);
"""

INSERT_CALL = """
INSERT INTO calls (
    base, contact, phone, phone_raw, direction, started_at, date, time,
    duration_seconds, channels, tag, tag_general, tag_specific,
    scribe_status, scribe_words, topic_status, recording, transcript, path
) VALUES (
    :base, :contact, :phone, :phone_raw, :direction, :started_at, :date, :time,
    :duration_seconds, :channels, :tag, :tag_general, :tag_specific,
    :scribe_status, :scribe_words, :topic_status, :recording, :transcript, :path
)
"""


class ArchiveIndexError(Exception):
    """The index cannot be built or read."""


@dataclass
class Filters:
    """Row filters shared by search and listing."""

    contact: str | None = None
    direction: str | None = None
    since: str | None = None
    until: str | None = None
    untagged: bool = False
    country_code: str = DEFAULT_COUNTRY_CODE

    def clauses(self) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.contact:
            # Match the needle as typed against the name and both stored phone
            # forms, and also normalize it and match that. Without the last
            # part, searching the dialled "0547602488" misses every call the
            # recorder happened to name "+972547602488" - which is exactly the
            # split that normalizing was introduced to heal.
            conditions = [
                "calls.contact LIKE ?",
                "calls.phone LIKE ?",
                "calls.phone_raw LIKE ?",
            ]
            params.extend([f"%{self.contact}%"] * 3)
            normalized = normalize_phone(self.contact, self.country_code)
            if normalized:
                conditions.append("calls.phone = ?")
                params.append(normalized)
            clauses.append(f"({' OR '.join(conditions)})")
        if self.direction:
            clauses.append("calls.direction = ?")
            params.append(self.direction)
        if self.since:
            clauses.append("calls.date >= ?")
            params.append(self.since)
        if self.until:
            clauses.append("calls.date <= ?")
            params.append(self.until)
        if self.untagged:
            clauses.append("calls.topic_status != 'ok'")
        return clauses, params


@dataclass
class BuildReport:
    indexed: int
    without_transcript: int
    without_tag: int
    skipped: list[str]


def _row_from_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    """One index row, or None if the file isn't a usable call sidecar."""
    try:
        data = read(sidecar_path)
    except Exception:
        return None
    if not data.get("base"):
        return None

    stages = data.get("stages") or {}
    scribe = stages.get("scribe") or {}
    topic = stages.get("topic") or {}
    tag = Tag.from_json(data.get("tag"))

    transcript_name = data.get("transcript")
    body = ""
    if transcript_name:
        transcript_path = sidecar_path.with_name(str(transcript_name))
        if transcript_path.is_file():
            body = transcript_path.read_text(encoding="utf-8")

    return {
        "base": data["base"],
        "contact": data.get("contact") or "",
        "phone": data.get("phone"),
        "phone_raw": data.get("phone_raw"),
        "direction": data.get("direction") or "unknown",
        "started_at": data.get("started_at") or "",
        "date": data.get("date") or "",
        "time": data.get("time") or "",
        "duration_seconds": data.get("duration_seconds"),
        "channels": data.get("channels"),
        "tag": tag.render(),
        "tag_general": tag.general,
        "tag_specific": json.dumps(tag.specific, ensure_ascii=False),
        "scribe_status": scribe.get("status") or "missing",
        "scribe_words": scribe.get("words"),
        "topic_status": topic.get("status") or "missing",
        "recording": data.get("recording"),
        "transcript": transcript_name,
        "path": str(sidecar_path.parent),
        "body": body,
    }


def build(config: Config) -> BuildReport:
    """Rebuild the whole index from scratch.

    A full rebuild rather than an incremental sync: the archive is small enough
    that this takes well under a second, and it removes any chance of the index
    drifting from the sidecars (a deleted call leaving a stale row, a re-tagged
    call keeping its old tag). The database is assembled beside the real one and
    moved into place, so a failure part-way through leaves the old index intact.
    """
    if not config.calls_dir.is_dir():
        raise ArchiveIndexError(f"calls directory not found: {config.calls_dir}")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for sidecar_path in sorted(config.calls_dir.rglob("*.json")):
        row = _row_from_sidecar(sidecar_path)
        if row is None:
            skipped.append(sidecar_path.name)
        else:
            rows.append(row)

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config.db_path.with_name(f".{config.db_path.name}.tmp")
    temp_path.unlink(missing_ok=True)

    try:
        connection = sqlite3.connect(temp_path)
        try:
            with connection:
                connection.executescript(SCHEMA)
                connection.executemany(INSERT_CALL, rows)
                connection.executemany(
                    "INSERT INTO search (base, contact, tag, body) "
                    "VALUES (:base, :contact, :tag, :body)",
                    rows,
                )
        finally:
            connection.close()
        os.replace(temp_path, config.db_path)
    except (sqlite3.Error, OSError) as exc:
        # Surface one clean message instead of a traceback, and don't leave a
        # half-built database lying next to the real one. The existing index is
        # untouched either way, since it is only replaced on success.
        temp_path.unlink(missing_ok=True)
        raise ArchiveIndexError(
            f"could not build index at {config.db_path}: {exc}"
        ) from exc

    return BuildReport(
        indexed=len(rows),
        without_transcript=sum(1 for row in rows if row["scribe_status"] != "ok"),
        without_tag=sum(1 for row in rows if row["topic_status"] != "ok"),
        skipped=skipped,
    )


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise ArchiveIndexError(f"index not found: {db_path} (run `call index` first)")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def as_fts_query(query: str) -> str:
    """Wrap a raw query as a single FTS5 phrase.

    Users type Hebrew phrases with punctuation, apostrophes and quotes, all of
    which are FTS5 query operators. Quoting the whole thing turns it into a
    literal phrase search, which is what "search my calls for X" means anyway.
    """
    return '"' + query.replace('"', '""') + '"'


def search(
    config: Config, query: str, filters: Filters, limit: int = 50
) -> list[sqlite3.Row]:
    if len(query.strip()) < MIN_SEARCH_LENGTH:
        raise ArchiveIndexError(
            f"trigram search needs at least {MIN_SEARCH_LENGTH} characters"
        )

    clauses, params = filters.clauses()
    where = " AND ".join(["search MATCH ?"] + clauses)
    connection = connect(config.db_path)
    try:
        return connection.execute(
            f"""
            SELECT calls.*, snippet(search, 3, '[', ']', ' … ', 16) AS snippet
            FROM search
            JOIN calls ON calls.base = search.base
            WHERE {where}
            ORDER BY bm25(search), calls.date DESC
            LIMIT ?
            """,
            [as_fts_query(query), *params, limit],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise ArchiveIndexError(f"search failed: {exc}") from exc
    finally:
        connection.close()


def listing(config: Config, filters: Filters, limit: int = 50) -> list[sqlite3.Row]:
    clauses, params = filters.clauses()
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    connection = connect(config.db_path)
    try:
        return connection.execute(
            f"SELECT calls.*, NULL AS snippet FROM calls {where} "
            "ORDER BY calls.date DESC, calls.time DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
    finally:
        connection.close()


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "--:--"
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"

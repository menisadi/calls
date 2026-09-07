from __future__ import annotations

import pytest

from calls import index, sidecar
from calls.config import Config
from calls.index import ArchiveIndexError, Filters, format_duration
from calls.sidecar import Tag


@pytest.fixture
def archive(config: Config, make_call):
    """Three indexed calls, one of them untagged."""

    def _add(base: str, transcript: str, tag: Tag | None, direction: str) -> None:
        recording = make_call(base, transcript=transcript)
        data = sidecar.refresh(recording, config, direction=direction)
        if tag is not None:
            data["tag"] = tag.to_json()
            data["stages"]["topic"]["status"] = "ok"
        sidecar.write(sidecar.sidecar_path_for(recording), data)

    _add(
        "[דור אקוקה]_[0547602488]_2026-08-19_11-48-33",
        "היי מני, יש לי בתקלה חשמלית בבית והתנור לא עובד",
        Tag(general="תקלה", specific=["חשמל"]),
        "in",
    )
    _add(
        "[Ariel Hanemann]_[0525252145]_2026-08-26_11-07-18",
        "the CPM numbers look wrong in the dashboard",
        Tag(general="debugging", specific=["CPM dashboard"]),
        "out",
    )
    _add(
        "[1455]_[1455]_2026-08-30_20-22-21",
        "שעון מדויק",
        None,
        "in",
    )
    return config


class TestBuild:
    def test_indexes_every_sidecar(self, archive: Config):
        report = index.build(archive)
        assert report.indexed == 3
        assert report.without_tag == 1
        assert report.without_transcript == 0
        assert report.skipped == []

    def test_is_a_full_rebuild_so_deletions_disappear(self, archive: Config, make_call):
        index.build(archive)
        assert len(index.listing(archive, Filters(), limit=50)) == 3

        for path in archive.calls_dir.glob("[[]1455[]]*"):
            path.unlink()
        index.build(archive)
        # No stale row: the index cannot outlive the sidecar it came from.
        assert len(index.listing(archive, Filters(), limit=50)) == 2

    def test_skips_a_json_file_that_is_not_a_sidecar(self, archive: Config):
        (archive.calls_dir / "notes.json").write_text('{"hello": 1}', encoding="utf-8")
        report = index.build(archive)
        assert report.indexed == 3
        assert report.skipped == ["notes.json"]

    def test_skips_a_malformed_sidecar_without_failing(self, archive: Config):
        (archive.calls_dir / "broken.json").write_text("{oh no", encoding="utf-8")
        report = index.build(archive)
        assert report.indexed == 3
        assert report.skipped == ["broken.json"]

    def test_a_missing_archive_raises(self, config: Config):
        config.calls_dir.rmdir()
        with pytest.raises(ArchiveIndexError):
            index.build(config)

    def test_leaves_the_old_index_intact_on_failure(self, archive: Config):
        index.build(archive)
        before = archive.db_path.read_bytes()

        archive.calls_dir.joinpath("x.json").write_text("{}", encoding="utf-8")
        archive.calls_dir.chmod(0o500)  # no new temp file can be created
        try:
            with pytest.raises(ArchiveIndexError, match="could not build index"):
                index.build(archive)
        finally:
            archive.calls_dir.chmod(0o700)
        assert archive.db_path.read_bytes() == before


class TestSearch:
    def test_matches_across_a_glued_hebrew_prefix(self, archive: Config):
        index.build(archive)
        # The transcript says "בתקלה"; trigram indexing still finds "תקלה".
        # A token-based tokenizer would miss this, which is why it is used.
        rows = index.search(archive, "תקלה", Filters(), limit=10)
        assert [row["base"] for row in rows] == [
            "[דור אקוקה]_[0547602488]_2026-08-19_11-48-33"
        ]

    def test_matches_latin_text_in_a_hebrew_archive(self, archive: Config):
        index.build(archive)
        rows = index.search(archive, "CPM", Filters(), limit=10)
        assert rows[0]["base"] == "[Ariel Hanemann]_[0525252145]_2026-08-26_11-07-18"

    def test_returns_a_snippet_marking_the_match(self, archive: Config):
        index.build(archive)
        rows = index.search(archive, "תנור", Filters(), limit=10)
        assert "[תנור]" in rows[0]["snippet"]

    def test_too_short_a_query_is_refused(self, archive: Config):
        index.build(archive)
        # Honest failure rather than silently returning nothing.
        with pytest.raises(ArchiveIndexError, match="at least 3"):
            index.search(archive, "תק", Filters(), limit=10)

    @pytest.mark.parametrize(
        "query", ['say "hi"', "it's fine", "a OR b", "x*y", "NEAR(a b)", "-nope"]
    )
    def test_query_punctuation_does_not_break_fts_syntax(
        self, archive: Config, query: str
    ):
        index.build(archive)
        # These are all FTS5 operators; quoting makes them a literal phrase.
        assert index.search(archive, query, Filters(), limit=10) == []

    def test_honours_filters(self, archive: Config):
        index.build(archive)
        assert index.search(archive, "תנור", Filters(direction="out"), limit=10) == []
        assert index.search(archive, "תנור", Filters(direction="in"), limit=10)

    def test_a_missing_index_raises(self, archive: Config):
        with pytest.raises(ArchiveIndexError, match="index not found"):
            index.search(archive, "תקלה", Filters(), limit=10)


class TestListing:
    def test_newest_first(self, archive: Config):
        index.build(archive)
        rows = index.listing(archive, Filters(), limit=50)
        assert [row["date"] for row in rows] == [
            "2026-08-30",
            "2026-08-26",
            "2026-08-19",
        ]

    def test_untagged_filter_finds_the_stuck_call(self, archive: Config):
        index.build(archive)
        rows = index.listing(archive, Filters(untagged=True), limit=50)
        assert [row["base"] for row in rows] == ["[1455]_[1455]_2026-08-30_20-22-21"]

    def test_contact_filter_matches_either_phone_form(self, archive: Config, make_call):
        # The same contact, recorded under both phone forms - which is the
        # real situation this filter has to see through.
        other = make_call(
            "[דור אקוקה]_[+972547602488]_2026-08-27_14-44-31", transcript="עוד שיחה"
        )
        sidecar.write(sidecar.sidecar_path_for(other), sidecar.refresh(other, archive))
        index.build(archive)

        for needle in ("0547602488", "+972547602488", "547602488", "אקוקה"):
            rows = index.listing(archive, Filters(contact=needle), limit=50)
            # Both calls, whichever form was typed. An earlier version only
            # matched the stored strings, so the dialled form found just one.
            assert len(rows) == 2, needle

    def test_date_range_filters(self, archive: Config):
        index.build(archive)
        rows = index.listing(
            archive, Filters(since="2026-08-20", until="2026-08-27"), limit=50
        )
        assert [row["date"] for row in rows] == ["2026-08-26"]

    def test_limit_is_applied(self, archive: Config):
        index.build(archive)
        assert len(index.listing(archive, Filters(), limit=2)) == 2


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, "--:--"),
            (0, "--:--"),
            (17.4, "00:17"),
            (264.2, "04:24"),
            (908.3, "15:08"),
            (3671, "61:11"),
        ],
    )
    def test_formats(self, seconds: float | None, expected: str):
        assert format_duration(seconds) == expected

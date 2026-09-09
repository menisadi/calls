from __future__ import annotations

import json
from pathlib import Path

import pytest

from calls import sidecar
from calls.config import Config
from calls.sidecar import CallName, SidecarError, Tag, normalize_phone

HEBREW_BASE = "[דור אקוקה]_[0547602488]_2026-08-19_11-48-33"


class TestCallName:
    def test_parses_a_recorder_name(self):
        parsed = CallName.parse(HEBREW_BASE)
        assert parsed is not None
        assert parsed.contact == "דור אקוקה"
        assert parsed.phone_raw == "0547602488"
        assert parsed.date == "2026-08-19"
        assert parsed.time == "11:48:33"

    def test_parses_a_non_numeric_phone_field(self):
        # The recorder writes whatever it has; "[שופרסל]_[Shufersal]" is real.
        parsed = CallName.parse("[שופרסל]_[Shufersal]_2026-09-03_20-35-15")
        assert parsed is not None
        assert parsed.phone_raw == "Shufersal"

    def test_parses_an_empty_contact_or_phone(self):
        parsed = CallName.parse("[]_[]_2026-09-03_20-35-15")
        assert parsed is not None
        assert parsed.contact == ""
        assert parsed.phone_raw == ""

    @pytest.mark.parametrize(
        "base",
        [
            "not-a-call",
            "[Contact]_[0501234567]_2026-13-01_10-00-00",  # bad month
            "[Contact]_[0501234567]_2026-09-03",  # no time
            "meet_group_call_2026-08-25_14-59-48",
        ],
    )
    def test_rejects_other_names(self, base: str):
        assert CallName.parse(base) is None


class TestNormalizePhone:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The one ambiguity this exists for: same contact, two forms.
            ("0547602488", "+972547602488"),
            ("+972547602488", "+972547602488"),
            ("072-252-0100", "+972722520100"),
            ("00972547602488", "+972547602488"),
            ("(054) 760 2488", "+972547602488"),
            ("972547602488", "+972547602488"),
            # Short codes have no country context; a prefix would be nonsense.
            ("1455", "1455"),
            ("*3555", None),
            ("Shufersal", None),
            ("null", None),
            ("", None),
            ("unknown", None),
            ("+not-a-number", None),
        ],
    )
    def test_normalizes(self, raw: str, expected: str | None):
        assert normalize_phone(raw, "972") == expected

    def test_honours_a_different_country_code(self):
        assert normalize_phone("0501234567", "44") == "+44501234567"

    def test_collapses_the_two_forms_of_one_number(self):
        assert normalize_phone("0547602488", "972") == normalize_phone(
            "+972547602488", "972"
        )


class TestTag:
    def test_renders_general_and_specific(self):
        tag = Tag(general="פגישת עבודה", specific=["תכנון משימות", "באגים"])
        assert tag.render() == "פגישת עבודה, תכנון משימות, באגים"

    def test_skips_blank_parts(self):
        assert Tag(general="", specific=["", "דיבוג"]).render() == "דיבוג"

    def test_absent_when_empty(self):
        assert not Tag().is_present()
        assert Tag(general="x").is_present()
        assert Tag(specific=["x"]).is_present()

    def test_round_trips_through_json(self):
        tag = Tag(general="a", specific=["b"], source="index.md-backfill")
        assert Tag.from_json(tag.to_json()) == tag

    def test_tolerates_a_non_list_specific(self):
        assert Tag.from_json({"general": "a", "specific": "b"}).specific == ["b"]

    def test_tolerates_junk(self):
        assert Tag.from_json(None) == Tag()
        assert Tag.from_json("nonsense") == Tag()


class TestReadWrite:
    def test_missing_sidecar_reads_as_empty(self, tmp_path: Path):
        assert sidecar.read(tmp_path / "nope.json") == {}

    def test_malformed_sidecar_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SidecarError):
            sidecar.read(path)

    def test_non_object_sidecar_raises(self, tmp_path: Path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(SidecarError):
            sidecar.read(path)

    def test_write_preserves_hebrew_unescaped(self, tmp_path: Path):
        path = tmp_path / "x.json"
        sidecar.write(path, {"contact": "דור אקוקה"})
        assert "דור אקוקה" in path.read_text(encoding="utf-8")
        assert sidecar.read(path) == {"contact": "דור אקוקה"}

    def test_write_leaves_no_temp_files_behind(self, tmp_path: Path):
        sidecar.write(tmp_path / "x.json", {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


class TestStageStatus:
    @pytest.mark.parametrize(
        "data",
        [
            {},
            {"stages": None},
            {"stages": "nonsense"},
            {"stages": {}},
            {"stages": {"scribe": None}},
            {"stages": {"scribe": {}}},
        ],
    )
    def test_anything_unusable_means_not_done(self, data: dict):
        # "Not done" is the safe answer: the stage simply runs again.
        assert sidecar.stage_status(data, "scribe") == "missing"

    def test_reads_a_recorded_status(self):
        data = {"stages": {"scribe": {"status": "ok"}}}
        assert sidecar.stage_status(data, "scribe") == "ok"


class TestRefresh:
    def test_derives_metadata_from_the_filename(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="שלום שלום")
        data = sidecar.refresh(recording, config)

        assert data["schema"] == sidecar.SCHEMA_VERSION
        assert data["base"] == HEBREW_BASE
        assert data["contact"] == "דור אקוקה"
        assert data["phone_raw"] == "0547602488"
        assert data["phone"] == "+972547602488"
        assert data["date"] == "2026-08-19"
        assert data["started_at"] == "2026-08-19T11:48:33"

    def test_transcript_present_makes_the_scribe_stage_ok(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="one two three")
        data = sidecar.refresh(recording, config)
        assert data["stages"]["scribe"]["status"] == "ok"
        assert data["stages"]["scribe"]["words"] == 3
        assert data["transcript"] == f"{HEBREW_BASE}.txt"

    def test_no_transcript_leaves_the_scribe_stage_missing(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE)
        data = sidecar.refresh(recording, config)
        assert data["stages"]["scribe"]["status"] == "missing"
        assert "words" not in data["stages"]["scribe"]
        assert data["transcript"] is None

    def test_a_whitespace_only_transcript_is_empty_not_ok(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="   \n  ")
        data = sidecar.refresh(recording, config)
        assert data["stages"]["scribe"]["status"] == "empty"

    def test_a_transcript_that_disappears_downgrades_the_stage(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        first = sidecar.refresh(recording, config)
        assert first["stages"]["scribe"]["status"] == "ok"

        recording.with_suffix(".txt").unlink()
        second = sidecar.refresh(recording, config, existing=first)
        # Self-healing: the sidecar can never claim more than what's on disk.
        assert second["stages"]["scribe"]["status"] == "missing"

    def test_topic_stage_follows_whether_a_tag_is_present(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        without = sidecar.refresh(recording, config)
        assert without["stages"]["topic"]["status"] == "missing"

        with_tag = sidecar.refresh(
            recording, config, existing={"tag": {"general": "דיבוג", "specific": []}}
        )
        assert with_tag["stages"]["topic"]["status"] == "ok"

    def test_translation_present_makes_the_translate_stage_ok(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        sidecar.translation_path_for(recording).write_text(
            "one two three", encoding="utf-8"
        )
        data = sidecar.refresh(recording, config)
        assert data["stages"]["translate"]["status"] == "ok"
        assert data["stages"]["translate"]["words"] == 3
        assert data["transcript_en"] == f"{HEBREW_BASE}.en.txt"

    def test_no_translation_leaves_the_translate_stage_missing(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        data = sidecar.refresh(recording, config)
        assert data["stages"]["translate"]["status"] == "missing"
        assert "words" not in data["stages"]["translate"]
        assert data["transcript_en"] is None

    def test_a_translation_that_disappears_downgrades_the_stage(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        sidecar.translation_path_for(recording).write_text("hi", encoding="utf-8")
        first = sidecar.refresh(recording, config)
        assert first["stages"]["translate"]["status"] == "ok"

        sidecar.translation_path_for(recording).unlink()
        second = sidecar.refresh(recording, config, existing=first)
        assert second["stages"]["translate"]["status"] == "missing"

    def test_summarize_stage_follows_whether_a_summary_is_present(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        without = sidecar.refresh(recording, config)
        assert without["stages"]["summarize"]["status"] == "missing"

        with_summary = sidecar.refresh(
            recording, config, existing={"summary": "A short call about billing."}
        )
        assert with_summary["stages"]["summarize"]["status"] == "ok"

    def test_preserves_the_tag_and_unknown_keys(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="text")
        existing = {
            "tag": {"general": "דיבוג", "specific": ["באג"]},
            "notes": "something a future version added",
        }
        data = sidecar.refresh(recording, config, existing=existing)
        assert data["tag"] == existing["tag"]
        assert data["notes"] == existing["notes"]

    def test_is_idempotent(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="text")
        first = sidecar.refresh(recording, config)
        second = sidecar.refresh(recording, config, existing=first)
        assert first == second

    def test_probe_degrades_when_the_audio_is_unreadable(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        data = sidecar.refresh(recording, config)
        assert data["duration_seconds"] is None
        assert data["channels"] is None

    def test_unparseable_name_falls_back_to_mtime(self, config: Config, make_call):
        recording = make_call("some_other_recording", transcript="text")
        data = sidecar.refresh(recording, config)
        assert data["contact"] == ""
        assert data["phone"] is None
        assert data["date"]  # from the file's own mtime, not blank


class TestDirection:
    def test_inferred_from_the_recorder_folder(
        self, config: Config, make_call, make_original
    ):
        recording = make_call(HEBREW_BASE)
        make_original(HEBREW_BASE, "outgoing")
        assert sidecar.refresh(recording, config)["direction"] == "out"

    def test_inferred_from_an_outgoing_contact_name(self, config: Config, make_call):
        recording = make_call("[Outgoing]_[null]_2026-08-13_12-02-30")
        assert sidecar.refresh(recording, config)["direction"] == "out"

    def test_unknown_when_the_original_is_gone(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE)
        assert sidecar.refresh(recording, config)["direction"] == "unknown"

    def test_explicit_direction_wins(self, config: Config, make_call, make_original):
        recording = make_call(HEBREW_BASE)
        make_original(HEBREW_BASE, "incoming")
        data = sidecar.refresh(recording, config, direction="out")
        assert data["direction"] == "out"

    def test_a_known_direction_is_never_downgraded(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE)
        # The original has since been pruned from the recorder's folder, so it
        # can no longer be inferred - but it was recorded at import time.
        data = sidecar.refresh(recording, config, existing={"direction": "in"})
        assert data["direction"] == "in"


class TestRecordTag:
    def test_writes_the_tag_and_its_provenance(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="text")
        sidecar.write(
            sidecar.sidecar_path_for(recording), sidecar.refresh(recording, config)
        )

        tag = Tag(general="תקלה", specific=["חשמל"])
        data = sidecar.record_tag(recording, tag, "model-he", "he")

        assert data["tag"] == {"general": "תקלה", "specific": ["חשמל"]}
        assert data["stages"]["topic"]["status"] == "ok"
        assert data["stages"]["topic"]["model"] == "model-he"
        assert data["stages"]["topic"]["lang"] == "he"

    def test_does_not_clobber_fields_owned_by_refresh(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="one two")
        refreshed = sidecar.refresh(recording, config, direction="in")
        sidecar.write(sidecar.sidecar_path_for(recording), refreshed)

        data = sidecar.record_tag(recording, Tag(general="x"), "model-he", "he")

        assert data["direction"] == "in"
        assert data["phone"] == "+972547602488"
        assert data["stages"]["scribe"]["words"] == 2

    def test_creates_a_minimal_sidecar_when_none_exists(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        data = sidecar.record_tag(recording, Tag(general="x"), "model-he", "he")
        # Enough for the index to place the call; refresh fills in the rest.
        assert data["base"] == HEBREW_BASE
        assert data["contact"] == "דור אקוקה"
        assert data["date"] == "2026-08-19"

    def test_a_later_refresh_keeps_the_tag(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="text")
        sidecar.record_tag(recording, Tag(general="תקלה"), "model-he", "he")
        path = sidecar.sidecar_path_for(recording)

        data = sidecar.refresh(recording, config)
        sidecar.write(path, data)

        assert json.loads(path.read_text(encoding="utf-8"))["tag"]["general"] == "תקלה"
        assert data["stages"]["topic"]["status"] == "ok"


class TestRecordTranslation:
    def test_writes_the_translation_and_its_provenance(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="שלום")
        sidecar.write(
            sidecar.sidecar_path_for(recording), sidecar.refresh(recording, config)
        )

        data = sidecar.record_translation(recording, "Hello", "model-translate")

        translation = sidecar.translation_path_for(recording)
        assert translation.read_text(encoding="utf-8") == "Hello"
        assert data["transcript_en"] == translation.name
        assert data["stages"]["translate"]["status"] == "ok"
        assert data["stages"]["translate"]["model"] == "model-translate"

    def test_does_not_clobber_fields_owned_by_refresh(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="one two")
        refreshed = sidecar.refresh(recording, config, direction="in")
        sidecar.write(sidecar.sidecar_path_for(recording), refreshed)

        data = sidecar.record_translation(recording, "one two", "model-translate")

        assert data["direction"] == "in"
        assert data["phone"] == "+972547602488"
        assert data["stages"]["scribe"]["words"] == 2

    def test_creates_a_minimal_sidecar_when_none_exists(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        data = sidecar.record_translation(recording, "text", "model-translate")
        assert data["base"] == HEBREW_BASE
        assert data["contact"] == "דור אקוקה"
        assert data["date"] == "2026-08-19"

    def test_a_later_refresh_keeps_the_model_provenance(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        sidecar.record_translation(recording, "text", "model-translate")
        path = sidecar.sidecar_path_for(recording)

        data = sidecar.refresh(recording, config)
        sidecar.write(path, data)

        assert data["stages"]["translate"]["status"] == "ok"
        assert data["stages"]["translate"]["model"] == "model-translate"


class TestRecordSummary:
    def test_writes_the_summary_and_its_provenance(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="שלום")
        sidecar.write(
            sidecar.sidecar_path_for(recording), sidecar.refresh(recording, config)
        )

        data = sidecar.record_summary(recording, "A short call.", "model-summarize")

        assert data["summary"] == "A short call."
        assert data["stages"]["summarize"]["status"] == "ok"
        assert data["stages"]["summarize"]["model"] == "model-summarize"

    def test_does_not_clobber_fields_owned_by_refresh(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="one two")
        refreshed = sidecar.refresh(recording, config, direction="in")
        sidecar.write(sidecar.sidecar_path_for(recording), refreshed)

        data = sidecar.record_summary(recording, "summary", "model-summarize")

        assert data["direction"] == "in"
        assert data["phone"] == "+972547602488"
        assert data["stages"]["scribe"]["words"] == 2

    def test_creates_a_minimal_sidecar_when_none_exists(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        data = sidecar.record_summary(recording, "summary", "model-summarize")
        assert data["base"] == HEBREW_BASE
        assert data["contact"] == "דור אקוקה"
        assert data["date"] == "2026-08-19"

    def test_a_later_refresh_keeps_the_summary_and_provenance(
        self, config: Config, make_call
    ):
        recording = make_call(HEBREW_BASE, transcript="text")
        sidecar.record_summary(recording, "A short call.", "model-summarize")
        path = sidecar.sidecar_path_for(recording)

        data = sidecar.refresh(recording, config)
        sidecar.write(path, data)

        assert data["summary"] == "A short call."
        assert data["stages"]["summarize"]["status"] == "ok"
        assert data["stages"]["summarize"]["model"] == "model-summarize"


class TestResolveInput:
    def test_accepts_a_transcript_path(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="text")
        resolved = sidecar.resolve_input(str(recording.with_suffix(".txt")), config)
        assert resolved == recording

    def test_accepts_a_sidecar_path(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE, transcript="text")
        resolved = sidecar.resolve_input(str(recording.with_suffix(".json")), config)
        assert resolved == recording

    def test_accepts_the_recording_itself(self, config: Config, make_call):
        recording = make_call(HEBREW_BASE)
        assert sidecar.resolve_input(str(recording), config) == recording

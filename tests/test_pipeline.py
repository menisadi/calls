from __future__ import annotations

import pytest

from calls import pipeline, sidecar
from calls.config import Config
from calls.pipeline import ImportReport, PipelineError
from calls.sidecar import Tag

BRACKETED = "[1455]_[1455]_2026-08-30_20-22-21"


def _notify(_message: str) -> None:
    pass


class TestMatchingFiles:
    def test_finds_every_file_of_one_call(self, config: Config, make_call):
        recording = make_call(BRACKETED, transcript="text")
        sidecar.write(sidecar.sidecar_path_for(recording), {"base": BRACKETED})
        recording.with_suffix(".srt").write_text("1\n", encoding="utf-8")

        names = [path.name for path in pipeline.matching_files(BRACKETED, config)]
        assert names == [
            f"{BRACKETED}.json",
            f"{BRACKETED}.opus",
            f"{BRACKETED}.srt",
            f"{BRACKETED}.txt",
        ]

    def test_matches_bracketed_names_literally(self, config: Config, make_call):
        # The bug this guards: a glob or `find -name` reads "[1455]" as a
        # character class matching one of "1", "4" or "5", so it matched
        # nothing at all - every recorder name is bracketed.
        make_call(BRACKETED, transcript="text")
        make_call("1", transcript="decoy")
        make_call("4", transcript="decoy")

        found = pipeline.matching_files(BRACKETED, config)
        assert {path.stem for path in found} == {BRACKETED}

    def test_does_not_match_a_longer_name_with_the_same_prefix(
        self, config: Config, make_call
    ):
        make_call(BRACKETED, transcript="text")
        make_call(f"{BRACKETED}-copy", transcript="text")
        found = pipeline.matching_files(BRACKETED, config)
        assert {path.stem for path in found} == {BRACKETED}

    def test_no_match_returns_empty(self, config: Config):
        assert pipeline.matching_files("nothing-here", config) == []

    def test_a_missing_archive_raises(self, config: Config):
        config.calls_dir.rmdir()
        with pytest.raises(PipelineError):
            pipeline.matching_files(BRACKETED, config)


class TestCallTag:
    def test_reads_the_tag_for_a_preview(self, config: Config, make_call):
        recording = make_call(BRACKETED, transcript="text")
        sidecar.record_tag(
            recording, Tag(general="שעון", specific=["דיבור"]), "m", "he"
        )
        assert pipeline.call_tag(BRACKETED, config) == "שעון, דיבור"

    def test_blank_when_there_is_no_sidecar(self, config: Config):
        assert pipeline.call_tag(BRACKETED, config) == ""

    def test_blank_when_the_sidecar_is_broken(self, config: Config):
        (config.calls_dir / f"{BRACKETED}.json").write_text("{oh no", encoding="utf-8")
        # A preview must never be the thing that stops a removal.
        assert pipeline.call_tag(BRACKETED, config) == ""


class TestPlanFor:
    def test_a_fresh_recording_needs_both_stages(self, config: Config, make_call):
        recording = make_call(BRACKETED)
        plan = pipeline.plan_for(recording, config)
        assert (plan.scribe, plan.topic) == ("missing", "missing")
        assert not plan.is_complete

    def test_transcribed_but_untagged_is_not_complete(self, config: Config, make_call):
        # This is the case the old "a .txt exists means done" rule got wrong:
        # such a call was skipped forever and never reached the index.
        recording = make_call(BRACKETED, transcript="text")
        plan = pipeline.plan_for(recording, config)
        assert (plan.scribe, plan.topic) == ("ok", "missing")
        assert not plan.is_complete

    def test_both_stages_done_is_complete(self, config: Config, make_call):
        recording = make_call(BRACKETED, transcript="text")
        sidecar.record_tag(recording, Tag(general="x"), "m", "he")
        plan = pipeline.plan_for(recording, config)
        assert plan.is_complete

    def test_writes_the_refreshed_sidecar(self, config: Config, make_call):
        recording = make_call(BRACKETED, transcript="text")
        pipeline.plan_for(recording, config)
        assert sidecar.sidecar_path_for(recording).is_file()

    def test_can_report_without_writing(self, config: Config, make_call):
        recording = make_call(BRACKETED, transcript="text")
        plan = pipeline.plan_for(recording, config, refresh=False)
        # Nothing on disk yet, so nothing is known - and nothing was written.
        assert (plan.scribe, plan.topic) == ("missing", "missing")
        assert not sidecar.sidecar_path_for(recording).is_file()


class TestCopyNew:
    def test_copies_and_records_direction(self, config: Config, make_original):
        make_original(BRACKETED, "outgoing")
        report = ImportReport()
        pipeline.copy_new(config, report, dry_run=False, notify=_notify)

        assert report.copied == [f"{BRACKETED}.opus"]
        destination = config.calls_dir / f"{BRACKETED}.opus"
        assert destination.is_file()
        data = sidecar.read(sidecar.sidecar_path_for(destination))
        assert data["direction"] == "out"

    def test_leaves_the_original_in_place(self, config: Config, make_original):
        # The recorder folder is a receive-only Syncthing share: deleting from
        # it risks losing the recording on the phone.
        original = make_original(BRACKETED, "incoming")
        pipeline.copy_new(config, ImportReport(), dry_run=False, notify=_notify)
        assert original.is_file()

    def test_dry_run_writes_nothing(self, config: Config, make_original):
        make_original(BRACKETED, "incoming")
        report = ImportReport()
        pipeline.copy_new(config, report, dry_run=True, notify=_notify)

        assert report.would_copy == [f"{BRACKETED}.opus"]
        assert report.copied == []
        assert list(config.calls_dir.iterdir()) == []

    def test_an_already_imported_call_is_not_copied_again(
        self, config: Config, make_call, make_original
    ):
        make_call(BRACKETED)
        make_original(BRACKETED, "incoming")
        report = ImportReport()
        pipeline.copy_new(config, report, dry_run=False, notify=_notify)
        assert report.copied == []

    def test_but_its_direction_is_still_captured(
        self, config: Config, make_call, make_original
    ):
        # Last chance: direction is unrecoverable once the original is pruned.
        recording = make_call(BRACKETED)
        make_original(BRACKETED, "outgoing")
        pipeline.copy_new(config, ImportReport(), dry_run=False, notify=_notify)
        data = sidecar.read(sidecar.sidecar_path_for(recording))
        assert data["direction"] == "out"

    def test_ignores_non_audio_files(self, config: Config):
        (config.phonerec_dir / "incoming" / "notes.txt").write_text("x")
        report = ImportReport()
        pipeline.copy_new(config, report, dry_run=False, notify=_notify)
        assert report.copied == []

    def test_a_missing_recorder_folder_is_not_an_error(self, config: Config):
        # The phone may simply not have synced yet.
        for folder in ("incoming", "outgoing"):
            (config.phonerec_dir / folder).rmdir()
        pipeline.copy_new(config, ImportReport(), dry_run=False, notify=_notify)


class TestFinish:
    def test_dry_run_records_what_is_outstanding_and_runs_nothing(
        self, config: Config, make_call
    ):
        recording = make_call(BRACKETED, transcript="text")
        plans = [pipeline.plan_for(recording, config)]
        report = ImportReport()

        messages: list[str] = []
        pipeline.finish(plans, config, "he", None, True, True, messages.append, report)

        assert [plan.base for plan in report.pending] == [BRACKETED]
        assert report.tagged == []
        assert any("would run" in message for message in messages)

    def test_a_failing_stage_is_recorded_and_the_batch_continues(
        self, config: Config, make_call
    ):
        # scribe_command points at a tool that does not exist, so both calls
        # fail - the point is that the second is still attempted.
        first = make_call("[A]_[0501111111]_2026-09-01_10-00-00")
        second = make_call("[B]_[0502222222]_2026-09-02_10-00-00")
        plans = [
            pipeline.plan_for(first, config),
            pipeline.plan_for(second, config),
        ]
        report = ImportReport()

        pipeline.finish(plans, config, "he", None, False, True, _notify, report)

        assert report.failures == [first.stem, second.stem]
        assert report.transcribed == []

    def test_nothing_outstanding_loads_no_model(self, config: Config, make_call):
        # Loading the model costs seconds; a no-op run must not pay for it.
        recording = make_call(BRACKETED, transcript="text")
        sidecar.record_tag(recording, Tag(general="x"), "m", "he")
        plans = [pipeline.plan_for(recording, config)]
        report = ImportReport()

        messages: list[str] = []
        pipeline.finish(plans, config, "he", None, False, True, messages.append, report)

        assert not any("loading model" in message for message in messages)
        assert report.failures == []


class TestRemove:
    def test_permanent_delete_removes_the_files(self, config: Config, make_call):
        recording = make_call(BRACKETED, transcript="text")
        files = pipeline.matching_files(BRACKETED, config)
        pipeline.remove(files, permanent=True)
        assert not recording.is_file()
        assert not recording.with_suffix(".txt").is_file()

    def test_a_missing_trash_tool_is_reported_clearly(
        self, config: Config, make_call, monkeypatch
    ):
        make_call(BRACKETED)
        monkeypatch.setenv("PATH", "")
        with pytest.raises(PipelineError, match="--permanent"):
            pipeline.remove(pipeline.matching_files(BRACKETED, config), False)

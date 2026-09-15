from __future__ import annotations

from datetime import date, timedelta

import pytest

from calls import cli, index, sidecar
from calls.config import Config


@pytest.fixture
def indexed_call(config: Config, make_call):
    """Create a call with a transcript and a refreshed sidecar, ready to filter."""

    def _add(base: str, transcript: str = "text", direction: str = "in"):
        recording = make_call(base, transcript=transcript)
        data = sidecar.refresh(recording, config, direction=direction)
        sidecar.write(sidecar.sidecar_path_for(recording), data)
        return recording

    return _add


def _rm(config: Config, argv: list[str]):
    parser = cli.build_parser(config)
    args = parser.parse_args(["rm", *argv])
    return args.handler(args, config)


class TestRecordingOnly:
    def test_removes_only_the_audio(self, config: Config, indexed_call):
        base = "[A]_[0501111111]_2026-08-30_20-22-21"
        recording = indexed_call(base)

        result = _rm(config, [base, "--recording-only", "--permanent", "--force"])

        assert result == 0
        assert not recording.is_file()
        assert recording.with_suffix(".txt").is_file()
        assert sidecar.sidecar_path_for(recording).is_file()

    def test_clears_the_recording_field_but_nothing_else(
        self, config: Config, indexed_call
    ):
        base = "[A]_[0501111111]_2026-08-30_20-22-21"
        recording = indexed_call(base)
        sidecar.record_tag(recording, sidecar.Tag(general="x"), "m", "he")

        _rm(config, [base, "--recording-only", "--permanent", "--force"])

        data = sidecar.read(sidecar.sidecar_path_for(recording))
        assert data["recording"] is None
        assert data["stages"]["recording"] == {
            "status": "removed",
            "at": data["stages"]["recording"]["at"],
        }
        assert data["tag"]["general"] == "x"

    def test_a_call_with_no_recording_left_is_skipped_not_fatal(
        self, config: Config, indexed_call
    ):
        base = "[A]_[0501111111]_2026-08-30_20-22-21"
        recording = indexed_call(base)
        recording.unlink()

        result = _rm(config, [base, "--recording-only", "--permanent", "--force"])

        assert result == 0
        assert sidecar.sidecar_path_for(recording).is_file()

    def test_full_removal_still_deletes_everything(self, config: Config, indexed_call):
        base = "[A]_[0501111111]_2026-08-30_20-22-21"
        recording = indexed_call(base)

        _rm(config, [base, "--permanent", "--force"])

        assert not recording.is_file()
        assert not recording.with_suffix(".txt").is_file()
        assert not sidecar.sidecar_path_for(recording).is_file()


class TestDateFilters:
    def test_older_than_selects_only_calls_past_the_cutoff(
        self, config: Config, indexed_call
    ):
        old_date = (date.today() - timedelta(days=100)).isoformat()
        new_date = (date.today() - timedelta(days=1)).isoformat()
        old_base = f"[Old]_[0501111111]_{old_date}_10-00-00"
        new_base = f"[New]_[0502222222]_{new_date}_10-00-00"
        old_recording = indexed_call(old_base)
        new_recording = indexed_call(new_base)
        index.build(config)

        result = _rm(
            config, ["--older-than", "30", "--recording-only", "--permanent", "--force"]
        )

        assert result == 0
        assert not old_recording.is_file()
        assert new_recording.is_file()

    def test_since_and_until_combine_like_index_dash_l(
        self, config: Config, indexed_call
    ):
        keep = indexed_call("[Keep]_[0501111111]_2026-01-01_10-00-00")
        drop = indexed_call("[Drop]_[0502222222]_2026-06-15_10-00-00")
        index.build(config)

        result = _rm(
            config,
            [
                "--since",
                "2026-06-01",
                "--until",
                "2026-06-30",
                "--recording-only",
                "--permanent",
                "--force",
            ],
        )

        assert result == 0
        assert keep.is_file()
        assert not drop.is_file()

    def test_until_and_older_than_are_mutually_exclusive(self, config: Config):
        parser = cli.build_parser(config)
        with pytest.raises(SystemExit):
            parser.parse_args(["rm", "--until", "2026-01-01", "--older-than", "5"])

    def test_no_matches_is_reported_and_not_an_error(self, config: Config):
        index.build(config)
        result = _rm(config, ["--older-than", "9999"])
        assert result == 1


class TestBatchConfirmation:
    def test_declining_the_batch_prompt_skips_everything(
        self, config: Config, indexed_call, monkeypatch
    ):
        first = indexed_call("[A]_[0501111111]_2026-08-01_10-00-00")
        second = indexed_call("[B]_[0502222222]_2026-08-02_10-00-00")
        index.build(config)

        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        result = _rm(config, ["--contact", "050", "--recording-only", "--permanent"])

        assert result == 0
        assert first.is_file()
        assert second.is_file()

    def test_accepting_the_batch_prompt_removes_everything(
        self, config: Config, indexed_call, monkeypatch
    ):
        first = indexed_call("[A]_[0501111111]_2026-08-01_10-00-00")
        second = indexed_call("[B]_[0502222222]_2026-08-02_10-00-00")
        index.build(config)

        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        result = _rm(config, ["--contact", "050", "--recording-only", "--permanent"])

        assert result == 0
        assert not first.is_file()
        assert not second.is_file()

    def test_explicit_names_still_prompt_one_at_a_time(
        self, config: Config, indexed_call, monkeypatch
    ):
        first = indexed_call("[A]_[0501111111]_2026-08-01_10-00-00")
        second = indexed_call("[B]_[0502222222]_2026-08-02_10-00-00")

        replies = iter(["y", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
        result = _rm(
            config,
            [first.stem, second.stem, "--recording-only", "--permanent"],
        )

        assert result == 0
        assert not first.is_file()
        assert second.is_file()

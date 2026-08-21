import inspect
import json
from pathlib import Path

import pytest

import asli.cli
from asli.turnbench.cli import main


FIXTURES = Path(__file__).parents[2] / "turnbench" / "fixtures"


@pytest.fixture
def fixture_paths():
    return type(
        "FixturePaths",
        (),
        {
            "recordings": FIXTURES / "recordings.jsonl",
            "labels": FIXTURES / "labels.jsonl",
            "events": FIXTURES / "events.jsonl",
        },
    )()


def _score_args(fixture_paths, out: Path, *, events: Path | None = None):
    return [
        "score",
        "--recordings",
        str(fixture_paths.recordings),
        "--labels",
        str(fixture_paths.labels),
        "--events",
        str(events or fixture_paths.events),
        "--out",
        str(out),
        "--provider",
        "fixture",
    ]


def test_score_command_writes_deterministic_report(tmp_path, fixture_paths):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main(_score_args(fixture_paths, first)) == 0
    assert main(_score_args(fixture_paths, second)) == 0
    report = json.loads(first.read_text())
    assert first.read_text() == second.read_text()
    assert report["run_id"] == "fixture-run"
    assert report["overall"]["aggregation"] == "macro_by_language"
    assert report["overall"]["pir"] == 1.0
    assert report["overall"]["delay_p50_ms"] == 300
    assert report["micro_overall"]["aggregation"] == "micro_by_decision"
    assert report["micro_overall"]["decision_n"] == 5


def test_existing_asli_cli_has_no_turnbench_subcommand():
    assert "turnbench" not in inspect.getsource(asli.cli.main)


def test_unknown_trace_decision_is_rejected(tmp_path, fixture_paths, capsys):
    bad_events = tmp_path / "events.jsonl"
    bad_events.write_text(
        fixture_paths.events.read_text().replace('"hi-continue"', '"missing"', 1)
    )
    with pytest.raises(SystemExit) as error:
        main(_score_args(fixture_paths, tmp_path / "out.json", events=bad_events))
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "usage: asli-turnbench" in captured.err
    assert "asli-turnbench: error: missing trace for decision_id: hi-continue" in captured.err


def test_malformed_jsonl_is_a_command_error(tmp_path, fixture_paths, capsys):
    bad_events = tmp_path / "events.jsonl"
    bad_events.write_text("not json\n")
    with pytest.raises(SystemExit) as error:
        main(_score_args(fixture_paths, tmp_path / "out.json", events=bad_events))
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "asli-turnbench: error:" in captured.err
    assert "events.jsonl:1" in captured.err


def test_config_must_be_an_object(tmp_path, fixture_paths, capsys):
    with pytest.raises(SystemExit) as error:
        main(_score_args(fixture_paths, tmp_path / "out.json") + ["--config-json", "[]"])
    assert error.value.code == 2
    assert "must decode to an object" in capsys.readouterr().err


def test_config_rejects_duplicate_keys_without_writing_report(
    tmp_path, fixture_paths, capsys
):
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as error:
        main(
            _score_args(fixture_paths, out)
            + ["--config-json", '{"threshold":500,"threshold":900}']
        )
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "asli-turnbench: error: duplicate key: threshold" in captured.err
    assert not out.exists()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_config_rejects_nonstandard_json_constants(tmp_path, fixture_paths, capsys, constant):
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as error:
        main(
            _score_args(fixture_paths, out)
            + ["--config-json", '{"temperature":' + constant + "}"]
        )
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "asli-turnbench: error:" in captured.err
    assert "non-standard JSON constant" in captured.err
    assert not out.exists()


def test_annotation_guide_names_both_excluded_outcomes():
    guide = Path("turnbench/ANNOTATION.md").read_text()
    assert "overlap" in guide
    assert "unclear" in guide


def test_report_docs_distinguish_macro_and_micro_denominators():
    docs = " ".join(Path("docs/turnbench.md").read_text().split())
    assert "Micro and grouped availability rates use that summary's `trace_n`" in docs
    assert "macro `overall` rates use their corresponding `*_language_n`" in docs
    assert "uncertainty for `micro_overall[\"pir\"]`, never `overall[\"pir\"]`" in docs


def test_only_validated_score_inputs_is_public_report_api():
    import asli.turnbench as turnbench
    import asli.turnbench.report as report_module

    docs = Path("docs/turnbench.md").read_text()
    assert hasattr(turnbench, "score_inputs")
    assert not hasattr(turnbench, "build_report")
    assert not hasattr(report_module, "build_report")
    assert "direct library report" not in docs
    assert "through `score_inputs()`" in docs

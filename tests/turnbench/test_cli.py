import inspect
import json
import socket
import tomllib
from pathlib import Path

import numpy as np
import pytest

import asli.cli
import asli.turnbench.cli as turnbench_cli
from asli.turnbench.auto_label import EndpointObservation
from asli.turnbench.auto_report import DiarBenchExportProvenance
from asli.turnbench.auto_schema import (
    AutoPrediction,
    DiarBenchCandidate,
    DiarBenchReference,
    REFERENCE_SOURCE,
    write_candidates,
    write_predictions,
    write_references,
)
from asli.turnbench.cli import main
from asli.synth import write_wav


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


def _diarbench_row():
    return {
        "sample_id": "sample-1",
        "recording_id": "call-42",
        "dataset_type": "telephone",
        "annotated_transcript": [
            {"speaker_id": "caller", "start_time": 0.0, "end_time": 1.0, "text": "namaste"},
            {"speaker_id": "caller", "start_time": 1.5, "end_time": 2.0, "text": "haan"},
        ],
        "audio": {"array": np.zeros(3_000, dtype=np.int16), "sampling_rate": 1000},
    }


def test_diarbench_export_writes_isolated_records_and_versioned_manifest(tmp_path, monkeypatch):
    out_dir = tmp_path / "diarbench"
    seen = {}

    def local_loader(*, dataset, requested_revision, config, split, streaming, limit):
        seen.update(
            dataset=dataset, requested_revision=requested_revision, config=config,
            split=split, streaming=streaming, limit=limit,
        )
        return turnbench_cli.DiarBenchLoad([_diarbench_row()], "a" * 40)

    monkeypatch.setattr(turnbench_cli, "DIARBENCH_LOADER_FACTORY", local_loader)

    assert main([
        "diarbench", "export", "--language", "Hindi", "--limit", "2",
        "--min-pause-ms", "300", "--max-pause-ms", "1000", "--out-dir", str(out_dir),
    ]) == 0

    assert seen == {
        "dataset": "sarvamai/indic-diarbench", "requested_revision": "main",
        "config": "Hindi", "split": "test", "streaming": True, "limit": 2,
    }
    candidate_rows = turnbench_cli.read_candidates(out_dir / "candidates.jsonl")
    reference_rows = turnbench_cli.read_references(out_dir / "references.jsonl")
    assert len(candidate_rows) == 1
    assert [row.outcome for row in reference_rows] == ["continue", "unclear"]
    assert candidate_rows[0].recording_id == "sample-1"
    assert candidate_rows[0].source_recording_id == "call-42"
    assert candidate_rows[0].condition == "telephone"
    assert Path(candidate_rows[0].audio_path).is_file()
    assert DiarBenchExportProvenance.from_dict(json.loads((out_dir / "manifest.json").read_text())).to_dict() == {
        "schema": "turnbench.diarbench.export.v1",
        "dataset": "sarvamai/indic-diarbench", "dataset_revision": "a" * 40,
        "requested_languages": ["Hindi"], "min_pause_ms": 300,
        "max_pause_ms": 1000, "context_ms": 5000,
    }


def test_diarbench_decoder_audio_downmixes_channel_first_without_torch_import(monkeypatch):
    class Samples:
        data = np.array([[1000, 3000, 5000], [3000, 5000, 7000]], dtype=np.int16)
        sample_rate = 16_000

    class AudioDecoder:
        def __init__(self, data):
            self.data = data

        def get_all_samples(self):
            return type("Decoded", (), {"data": self.data, "sample_rate": 16_000})()

    original_import = __import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("decoder handling must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_torch)
    pcm, rate = turnbench_cli._decoded_audio(
        {"audio": AudioDecoder(Samples.data)}, sample_id="decoder-sample"
    )
    mono_pcm, mono_rate = turnbench_cli._decoded_audio(
        {"audio": AudioDecoder(np.array([[1000, 3000, 5000]], dtype=np.int16))},
        sample_id="mono-decoder-sample",
    )

    assert rate == 16_000
    assert pcm.tolist() == [2000, 4000, 6000]
    assert mono_rate == 16_000
    assert mono_pcm.tolist() == [1000, 3000, 5000]


def test_diarbench_export_never_pulls_more_than_limit_stream_rows(tmp_path, monkeypatch):
    class TrackingRows:
        def __init__(self):
            self.calls = 0
            self.rows = [_diarbench_row() for _ in range(3)]
            self.rows[1]["sample_id"] = "sample-2"
            self.rows[2]["sample_id"] = "sample-3"

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls > len(self.rows):
                raise StopIteration
            return self.rows[self.calls - 1]

    rows = TrackingRows()
    monkeypatch.setattr(
        turnbench_cli,
        "DIARBENCH_LOADER_FACTORY",
        lambda **_: turnbench_cli.DiarBenchLoad(rows, "b" * 40),
    )

    assert main([
        "diarbench", "export", "--language", "Hindi", "--limit", "2",
        "--min-pause-ms", "300", "--max-pause-ms", "1000",
        "--out-dir", str(tmp_path / "diarbench"),
    ]) == 0

    assert rows.calls == 2


def test_diarbench_export_missing_optional_dependency_writes_nothing(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "not-created"
    original_import = __import__

    def missing_datasets(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_datasets)
    with pytest.raises(SystemExit) as error:
        main([
            "diarbench", "export", "--language", "Hindi", "--limit", "1",
            "--min-pause-ms", "300", "--max-pause-ms", "1000", "--out-dir", str(out_dir),
        ])
    assert error.value.code == 2
    assert "pip install 'asli[diarbench]'" in capsys.readouterr().err
    assert not out_dir.exists()


def test_auto_label_uses_candidates_and_never_opens_references(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(3_000, dtype=np.int16), 1000)
    candidate = DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="source-1",
        audio_path=str(audio), language="Hindi", condition="diarbench",
        target_speaker_id="caller", context_start_ms=500,
        previous_speech_end_ms=1000, observation_end_ms=1500,
    )
    candidates = tmp_path / "candidates.jsonl"
    write_candidates(candidates, [candidate])
    (tmp_path / "manifest.json").write_text(json.dumps(DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench", dataset_revision="a" * 40,
        requested_languages=("Hindi",), min_pause_ms=300, max_pause_ms=1000,
        context_ms=500,
    ).to_dict()))
    output = tmp_path / "predictions.jsonl"
    seen = {}

    def fake_observer(*, model):
        seen["model"] = model

        def observe(pcm, rate, language):
            seen["language"] = language
            return EndpointObservation(None, False, None)

        return observe

    monkeypatch.setattr(turnbench_cli, "AUTO_OBSERVER_FACTORY", fake_observer)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    assert main([
        "auto", "label", "--candidates", str(candidates), "--agent", "openai",
        "--model", "gpt-4o-transcribe", "--context-ms", "500", "--out", str(output),
    ]) == 0

    predictions = turnbench_cli.read_predictions(output)
    assert seen == {"model": "gpt-4o-transcribe", "language": "hi"}
    assert predictions[0].outcome == "continue"
    assert predictions[0].model == "gpt-realtime"
    assert predictions[0].config == {
        "context_ms": 500, "turn_detection": "semantic_vad", "eagerness": "auto",
        "create_response": False, "require_endpoint_timestamps": True,
        "trailing_silence_ms": 0, "effective_language": "hi",
        "transcription_model": "gpt-4o-transcribe", "realtime_model": "gpt-realtime",
    }


def test_auto_label_rejects_context_mismatch_before_creating_observer(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(3_000, dtype=np.int16), 1000)
    candidates = tmp_path / "candidates.jsonl"
    write_candidates(candidates, [DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="source-1",
        audio_path=str(audio), language="Hindi", condition="diarbench",
        target_speaker_id="caller", context_start_ms=0,
        previous_speech_end_ms=100, observation_end_ms=500,
    )])
    (tmp_path / "manifest.json").write_text(json.dumps(DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench", dataset_revision="a" * 40,
        requested_languages=("Hindi",), min_pause_ms=300, max_pause_ms=1000,
        context_ms=500,
    ).to_dict()))
    created = False

    def forbidden_observer(**kwargs):
        nonlocal created
        created = True
        return lambda *_: pytest.fail("provider must not run")

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(turnbench_cli, "AUTO_OBSERVER_FACTORY", forbidden_observer)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(SystemExit):
        main([
            "auto", "label", "--candidates", str(candidates), "--agent", "openai",
            "--model", "gpt-4o-transcribe", "--context-ms", "400", "--out", str(output),
        ])
    assert "--context-ms does not match export manifest" in capsys.readouterr().err
    assert not created
    assert not output.exists()


def test_auto_label_requires_nonblank_api_key_before_creating_observer(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(3_000, dtype=np.int16), 1000)
    candidates = tmp_path / "candidates.jsonl"
    write_candidates(candidates, [DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="source-1",
        audio_path=str(audio), language="Hindi", condition="diarbench",
        target_speaker_id="caller", context_start_ms=500,
        previous_speech_end_ms=1000, observation_end_ms=1500,
    )])
    (tmp_path / "manifest.json").write_text(json.dumps(DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench", dataset_revision="a" * 40,
        requested_languages=("Hindi",), min_pause_ms=300, max_pause_ms=1000,
        context_ms=500,
    ).to_dict()))
    created = False

    def forbidden_observer(**kwargs):
        nonlocal created
        created = True
        return lambda *_: pytest.fail("provider must not run")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(turnbench_cli, "AUTO_OBSERVER_FACTORY", forbidden_observer)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(SystemExit):
        main([
            "auto", "label", "--candidates", str(candidates), "--agent", "openai",
            "--model", "gpt-4o-transcribe", "--context-ms", "500", "--out", str(output),
        ])
    assert "OPENAI_API_KEY must be set" in capsys.readouterr().err
    assert not created
    assert not output.exists()


def test_auto_label_requires_export_manifest_before_creating_observer(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(3_000, dtype=np.int16), 1000)
    candidates = tmp_path / "candidates.jsonl"
    write_candidates(candidates, [DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="source-1",
        audio_path=str(audio), language="Hindi", condition="diarbench",
        target_speaker_id="caller", context_start_ms=500,
        previous_speech_end_ms=1000, observation_end_ms=1500,
    )])
    created = False

    def forbidden_observer(**kwargs):
        nonlocal created
        created = True
        return lambda *_: pytest.fail("provider must not run")

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(turnbench_cli, "AUTO_OBSERVER_FACTORY", forbidden_observer)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(SystemExit):
        main([
            "auto", "label", "--candidates", str(candidates), "--agent", "openai",
            "--model", "gpt-4o-transcribe", "--context-ms", "500", "--out", str(output),
        ])
    assert "No such file" in capsys.readouterr().err
    assert not created
    assert not output.exists()


def test_auto_compare_uses_default_manifest_and_writes_deterministic_json(tmp_path):
    candidate = DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="source-1",
        audio_path="audio.wav", language="Hindi", condition="diarbench",
        target_speaker_id="caller", context_start_ms=500,
        previous_speech_end_ms=1000, observation_end_ms=1500,
    )
    candidates = tmp_path / "candidates.jsonl"
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_candidates(candidates, [candidate])
    write_references(references, [DiarBenchReference(candidate, "yield", REFERENCE_SOURCE, None)])
    write_predictions(predictions, [AutoPrediction(
        decision_id="d1", run_id="run-1", agent="openai", model="gpt-4o-transcribe",
        config={"context_ms": 500, "mode": "semantic_vad"}, status="available",
        outcome="yield", endpoint_ms=1200, unavailable_reason=None,
    )])
    (tmp_path / "manifest.json").write_text(json.dumps(DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench", dataset_revision="a" * 40,
        requested_languages=("Hindi",), min_pause_ms=300, max_pause_ms=1000,
        context_ms=500,
    ).to_dict()))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = [
        "auto", "compare", "--candidates", str(candidates), "--references", str(references),
        "--predictions", str(predictions),
    ]
    assert main(args + ["--out", str(first)]) == 0
    assert main(args + ["--out", str(second)]) == 0
    assert first.read_text() == second.read_text()
    assert json.loads(first.read_text())["micro_overall"]["accuracy"] == 1.0


def test_score_neither_imports_datasets_nor_uses_network(tmp_path, fixture_paths, monkeypatch):
    original_import = __import__

    def forbidden_datasets(name, *args, **kwargs):
        if name == "datasets":
            raise AssertionError("score imported optional datasets")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", forbidden_datasets)
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("network used"))
    assert main(_score_args(fixture_paths, tmp_path / "report.json")) == 0


def test_diarbench_extra_uses_datasets_coupled_audio_dependencies():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = metadata["project"]["optional-dependencies"]["diarbench"]

    assert dependencies == ["datasets[audio]>=4.0"]


def test_export_filters_binary_references_without_candidates_before_compare(tmp_path, monkeypatch):
    row = _diarbench_row()
    row["annotated_transcript"] = [
        {"speaker_id": "a", "start_time": 0.0, "end_time": 1.0},
        {"speaker_id": "a", "start_time": 1.1, "end_time": 2.0},  # short
        {"speaker_id": "a", "start_time": 2.5, "end_time": 3.0},  # eligible
        {"speaker_id": "b", "start_time": 5.5, "end_time": 6.0},  # long
        {"speaker_id": "b", "start_time": 6.0, "end_time": 6.5},  # zero
        {"speaker_id": "a", "start_time": 6.4, "end_time": 6.8},  # overlap
    ]
    row["audio"] = {"array": np.zeros(8_000, dtype=np.int16), "sampling_rate": 1000}
    monkeypatch.setattr(
        turnbench_cli, "DIARBENCH_LOADER_FACTORY",
        lambda **_: turnbench_cli.DiarBenchLoad([row], "c" * 40),
    )
    exported = tmp_path / "export"
    assert main([
        "diarbench", "export", "--language", "Hindi", "--limit", "1",
        "--min-pause-ms", "300", "--max-pause-ms", "1000", "--out-dir", str(exported),
    ]) == 0
    candidates = turnbench_cli.read_candidates(exported / "candidates.jsonl")
    references = turnbench_cli.read_references(exported / "references.jsonl")
    assert all(
        reference.outcome not in {"continue", "yield"}
        or reference.candidate.decision_id in {candidate.decision_id for candidate in candidates}
        for reference in references
    )
    predictions = []
    for candidate in candidates:
        reference = next(row for row in references if row.candidate == candidate)
        predictions.append(AutoPrediction(
            decision_id=candidate.decision_id, run_id="fixture-run", agent="openai",
            model="gpt-realtime", config={"context_ms": 5000}, status="available",
            outcome=reference.outcome,
            endpoint_ms=(candidate.previous_speech_end_ms + 1 if reference.outcome == "yield" else None),
            unavailable_reason=None,
        ))
    prediction_path = tmp_path / "predictions.jsonl"
    write_predictions(prediction_path, predictions)
    report = tmp_path / "report.json"
    assert main([
        "auto", "compare", "--candidates", str(exported / "candidates.jsonl"),
        "--references", str(exported / "references.jsonl"),
        "--predictions", str(prediction_path), "--out", str(report),
    ]) == 0
    assert json.loads(report.read_text())["candidate_n"] == len(candidates)


def test_diarbench_provider_language_mapping_uses_codes_or_omits_hint():
    assert turnbench_cli._effective_provider_language("Hindi") == "hi"
    assert turnbench_cli._effective_provider_language("Tamil") == "ta"
    assert turnbench_cli._effective_provider_language("Bodo") is None


def test_export_rejects_annotation_past_decoded_audio_without_output(tmp_path, monkeypatch, capsys):
    row = _diarbench_row()
    row["annotated_transcript"][1]["end_time"] = 4.0
    monkeypatch.setattr(
        turnbench_cli, "DIARBENCH_LOADER_FACTORY",
        lambda **_: turnbench_cli.DiarBenchLoad([row], "d" * 40),
    )
    out_dir = tmp_path / "export"
    with pytest.raises(SystemExit):
        main([
            "diarbench", "export", "--language", "Hindi", "--limit", "1",
            "--min-pause-ms", "300", "--max-pause-ms", "1000", "--out-dir", str(out_dir),
        ])
    assert "sample-1: annotated_transcript[1].end_time exceeds decoded audio duration" in capsys.readouterr().err
    assert not out_dir.exists()


def test_auto_label_preflight_rejects_duplicate_ids_and_manifest_mismatch(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(3_000, dtype=np.int16), 1000)
    candidate = DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="s1", audio_path=str(audio),
        language="Hindi", condition="diarbench", target_speaker_id="a", context_start_ms=500,
        previous_speech_end_ms=1000, observation_end_ms=1500,
    )
    candidates = tmp_path / "candidates.jsonl"
    write_candidates(candidates, [candidate, candidate])
    (tmp_path / "manifest.json").write_text(json.dumps(DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench", dataset_revision="e" * 40,
        requested_languages=("Hindi",), min_pause_ms=300, max_pause_ms=1000, context_ms=500,
    ).to_dict()))
    created = False
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")

    def forbidden_observer(**kwargs):
        nonlocal created
        created = True
        return lambda *_: pytest.fail("provider must not run")

    monkeypatch.setattr(turnbench_cli, "AUTO_OBSERVER_FACTORY", forbidden_observer)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(SystemExit):
        main([
            "auto", "label", "--candidates", str(candidates), "--agent", "openai",
            "--model", "gpt-4o-transcribe", "--context-ms", "500", "--out", str(output),
        ])
    assert "duplicate candidate decision_id" in capsys.readouterr().err
    assert not created
    assert not output.exists()

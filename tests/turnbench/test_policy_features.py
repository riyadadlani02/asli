import numpy as np
import pytest

import asli.turnbench.policy_features as policy_features
from asli.turnbench.auto_report import DiarBenchExportProvenance
from asli.turnbench.auto_schema import AutoPrediction, DiarBenchCandidate
from asli.turnbench.policy_features import extract_policy_features


def make_candidate(
    decision_id="d1", *, previous_speech_end_ms=800, observation_end_ms=1600,
    audio_path="fixture.wav",
):
    return DiarBenchCandidate(
        decision_id=decision_id,
        recording_id=f"recording-{decision_id}",
        source_recording_id="source-1",
        audio_path=audio_path,
        language="Hindi",
        condition="clean",
        target_speaker_id="speaker-1",
        context_start_ms=0,
        previous_speech_end_ms=previous_speech_end_ms,
        observation_end_ms=observation_end_ms,
    )


def provenance():
    return DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench",
        dataset_revision="test",
        requested_languages=("Hindi",),
        min_pause_ms=300,
        max_pause_ms=2000,
        context_ms=800,
    )


def unavailable_prediction(decision_id):
    return AutoPrediction(
        decision_id=decision_id,
        run_id="semantic-1",
        agent="local",
        model="completed-observation",
        config={"context_ms": 800},
        status="unavailable",
        outcome=None,
        endpoint_ms=None,
        unavailable_reason="observer_failed",
    )


def test_feature_extraction_uses_only_audio_before_observation_boundary():
    """Fails if post-boundary samples influence a local feature."""
    pcm = np.concatenate([
        np.full(800, 1000, np.int16),
        np.zeros(800, np.int16),
        np.full(800, 30000, np.int16),
    ])
    candidate = make_candidate()

    rows = extract_policy_features(
        [candidate], export_provenance=provenance(), read_audio=lambda _: (pcm, 1000),
    )

    assert rows[0].pause_ms == 800
    assert rows[0].trailing_energy == pytest.approx(1000 / 32768)
    assert rows[0].semantic_status == "absent"


def test_default_wav_reader_never_requests_audio_after_observation_boundary(
    tmp_path, monkeypatch,
):
    """Fails if the default WAV reader decodes samples past the feature window."""
    path = tmp_path / "fixture.wav"
    pcm = np.concatenate([np.full(1600, 1000, np.int16), np.full(800, 30000, np.int16)])
    with policy_features.wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(1000)
        output.writeframes(pcm.tobytes())

    original_open = policy_features.wave.open
    requested_frames = []

    class ReaderSpy:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def readframes(self, count):
            requested_frames.append(count)
            return self.source.readframes(count)

        def __getattr__(self, name):
            return getattr(self.source, name)

    monkeypatch.setattr(
        policy_features.wave, "open", lambda *args, **kwargs: ReaderSpy(original_open(*args, **kwargs)),
    )
    extract_policy_features(
        [make_candidate(audio_path=str(path))], export_provenance=provenance(),
    )

    assert requested_frames == [1600]


def test_feature_extraction_keeps_unavailable_semantic_evidence_unavailable():
    """Fails if unavailable semantic evidence is fabricated as a decision."""
    rows = extract_policy_features(
        [make_candidate()], export_provenance=provenance(),
        semantic_predictions=[unavailable_prediction("d1")],
        read_audio=lambda _: (np.ones(4000, np.int16), 1000),
    )

    assert rows[0].semantic_status == "unavailable"
    assert rows[0].semantic_outcome is None


def test_feature_extraction_rejects_truncated_audio():
    """Fails if a feature can be made from audio short of its observation window."""
    with pytest.raises(ValueError, match="observation_end_ms"):
        extract_policy_features(
            [make_candidate()], export_provenance=provenance(),
            read_audio=lambda _: (np.ones(1599, np.int16), 1000),
        )


def test_feature_extraction_zero_energy_has_finite_zero_features():
    """Fails if silence produces NaN or infinite acoustic features."""
    row = extract_policy_features(
        [make_candidate()], export_provenance=provenance(),
        read_audio=lambda _: (np.zeros(1600, np.int16), 1000),
    )[0]

    assert (row.trailing_energy, row.trailing_energy_slope) == (0.0, 0.0)
    assert row.trailing_speech_ms == 0
    assert row.local_speech_rate_hz == 0.0


def test_feature_extraction_uses_a_varying_no_lookahead_onset_rate():
    """Fails if non-silent temporal patterns collapse to the same speech-rate value."""
    steady = np.concatenate([np.full(800, 1000, np.int16), np.zeros(800, np.int16)])
    alternating = np.concatenate([
        np.tile(np.concatenate([np.full(20, 1000, np.int16), np.zeros(20, np.int16)]), 20),
        np.zeros(800, np.int16),
    ])
    candidate = make_candidate()

    steady_rate = extract_policy_features(
        [candidate], export_provenance=provenance(), read_audio=lambda _: (steady, 1000),
    )[0].local_speech_rate_hz
    alternating_rate = extract_policy_features(
        [candidate], export_provenance=provenance(), read_audio=lambda _: (alternating, 1000),
    )[0].local_speech_rate_hz
    post_boundary_rate = extract_policy_features(
        [candidate], export_provenance=provenance(),
        read_audio=lambda _: (np.concatenate([steady, alternating[:800]]), 1000),
    )[0].local_speech_rate_hz

    assert steady_rate == 1.25
    assert alternating_rate == 25.0
    assert post_boundary_rate == steady_rate


def test_feature_extraction_rejects_duplicate_semantic_ids():
    """Fails if ambiguous optional semantic evidence is silently selected."""
    with pytest.raises(ValueError, match="duplicate prediction decision_id: d1"):
        extract_policy_features(
            [make_candidate()], export_provenance=provenance(),
            semantic_predictions=[unavailable_prediction("d1"), unavailable_prediction("d1")],
            read_audio=lambda _: (np.ones(1600, np.int16), 1000),
        )


def test_feature_extraction_is_sorted_and_carries_yield_offset():
    """Fails if input ordering or absolute semantic endpoint leaks into feature rows."""
    yield_prediction = AutoPrediction(
        decision_id="a", run_id="semantic-1", agent="local", model="completed-observation",
        config={"context_ms": 800}, status="available", outcome="yield", endpoint_ms=1000,
        unavailable_reason=None,
    )
    rows = extract_policy_features(
        [make_candidate("z"), make_candidate("a")], export_provenance=provenance(),
        semantic_predictions=[yield_prediction],
        read_audio=lambda _: (np.ones(1600, np.int16), 1000),
    )

    assert [row.decision_id for row in rows] == ["a", "z"]
    assert rows[0].semantic_endpoint_offset_ms == 200
    assert rows[0].semantic_outcome == "yield"

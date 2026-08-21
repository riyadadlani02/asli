import pytest

from asli.turnbench.policy_runtime import decide_policy, probability_continue
from asli.turnbench.policy_schema import POLICY_FEATURE_SCHEMA, PolicyArtifact, PolicyFeature


def make_feature(decision_id="d1", *, pause_ms=700, language="Hindi", fingerprint="e" * 64):
    return PolicyFeature(
        decision_id=decision_id, recording_id="recording-1", source_recording_id="source-1",
        language=language, condition="clean", export_fingerprint=fingerprint,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1},
        audio_fingerprint="a" * 64, pause_ms=pause_ms, trailing_energy=0.2,
        trailing_energy_slope=-0.1, trailing_speech_ms=500, local_speech_rate_hz=4.0,
        semantic_status="absent", semantic_outcome=None, semantic_endpoint_offset_ms=None,
    )


def make_artifact(*, yield_threshold=0.3, hold_threshold=0.7, language="Hindi", fingerprint="e" * 64):
    return PolicyArtifact(
        policy_id="policy-1", language=language, feature_schema=POLICY_FEATURE_SCHEMA,
        export_fingerprint=fingerprint,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1},
        coefficients=(-5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        means=(0.0,) * 7, scales=(1.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        yield_threshold=yield_threshold, hold_threshold=hold_threshold, grace_ms=150,
        hard_deadline_ms=800, train_source_recording_ids=("train",),
        calibration_source_recording_ids=("cal",),
    )


def test_runtime_does_not_accept_or_need_a_reference():
    """Fails if live policy decisions acquire a reference dependency."""
    decision = decide_policy(make_feature(pause_ms=700), make_artifact())

    assert decision.status == "available"
    assert decision.action in {"hold", "yield", "uncertain"}
    assert decision.probability_continue is not None


@pytest.mark.parametrize(
    ("pause_ms", "action"),
    [(0, "yield"), (500, "uncertain"), (1000, "hold")],
)
def test_runtime_uses_the_calibrated_three_way_threshold_band(pause_ms, action):
    """Fails if threshold boundaries choose the wrong runtime action."""
    assert decide_policy(make_feature(pause_ms=pause_ms), make_artifact()).action == action


def test_runtime_marks_artifact_feature_mismatch_unavailable():
    """Fails if an artifact is applied to a different feature export."""
    decision = decide_policy(make_feature(fingerprint="x" * 64), make_artifact())

    assert decision.status == "unavailable"
    assert decision.action is None
    assert decision.unavailable_reason == "export_fingerprint_mismatch"


def test_probability_rejects_language_mismatch_before_scoring():
    """Fails if a Hindi artifact can silently score another language."""
    with pytest.raises(ValueError, match="language mismatch"):
        probability_continue(make_feature(language="Tamil"), make_artifact())

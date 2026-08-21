import pytest

from asli.turnbench.policy_schema import (
    POLICY_FEATURE_SCHEMA,
    PolicyArtifact,
    PolicyFeature,
    PolicySplit,
)
from asli.turnbench.schema import SchemaError


def test_policy_feature_round_trips_without_a_label_or_audio_path():
    row = PolicyFeature(
        decision_id="d1", recording_id="clip-1", source_recording_id="call-1",
        language="Hindi", condition="Near field", export_fingerprint="e" * 64,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1},
        audio_fingerprint="a" * 64, pause_ms=600,
        trailing_energy=0.25, trailing_energy_slope=-0.1,
        trailing_speech_ms=740, local_speech_rate_hz=4.0,
        semantic_status="available", semantic_outcome="continue",
        semantic_endpoint_offset_ms=None,
    )

    encoded = row.to_dict()
    assert "outcome" not in encoded
    assert "audio_path" not in encoded
    assert PolicyFeature.from_dict(encoded) == row


def test_policy_artifact_rejects_reversed_threshold_band():
    with pytest.raises(SchemaError, match="yield_threshold"):
        PolicyArtifact(
            policy_id="p1", language="Hindi", feature_schema=POLICY_FEATURE_SCHEMA,
            export_fingerprint="e" * 64,
            extractor_config={}, coefficients=(0.1,) * 7, means=(0.0,) * 7,
            scales=(1.0,) * 7, yield_threshold=0.8,
            hold_threshold=0.2, grace_ms=150, hard_deadline_ms=800,
            train_source_recording_ids=("call-1",), calibration_source_recording_ids=("call-2",),
        )


def test_policy_split_rejects_non_array_source_recording_ids():
    with pytest.raises(SchemaError, match="train_source_recording_ids must be a list"):
        PolicySplit.from_dict({
            "schema": "turnbench.policy_split.v1", "seed": 1, "language": "Hindi",
            "train_source_recording_ids": "call-1",
            "calibration_source_recording_ids": ["call-2"],
            "test_source_recording_ids": ["call-3"],
        })


def test_policy_artifact_rejects_non_array_source_recording_ids():
    row = {
        "schema": "turnbench.policy_artifact.v1", "policy_id": "p1", "language": "Hindi",
        "feature_schema": POLICY_FEATURE_SCHEMA, "export_fingerprint": "e" * 64,
        "extractor_config": {}, "coefficients": [0.1] * 7, "means": [0.0] * 7,
        "scales": [1.0] * 7, "yield_threshold": 0.2, "hold_threshold": 0.8,
        "grace_ms": 150, "hard_deadline_ms": 800,
        "train_source_recording_ids": "call-1", "calibration_source_recording_ids": ["call-2"],
    }
    with pytest.raises(SchemaError, match="train_source_recording_ids must be a list"):
        PolicyArtifact.from_dict(row)

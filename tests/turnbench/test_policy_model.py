import pytest

from asli.turnbench.auto_schema import (
    REFERENCE_SOURCE,
    DiarBenchCandidate,
    DiarBenchReference,
)
from asli.turnbench.policy_model import (
    calibrate_thresholds,
    fit_policy,
    make_group_split,
)
from asli.turnbench.policy_schema import PolicyFeature, PolicySplit


def make_feature(
    decision_id, *, source, pause_ms=600, energy=0.25, slope=-0.1,
    speech_rate=4.0, semantic_outcome="continue",
):
    semantic_status = "available" if semantic_outcome is not None else "absent"
    return PolicyFeature(
        decision_id=decision_id, recording_id=f"recording-{source}",
        source_recording_id=source, language="Hindi", condition="clean",
        export_fingerprint="e" * 64,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1},
        audio_fingerprint="a" * 64, pause_ms=pause_ms,
        trailing_energy=energy, trailing_energy_slope=slope,
        trailing_speech_ms=740, local_speech_rate_hz=speech_rate,
        semantic_status=semantic_status, semantic_outcome=semantic_outcome,
        semantic_endpoint_offset_ms=100 if semantic_outcome == "yield" else None,
    )


def make_reference(feature, outcome):
    candidate = DiarBenchCandidate(
        decision_id=feature.decision_id, recording_id=feature.recording_id,
        source_recording_id=feature.source_recording_id, audio_path="fixture.wav",
        language=feature.language, condition=feature.condition,
        target_speaker_id="speaker-1", context_start_ms=0,
        previous_speech_end_ms=800, observation_end_ms=feature.pause_ms + 800,
    )
    return DiarBenchReference(candidate, outcome, REFERENCE_SOURCE, None)


def explicit_split(*, train, calibration, test):
    return PolicySplit(
        seed=7, language="Hindi", train_source_recording_ids=train,
        calibration_source_recording_ids=calibration,
        test_source_recording_ids=test,
    )


def test_group_split_never_leaks_a_source_recording():
    """Fails if rows from one source can reach more than one partition."""
    rows = [make_feature(f"d{i}", source=f"source-{i}") for i in range(20)]

    split = make_group_split(rows, language="Hindi", seed=7)

    groups = [
        set(split.train_source_recording_ids),
        set(split.calibration_source_recording_ids),
        set(split.test_source_recording_ids),
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == {f"source-{i}" for i in range(20)}


def test_group_split_is_stable_for_a_seed():
    """Fails if the declared split seed does not fully determine assignment."""
    rows = [make_feature(f"d{i}", source=f"source-{i}") for i in range(20)]

    assert make_group_split(rows, language="Hindi", seed=11) == make_group_split(
        reversed(rows), language="Hindi", seed=11,
    )


def test_group_split_rejects_fewer_than_required_source_groups():
    """Fails if an underpowered language can be treated as a held-out study."""
    rows = [make_feature(f"d{i}", source=f"source-{i}") for i in range(19)]

    with pytest.raises(ValueError, match="at least 20 source recording groups"):
        make_group_split(rows, language="Hindi", seed=7)


def test_fit_uses_train_groups_not_test_labels():
    """Fails if a held-out reference can change the fitted artifact."""
    features = [
        make_feature("train-continue", source="train-a", pause_ms=400, energy=0.1),
        make_feature("train-yield", source="train-b", pause_ms=800, energy=0.4, slope=0.2),
        make_feature("calibration", source="cal-a", pause_ms=650, energy=0.3),
        make_feature("test", source="test-a", pause_ms=900, energy=0.5),
    ]
    split = explicit_split(train=("train-a", "train-b"), calibration=("cal-a",), test=("test-a",))

    first = fit_policy(
        features,
        [
            make_reference(features[0], "continue"), make_reference(features[1], "yield"),
            make_reference(features[2], "continue"), make_reference(features[3], "continue"),
        ],
        split, language="Hindi",
    )
    second = fit_policy(
        features,
        [
            make_reference(features[0], "continue"), make_reference(features[1], "yield"),
            make_reference(features[2], "continue"), make_reference(features[3], "yield"),
        ],
        split, language="Hindi",
    )

    assert first.coefficients == second.coefficients
    assert first.means == second.means
    assert first.scales == second.scales
    assert first.yield_threshold == second.yield_threshold
    assert first.hold_threshold == second.hold_threshold


def test_fit_replaces_zero_train_scales_with_one():
    """Fails if constant train features make normalization invalid."""
    features = [
        make_feature("train-continue", source="train-a", semantic_outcome=None),
        make_feature("train-yield", source="train-b", semantic_outcome=None),
        make_feature("calibration", source="cal-a", semantic_outcome=None),
        make_feature("test", source="test-a", semantic_outcome=None),
    ]
    split = explicit_split(train=("train-a", "train-b"), calibration=("cal-a",), test=("test-a",))

    artifact = fit_policy(
        features,
        [
            make_reference(features[0], "continue"), make_reference(features[1], "yield"),
            make_reference(features[2], "continue"), make_reference(features[3], "yield"),
        ],
        split, language="Hindi",
    )

    assert artifact.scales == (1.0,) * 7


def test_fit_rejects_features_outside_the_declared_split():
    """Fails if an unsplit source can enter fitting or calibration."""
    feature = make_feature("unknown", source="unknown")
    split = explicit_split(train=("train-a",), calibration=("cal-a",), test=("test-a",))

    with pytest.raises(ValueError, match="feature group absent from split: unknown"):
        fit_policy([feature], [], split, language="Hindi")


def test_calibration_uses_the_lowest_thresholds_on_a_utility_tie():
    """Fails if calibration is nondeterministic when several bands score equally."""
    rows = [
        make_feature("continue", source="cal-a"),
        make_feature("yield", source="cal-b"),
    ]
    references = [make_reference(rows[0], "continue"), make_reference(rows[1], "yield")]

    assert calibrate_thresholds(
        {"continue": 0.60, "yield": 0.20}, references, {"continue", "yield"},
    ) == (0.05, 0.55)

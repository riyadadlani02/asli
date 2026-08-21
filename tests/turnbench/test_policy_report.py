from dataclasses import replace

import pytest

from asli.turnbench.auto_schema import REFERENCE_SOURCE, AutoPrediction, DiarBenchCandidate, DiarBenchReference
from asli.turnbench.policy_report import _failed_constraints, replay_policy
from asli.turnbench.policy_schema import POLICY_FEATURE_SCHEMA, PolicyArtifact, PolicyFeature, PolicySplit


def make_feature(decision_id, *, source, pause_ms, condition="clean", energy=0.2):
    return PolicyFeature(
        decision_id=decision_id, recording_id=f"recording-{source}", source_recording_id=source,
        language="Hindi", condition=condition, export_fingerprint="e" * 64,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1,
                          "speech_rate_proxy": "voiced_onsets_per_observed_second.v1"},
        audio_fingerprint="a" * 64, pause_ms=pause_ms, trailing_energy=energy,
        trailing_energy_slope=-0.1, trailing_speech_ms=500, local_speech_rate_hz=4.0,
        semantic_status="absent", semantic_outcome=None, semantic_endpoint_offset_ms=None,
    )


def make_reference(feature, outcome):
    return DiarBenchReference(
        DiarBenchCandidate(
            decision_id=feature.decision_id, recording_id=feature.recording_id,
            source_recording_id=feature.source_recording_id, audio_path="fixture.wav",
            language=feature.language, condition=feature.condition, target_speaker_id="speaker-1",
            context_start_ms=0, previous_speech_end_ms=100, observation_end_ms=feature.pause_ms + 100,
        ),
        outcome, REFERENCE_SOURCE, None,
    )


def make_artifact():
    return PolicyArtifact(
        policy_id="policy-1", language="Hindi", feature_schema=POLICY_FEATURE_SCHEMA,
        export_fingerprint="e" * 64,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1,
                          "speech_rate_proxy": "voiced_onsets_per_observed_second.v1"},
        coefficients=(-5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        means=(0.0,) * 7, scales=(1.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        yield_threshold=0.3, hold_threshold=0.7, grace_ms=150, hard_deadline_ms=800,
        train_source_recording_ids=tuple(f"train-{index}" for index in range(18)),
        calibration_source_recording_ids=("cal",),
    )


def split():
    return PolicySplit(7, "Hindi", tuple(f"train-{index}" for index in range(18)), ("cal",), ("test",))


def study_rows(rows):
    """Add non-held-out rows so replay exercises a complete 20-source study."""
    support = [
        make_feature(f"train-{index}", source=f"train-{index}", pause_ms=1000)
        for index in range(18)
    ]
    support.append(make_feature("cal", source="cal", pause_ms=1000))
    return support + rows


def semantic_rows(features, outcomes):
    rows = []
    for feature, outcome in zip(features, outcomes, strict=True):
        rows.append(AutoPrediction(
            decision_id=feature.decision_id, run_id="semantic-run", agent="local", model="semantic-vad",
            config={"context_ms": 1000}, status="available", outcome=outcome,
            endpoint_ms=150 if outcome == "yield" else None, unavailable_reason=None,
        ))
    return rows


def test_replay_scores_only_test_groups_and_labels_split_counts():
    """Fails if train or calibration decisions contaminate held-out metrics."""
    rows = study_rows([
        make_feature("test-yield", source="test", pause_ms=100, condition="noisy"),
        make_feature("test-continue", source="test", pause_ms=1000),
    ])
    report = replay_policy(rows, [
        make_reference(rows[-4], "continue"), make_reference(rows[-3], "continue"),
        make_reference(rows[-2], "yield"), make_reference(rows[-1], "continue"),
    ], split(), make_artifact(), semantic_predictions=semantic_rows(rows[-2:], ["yield", "yield"]))

    assert report["split_group_counts"] == {"train": 18, "calibration": 1, "test": 1}
    assert report["test"]["eligible_n"] == 2
    assert report["test"]["accuracy"] == 1.0
    assert report["by_condition"]["noisy"]["eligible_n"] == 1


def test_replay_rejects_declared_ghost_source_groups_before_a_report():
    """Fails if absent split provenance can create an underpowered replay report."""
    rows = [
        make_feature("train", source="train", pause_ms=1000),
        make_feature("cal", source="cal", pause_ms=1000),
        make_feature("test", source="test", pause_ms=100),
    ]
    ghost_split = PolicySplit(7, "Hindi", ("train", "ghost-train"), ("cal",), ("test",))
    ghost_artifact = replace(make_artifact(), train_source_recording_ids=("train", "ghost-train"))

    with pytest.raises(ValueError, match="feature source recording IDs must exactly match split"):
        replay_policy(
            rows, [make_reference(row, "yield") for row in rows], ghost_split,
            ghost_artifact, semantic_predictions=semantic_rows(rows[2:], ["yield"]),
        )


def test_replay_rejects_an_exact_but_underpowered_study_before_a_report():
    """Fails if a complete three-way split below 20 sources can produce any report."""
    rows = [
        make_feature("train", source="train", pause_ms=1000),
        make_feature("cal", source="cal", pause_ms=1000),
        make_feature("test", source="test", pause_ms=100),
    ]
    small_split = PolicySplit(7, "Hindi", ("train",), ("cal",), ("test",))
    small_artifact = replace(
        make_artifact(), train_source_recording_ids=("train",), calibration_source_recording_ids=("cal",),
    )

    with pytest.raises(ValueError, match="replay requires 20 independent source recordings"):
        replay_policy(
            rows, [make_reference(row, "yield") for row in rows], small_split,
            small_artifact, semantic_predictions=semantic_rows(rows[2:], ["yield"]),
        )


def test_missing_semantic_baseline_reports_always_yield_utility_failure_too():
    """Fails if semantic absence hides a separate always-yield utility failure."""
    assert _failed_constraints(
        {"continuation_recall": 1.0, "unnecessary_hold_rate": 0.0, "coverage_rate": 1.0, "utility": 0.0},
        {"utility": 0.0},
        None,
    ) == ["utility_over_always_yield", "semantic_baseline"]


def test_high_accuracy_policy_is_not_a_win_when_it_interrupts_too_often():
    """Fails if raw accuracy can override the continuation-recall gate."""
    rows = study_rows([make_feature(f"test-{index}", source="test", pause_ms=100) for index in range(10)])
    test_rows = rows[-10:]
    references = [make_reference(feature, "continue" if index < 1 else "yield") for index, feature in enumerate(test_rows)]

    report = replay_policy(rows, references, split(), make_artifact(), semantic_predictions=semantic_rows(test_rows, ["yield"] * 10))

    assert report["test"]["accuracy"] == 0.9
    assert report["policy_win"] is False
    assert "continuation_recall" in report["failed_constraints"]


def test_missing_complete_semantic_baseline_prevents_a_policy_win():
    """Fails if a policy can win without the required semantic comparison."""
    rows = study_rows([
        make_feature("test-yield", source="test", pause_ms=100),
        make_feature("test-continue", source="test", pause_ms=1000),
    ])
    report = replay_policy(rows, [make_reference(rows[-2], "yield"), make_reference(rows[-1], "continue")], split(), make_artifact())

    assert report["policy_win"] is False
    assert "semantic_baseline" in report["failed_constraints"]


def test_replay_requires_complete_compatible_semantic_predictions():
    """Fails if a partial semantic run can be used as a favorable baseline."""
    rows = study_rows([
        make_feature("test-yield", source="test", pause_ms=100),
        make_feature("test-continue", source="test", pause_ms=1000),
    ])
    with pytest.raises(ValueError, match="semantic prediction IDs must exactly match test features"):
        replay_policy(rows, [make_reference(rows[-2], "yield"), make_reference(rows[-1], "continue")], split(), make_artifact(), semantic_predictions=semantic_rows(rows[-2:-1], ["yield"]))


def test_replay_rejects_a_held_out_binary_reference_without_a_feature():
    """Fails if an unmatched held-out label can be excluded before a win decision."""
    feature = make_feature("test-yield", source="test", pause_ms=100)
    missing_feature = make_feature("test-missing", source="test", pause_ms=100)

    with pytest.raises(ValueError, match="held-out binary reference IDs do not match test features"):
        replay_policy(
            study_rows([feature]), [make_reference(feature, "yield"), make_reference(missing_feature, "yield")],
            split(), make_artifact(), semantic_predictions=semantic_rows([feature], ["yield"]),
        )


@pytest.mark.parametrize("leaking_source", ["train", "cal"])
def test_replay_rejects_semantic_predictions_outside_the_held_out_test_set(leaking_source):
    """Fails if train or calibration semantic rows can enter a held-out baseline."""
    test_feature = make_feature("test-yield", source="test", pause_ms=100)
    leaking_feature = make_feature(f"{leaking_source}-yield", source=f"{leaking_source}-0" if leaking_source == "train" else "cal", pause_ms=100)

    with pytest.raises(ValueError, match="semantic prediction IDs must exactly match test features"):
        replay_policy(
            study_rows([test_feature, leaking_feature]),
            [make_reference(test_feature, "yield"), make_reference(leaking_feature, "yield")],
            split(), make_artifact(),
            semantic_predictions=semantic_rows([test_feature, leaking_feature], ["yield", "yield"]),
        )


def test_insufficient_runtime_coverage_prevents_a_policy_win():
    """Fails if an unavailable held-out policy decision does not fail coverage."""
    rows = study_rows([
        make_feature("test-yield", source="test", pause_ms=100, energy=1e308),
        make_feature("test-continue", source="test", pause_ms=1000),
    ])
    artifact = replace(make_artifact(), coefficients=(-5.0, 10.0, 1e308, 0.0, 0.0, 0.0, 0.0))
    report = replay_policy(
        rows, [make_reference(rows[-2], "yield"), make_reference(rows[-1], "continue")],
        split(), artifact, semantic_predictions=semantic_rows(rows[-2:], ["yield", "yield"]),
    )

    assert report["test"]["coverage_rate"] == 0.5
    assert report["policy_win"] is False
    assert "coverage_rate" in report["failed_constraints"]


def test_replay_fails_closed_when_reference_identity_does_not_match_feature():
    """Fails if same IDs with different source metadata can be scored together."""
    row = make_feature("test", source="test", pause_ms=1000)
    reference = make_reference(row, "continue")
    candidate = DiarBenchCandidate(
        decision_id="test", recording_id="other-recording", source_recording_id="test", audio_path="fixture.wav",
        language="Hindi", condition="clean", target_speaker_id="speaker-1", context_start_ms=0,
        previous_speech_end_ms=100, observation_end_ms=1100,
    )
    mismatched = DiarBenchReference(candidate, "continue", REFERENCE_SOURCE, None)

    with pytest.raises(ValueError, match="feature/reference metadata mismatch: test"):
        replay_policy(study_rows([row]), [mismatched], split(), make_artifact(), semantic_predictions=semantic_rows([row], ["yield"]))


def test_held_out_policy_win_requires_every_gate_and_strictly_beats_baselines():
    """Fails if a safe held-out policy win omits any baseline comparison."""
    rows = study_rows([
        make_feature("test-yield", source="test", pause_ms=100),
        make_feature("test-continue", source="test", pause_ms=1000),
    ])
    report = replay_policy(
        rows, [make_reference(rows[-2], "yield"), make_reference(rows[-1], "continue")],
        split(), make_artifact(), semantic_predictions=semantic_rows(rows[-2:], ["yield", "yield"]),
    )

    assert report["test"]["continuation_recall"] == 1.0
    assert report["test"]["unnecessary_hold_rate"] == 0.0
    assert report["test"]["coverage_rate"] == 1.0
    assert report["policy_win"] is True
    assert report["failed_constraints"] == []

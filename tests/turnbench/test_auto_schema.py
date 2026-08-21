import pytest

from asli.turnbench import (
    AutoPrediction,
    DiarBenchCandidate,
    DiarBenchReference,
    SchemaError,
)


def candidate_row():
    return {
        "schema": "turnbench.diarbench.candidate.v1",
        "decision_id": "sample-1:0000",
        "recording_id": "sample-1",
        "source_recording_id": "sample-1",
        "audio_path": "audio/sample-1.wav",
        "language": "Hindi",
        "condition": "diarbench",
        "target_speaker_id": "caller",
        "context_start_ms": 0,
        "previous_speech_end_ms": 1000,
        "observation_end_ms": 1600,
    }


def test_candidate_rejects_reference_outcome_leak():
    row = candidate_row()
    row["outcome"] = "continue"
    with pytest.raises(SchemaError, match="unknown field: outcome"):
        DiarBenchCandidate.from_dict(row)


@pytest.mark.parametrize("field", ["next_speaker_id", "reference", "unknown"])
def test_candidate_rejects_all_non_contract_fields(field):
    row = candidate_row()
    row[field] = "leak"
    with pytest.raises(SchemaError, match=fr"unknown field: {field}"):
        DiarBenchCandidate.from_dict(row)


def test_candidate_requires_a_nonempty_observation_window():
    row = candidate_row()
    row["context_start_ms"] = 1001
    with pytest.raises(SchemaError, match="context_start_ms"):
        DiarBenchCandidate.from_dict(row)


def test_reference_exposes_candidates_only_for_binary_observed_continuations():
    candidate = DiarBenchCandidate.from_dict(candidate_row())
    binary = DiarBenchReference(candidate, "continue", "indic_diarbench_human_timing.v1", None)
    excluded = DiarBenchReference(candidate, "overlap", "indic_diarbench_human_timing.v1", "mixed speakers")

    assert binary.as_candidate() == candidate
    assert excluded.as_candidate() is None


def test_reference_round_trips_its_flat_wire_contract():
    candidate = DiarBenchCandidate.from_dict(candidate_row())
    reference = DiarBenchReference(
        candidate, "yield", "indic_diarbench_human_timing.v1", None
    )

    assert DiarBenchReference.from_dict(reference.to_dict()) == reference


def test_reference_rejects_a_non_reference_schema():
    candidate = DiarBenchCandidate.from_dict(candidate_row())
    reference = DiarBenchReference(
        candidate, "yield", "indic_diarbench_human_timing.v1", None
    ).to_dict()
    reference["schema"] = "turnbench.diarbench.candidate.v1"

    with pytest.raises(SchemaError, match="schema must be turnbench.diarbench.reference.v1"):
        DiarBenchReference.from_dict(reference)


def test_prediction_requires_binary_outcome_only_when_available():
    prediction = AutoPrediction(
        decision_id="sample-1:0000",
        run_id="run-1",
        agent="openai",
        model="model",
        config={"context_ms": 5000},
        status="unavailable",
        outcome=None,
        endpoint_ms=None,
        unavailable_reason="provider_error",
    )
    assert AutoPrediction.from_dict(prediction.to_dict()) == prediction

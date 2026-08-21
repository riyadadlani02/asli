import pytest

from asli.turnbench import Annotation, DecisionLabel, Recording, SourceTurn


@pytest.fixture
def recording():
    return Recording(
        recording_id="r1",
        audio_path="audio/r1.wav",
        language="hi",
        condition="telephone",
        source_recording_id="src1",
        target_speaker_id="caller",
        turns=(
            SourceTurn("caller", 0, 1000, "hello"),
            SourceTurn("agent", 1600, 1900, None),
        ),
    )


@pytest.fixture
def label_row():
    return {
        "schema": "turnbench.label.v1",
        "decision_id": "d1",
        "recording_id": "r1",
        "source_recording_id": "src1",
        "target_speaker_id": "caller",
        "previous_speech_end_ms": 1000,
        "next_event_start_ms": 1600,
        "outcome": "continue",
        "true_end_ms": None,
        "exclusion_reason": None,
        "final_label": "fixture",
        "annotations": [],
        "continuation_resume_ms": 1800,
    }

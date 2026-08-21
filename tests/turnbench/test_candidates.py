from dataclasses import replace

import pytest

from asli.turnbench import PauseCandidate, SourceTurn, extract_candidates
from asli.turnbench.schema import Recording


@pytest.fixture
def overlap_recording(recording):
    return Recording(
        recording_id=recording.recording_id,
        audio_path=recording.audio_path,
        language=recording.language,
        condition=recording.condition,
        source_recording_id=recording.source_recording_id,
        target_speaker_id=recording.target_speaker_id,
        turns=(
            SourceTurn("caller", 0, 1000, "hello"),
            SourceTurn("agent", 900, 1400, "interruption"),
        ),
    )


def test_extracts_only_gaps_in_requested_window(recording):
    candidates = extract_candidates(recording, min_pause_ms=300, max_pause_ms=1000)
    assert [
        (item.previous_speech_end_ms, item.next_event_start_ms)
        for item in candidates
    ] == [(1000, 1600)]


def test_extractor_keeps_next_speaker_without_labelling_outcome(recording):
    candidate = extract_candidates(
        recording, min_pause_ms=300, max_pause_ms=1000
    )[0]
    assert candidate.next_speaker_id == "agent"
    assert not hasattr(candidate, "outcome")
    assert isinstance(candidate, PauseCandidate)


def test_active_overlapping_speech_prevents_a_silence_candidate(
    overlap_recording,
):
    assert (
        extract_candidates(overlap_recording, min_pause_ms=300, max_pause_ms=1000)
        == []
    )


def test_active_same_speaker_segment_prevents_a_silence_candidate(recording):
    self_overlap = Recording(
        recording_id=recording.recording_id,
        audio_path=recording.audio_path,
        language=recording.language,
        condition=recording.condition,
        source_recording_id=recording.source_recording_id,
        target_speaker_id=recording.target_speaker_id,
        turns=(
            SourceTurn("caller", 0, 1000, "first segment"),
            SourceTurn("caller", 900, 1400, "overlapping segment"),
            SourceTurn("agent", 2000, 2200, "next event"),
        ),
    )

    candidates = extract_candidates(
        self_overlap, min_pause_ms=500, max_pause_ms=1000
    )

    assert [item.previous_speech_end_ms for item in candidates] == [1400]


def test_gap_equal_to_minimum_is_included(recording):
    assert len(extract_candidates(recording, min_pause_ms=600, max_pause_ms=600)) == 1


def test_gap_equal_to_maximum_is_included(recording):
    assert len(extract_candidates(recording, min_pause_ms=0, max_pause_ms=600)) == 1


def test_invalid_window_is_rejected(recording):
    with pytest.raises(ValueError, match="min_pause_ms"):
        extract_candidates(recording, min_pause_ms=700, max_pause_ms=600)


def test_zero_duration_target_turn_cannot_select_itself(recording):
    zero_duration = Recording(
        recording_id=recording.recording_id,
        audio_path=recording.audio_path,
        language=recording.language,
        condition=recording.condition,
        source_recording_id=recording.source_recording_id,
        target_speaker_id=recording.target_speaker_id,
        turns=(
            SourceTurn("caller", 0, 0),
            SourceTurn("agent", 100, 200),
        ),
    )
    candidates = extract_candidates(zero_duration, min_pause_ms=0, max_pause_ms=100)
    assert len(candidates) == 1
    assert candidates[0].previous_speech_end_ms == 0
    assert candidates[0].next_event_start_ms == 100
    assert candidates[0].next_speaker_id == "agent"


def test_co_starting_speakers_do_not_select_one_by_source_order(recording):
    co_starting = replace(
        recording,
        turns=(
            recording.turns[0],
            SourceTurn("caller", 1600, 1700),
            recording.turns[1],
        ),
    )

    candidate = extract_candidates(
        co_starting, min_pause_ms=300, max_pause_ms=1000
    )[0]

    assert candidate.next_event_start_ms == 1600
    assert candidate.next_speaker_id is None

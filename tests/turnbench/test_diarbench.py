from asli.turnbench import convert_diarbench_sample


def test_conversion_separates_binary_candidates_from_overlap_references():
    row = {
        "sample_id": "sample-1",
        "language": "Hindi",
        "segments": [
            {"speaker_id": "caller", "start": 0.0, "end": 1.0},
            {"speaker_id": "caller", "start": 1.6, "end": 2.2},
            {"speaker_id": "other", "start": 3.0, "end": 3.4},
            {"speaker_id": "caller", "start": 3.6, "end": 3.9},
            {"speaker_id": "other", "start": 3.85, "end": 4.1},
        ],
    }

    candidates, references = convert_diarbench_sample(
        row,
        min_pause_ms=300,
        max_pause_ms=1000,
        audio_path="audio/sample-1.wav",
        context_ms=5000,
    )

    assert [(item.candidate.target_speaker_id, item.outcome) for item in references] == [
        ("caller", "continue"),
        ("caller", "yield"),
        ("other", "yield"),
        ("caller", "overlap"),
        ("other", "unclear"),
    ]
    assert [(item.target_speaker_id, item.previous_speech_end_ms) for item in candidates] == [
        ("caller", 1000),
        ("caller", 2200),
    ]
    assert references[-2].exclusion_reason == "active_or_co_start_mixed_speakers"
    assert all(item.decision_id == reference.candidate.decision_id for item, reference in zip(candidates, references[:2]))


def test_conversion_rounds_seconds_and_preserves_unclear_references():
    candidates, references = convert_diarbench_sample(
        {
            "sample_id": "sample-2",
            "language": "Hindi",
            "segments": [{"speaker_id": "caller", "start": 0.0005, "end": 1.0005}],
        },
        min_pause_ms=0,
        max_pause_ms=1000,
        audio_path="audio/sample-2.wav",
        context_ms=500,
    )

    assert candidates == []
    assert references[0].outcome == "unclear"
    assert references[0].candidate.previous_speech_end_ms == 1000


def test_conversion_marks_an_active_crossing_segment_as_overlap():
    _, references = convert_diarbench_sample(
        {
            "sample_id": "sample-3",
            "language": "Hindi",
            "segments": [
                {"speaker_id": "caller", "start": 0.0, "end": 1.0},
                {"speaker_id": "caller", "start": 0.9, "end": 1.5},
            ],
        },
        min_pause_ms=0,
        max_pause_ms=1000,
        audio_path="audio/sample-3.wav",
        context_ms=500,
    )

    assert references[0].outcome == "overlap"


def test_conversion_retains_a_zero_gap_reference_without_a_candidate():
    candidates, references = convert_diarbench_sample(
        {
            "sample_id": "sample-4",
            "language": "Hindi",
            "segments": [
                {"speaker_id": "caller", "start": 0.0, "end": 1.0},
                {"speaker_id": "caller", "start": 1.0, "end": 2.0},
            ],
        },
        min_pause_ms=0,
        max_pause_ms=1000,
        audio_path="audio/sample-4.wav",
        context_ms=500,
    )

    assert candidates == []
    assert references[0].outcome == "continue"

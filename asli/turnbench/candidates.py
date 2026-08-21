"""Timing-only extraction of pauses that can be sent for annotation.

This module deliberately does not inspect transcript text or assign a semantic
meaning to a pause.  Labels such as ``continue`` and ``yield`` belong to the
human annotation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Recording


@dataclass(frozen=True)
class PauseCandidate:
    """A target-speaker pause with its next timed speech event."""

    decision_id: str
    recording_id: str
    source_recording_id: str
    target_speaker_id: str
    previous_speech_end_ms: int
    next_event_start_ms: int
    next_speaker_id: str | None


@dataclass(frozen=True)
class MechanicalBoundary:
    """One source-indexed target turn and its earliest following event."""

    previous_turn_index: int
    next_turn_indices: tuple[int, ...]
    next_event_start_ms: int


def resolve_mechanical_boundary(
    recording: Recording,
    *,
    target_speaker_id: str,
    previous_speech_end_ms: int,
    next_event_start_ms: int | None = None,
    previous_turn_index: int | None = None,
) -> MechanicalBoundary:
    """Resolve and validate a real silent boundary from source-turn indices.

    Linked labels omit a source-turn ID, so their preceding target turn must be
    uniquely identifiable by speaker and end timestamp.  Extraction already
    has the exact source index and supplies it explicitly.  In both paths, a
    distinct segment active strictly across the boundary makes the interval
    non-silent, and the next event is the earliest distinct source-turn start
    at or after that boundary.
    """

    if previous_turn_index is None:
        matching_previous = [
            index
            for index, turn in enumerate(recording.turns)
            if turn.speaker_id == target_speaker_id
            and turn.end_ms == previous_speech_end_ms
        ]
        if not matching_previous:
            raise ValueError(
                "previous_speech_end_ms does not match a target-speaker turn"
            )
        if len(matching_previous) > 1:
            raise ValueError("ambiguous preceding target-speaker turn")
        previous_turn_index = matching_previous[0]
    else:
        if not 0 <= previous_turn_index < len(recording.turns):
            raise ValueError("previous_turn_index is outside recording turns")
        previous_turn = recording.turns[previous_turn_index]
        if (
            previous_turn.speaker_id != target_speaker_id
            or previous_turn.end_ms != previous_speech_end_ms
        ):
            raise ValueError(
                "previous_turn_index does not match the requested target boundary"
            )

    active_distinct_turn = any(
        index != previous_turn_index
        and turn.start_ms < previous_speech_end_ms < turn.end_ms
        for index, turn in enumerate(recording.turns)
    )
    if active_distinct_turn:
        raise ValueError("preceding target turn does not end at a silent boundary")

    following_turns = [
        (index, turn)
        for index, turn in enumerate(recording.turns)
        if index != previous_turn_index
        and turn.start_ms >= previous_speech_end_ms
    ]
    if not following_turns:
        raise ValueError("no distinct source turn starts after the boundary")

    earliest_start_ms = min(
        turn.start_ms for _, turn in following_turns
    )
    next_turn_indices = tuple(
        index
        for index, turn in following_turns
        if turn.start_ms == earliest_start_ms
    )
    if (
        next_event_start_ms is not None
        and next_event_start_ms != earliest_start_ms
    ):
        raise ValueError(
            "next_event_start_ms is not the earliest distinct source-turn start"
        )
    return MechanicalBoundary(
        previous_turn_index=previous_turn_index,
        next_turn_indices=next_turn_indices,
        next_event_start_ms=earliest_start_ms,
    )


def extract_candidates(
    recording: Recording, *, min_pause_ms: int, max_pause_ms: int
) -> list[PauseCandidate]:
    """Extract target-speaker gaps whose duration falls in the requested window.

    A target turn ending inside any distinct active source segment is excluded:
    the interval is overlap, not silence.  The extractor only uses timestamps
    and speaker IDs; semantic labels are assigned later by annotators.
    """

    if min_pause_ms < 0 or max_pause_ms < min_pause_ms:
        raise ValueError("require 0 <= min_pause_ms <= max_pause_ms")

    candidates: list[PauseCandidate] = []
    for index, turn in enumerate(recording.turns):
        if turn.speaker_id != recording.target_speaker_id:
            continue

        try:
            boundary = resolve_mechanical_boundary(
                recording,
                target_speaker_id=recording.target_speaker_id,
                previous_speech_end_ms=turn.end_ms,
                previous_turn_index=index,
            )
        except ValueError:
            continue

        next_speakers = {
            recording.turns[next_index].speaker_id
            for next_index in boundary.next_turn_indices
        }
        next_speaker_id = (
            next(iter(next_speakers)) if len(next_speakers) == 1 else None
        )

        pause_ms = boundary.next_event_start_ms - turn.end_ms
        if min_pause_ms <= pause_ms <= max_pause_ms:
            candidates.append(
                PauseCandidate(
                    decision_id=f"{recording.recording_id}:{index}:{turn.end_ms}",
                    recording_id=recording.recording_id,
                    source_recording_id=recording.source_recording_id,
                    target_speaker_id=recording.target_speaker_id,
                    previous_speech_end_ms=turn.end_ms,
                    next_event_start_ms=boundary.next_event_start_ms,
                    next_speaker_id=next_speaker_id,
                )
            )

    return candidates

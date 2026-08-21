"""Pure conversion of DiarBench timing annotations into TurnBench records."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .auto_schema import DiarBenchCandidate, DiarBenchReference, REFERENCE_SOURCE
from .schema import SchemaError


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field} must be a non-empty string")
    return value


def _milliseconds(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise SchemaError(f"{field} must be a non-negative finite seconds value")
    return int(round(value * 1000))


def convert_diarbench_sample(
    row: Mapping[str, object],
    *,
    min_pause_ms: int,
    max_pause_ms: int,
    audio_path: str,
    context_ms: int,
) -> tuple[list[DiarBenchCandidate], list[DiarBenchReference]]:
    """Build timing references without exposing timing outcomes to candidates."""

    if not isinstance(row, Mapping):
        raise SchemaError("DiarBench sample must be an object")
    if isinstance(min_pause_ms, bool) or not isinstance(min_pause_ms, int) or min_pause_ms < 0:
        raise ValueError("min_pause_ms must be a non-negative integer")
    if isinstance(max_pause_ms, bool) or not isinstance(max_pause_ms, int) or max_pause_ms < min_pause_ms:
        raise ValueError("max_pause_ms must be an integer greater than or equal to min_pause_ms")
    if isinstance(context_ms, bool) or not isinstance(context_ms, int) or context_ms < 0:
        raise ValueError("context_ms must be a non-negative integer")

    sample_id = _text(row.get("sample_id"), "sample_id")
    source_recording_id = _text(row.get("recording_id", sample_id), "recording_id")
    language = _text(row.get("language"), "language")
    condition = _text(row.get("condition", "diarbench"), "condition")
    audio_path = _text(audio_path, "audio_path")
    raw_segments = row.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise SchemaError("segments must be a non-empty list")

    segments: list[tuple[str, int, int]] = []
    last_start = -1
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise SchemaError(f"segments[{index}] must be an object")
        speaker = _text(raw.get("speaker_id"), f"segments[{index}].speaker_id")
        start = _milliseconds(raw.get("start"), f"segments[{index}].start")
        end = _milliseconds(raw.get("end"), f"segments[{index}].end")
        if end < start:
            raise SchemaError(f"segments[{index}].end must be greater than or equal to start")
        if start < last_start:
            raise SchemaError("segments must be nondecreasing by start time")
        last_start = start
        segments.append((speaker, start, end))

    candidates: list[DiarBenchCandidate] = []
    references: list[DiarBenchReference] = []
    for index, (speaker, start, end) in enumerate(segments):
        decision_id = f"{sample_id}:{index:04d}"
        crossing = [
            other for other_index, other in enumerate(segments)
            if other_index != index and other[1] < end < other[2]
        ]
        following_starts = [other[1] for other_index, other in enumerate(segments) if other_index != index and other[1] >= end]
        next_start = min(following_starts) if following_starts else None
        earliest = [other for other_index, other in enumerate(segments) if other_index != index and other[1] == next_start] if next_start is not None else []
        observation_end = max(end + 1, next_start if next_start is not None else end + 1)
        candidate = DiarBenchCandidate(
            decision_id=decision_id,
            recording_id=sample_id,
            source_recording_id=source_recording_id,
            audio_path=audio_path,
            language=language,
            condition=condition,
            target_speaker_id=speaker,
            context_start_ms=max(0, end - context_ms),
            previous_speech_end_ms=end,
            observation_end_ms=observation_end,
        )
        if crossing or (earliest and {item[0] == speaker for item in earliest} == {True, False}):
            outcome, reason = "overlap", "active_or_co_start_mixed_speakers"
        elif not earliest:
            outcome, reason = "unclear", "no_reliable_next_event"
        elif all(item[0] == speaker for item in earliest):
            outcome, reason = "continue", None
        else:
            outcome, reason = "yield", None
        reference = DiarBenchReference(candidate, outcome, REFERENCE_SOURCE, reason)
        references.append(reference)
        if outcome in {"continue", "yield"} and next_start is not None:
            gap = next_start - end
            if 0 < gap and min_pause_ms <= gap <= max_pause_ms:
                candidates.append(candidate)

    return (
        sorted(candidates, key=lambda item: item.decision_id),
        sorted(references, key=lambda item: item.candidate.decision_id),
    )

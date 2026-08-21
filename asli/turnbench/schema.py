"""Strict, versioned JSONL records for TurnBench.

The records intentionally have no dependency on the rest of ASLI.  Validation is
performed at the boundary so malformed corpus, annotation, or provider data cannot
silently enter a benchmark run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar


class SchemaError(ValueError):
    """Raised when a TurnBench record does not satisfy its exchange schema."""


T = TypeVar("T")

# Public top-level wire-format versions for adapters and consumers.
RECORDING_SCHEMA = "turnbench.recording.v1"
LABEL_SCHEMA = "turnbench.label.v1"
EVENT_SCHEMA = "turnbench.event.v1"


def _mapping(row: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise SchemaError("record must be an object")
    return row


def _fields(
    row: Mapping[str, object], required: set[str], optional: set[str] | None = None
) -> None:
    optional = optional or set()
    extra = set(row) - required - optional
    missing = required - set(row)
    if missing:
        raise SchemaError(f"missing field: {sorted(missing)[0]}")
    if extra:
        raise SchemaError(f"unknown field: {sorted(extra)[0]}")


def _text(row: Mapping[str, object], name: str, *, nonempty: bool = False) -> str:
    value = row[name]
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise SchemaError(f"{name} must be a non-empty string" if nonempty else f"{name} must be a string")
    return value


def _timestamp(row: Mapping[str, object], name: str, *, allow_none: bool = False) -> int | None:
    value = row[name]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer timestamp")
    return value


def _validate_text(value: object, name: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        requirement = "a non-empty string" if nonempty else "a string"
        raise SchemaError(f"{name} must be {requirement}")


def _validate_timestamp(value: object, name: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer timestamp")


@dataclass(frozen=True)
class SourceTurn:
    speaker_id: str
    start_ms: int
    end_ms: int
    transcript: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_text(self.speaker_id, "speaker_id", nonempty=True)
        _validate_timestamp(self.start_ms, "start_ms")
        _validate_timestamp(self.end_ms, "end_ms")
        if self.end_ms < self.start_ms:
            raise SchemaError("end_ms must be greater than or equal to start_ms")
        if self.transcript is not None and not isinstance(self.transcript, str):
            raise SchemaError("transcript must be a string or null")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "SourceTurn":
        row = _mapping(row)
        _fields(row, {"speaker_id", "start_ms", "end_ms"}, {"transcript"})
        speaker = _text(row, "speaker_id", nonempty=True)
        start = _timestamp(row, "start_ms")
        end = _timestamp(row, "end_ms")
        if end < start:
            raise SchemaError("end_ms must be greater than or equal to start_ms")
        transcript = row.get("transcript")
        if transcript is not None and not isinstance(transcript, str):
            raise SchemaError("transcript must be a string or null")
        return cls(speaker, start, end, transcript)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"speaker_id": self.speaker_id, "start_ms": self.start_ms, "end_ms": self.end_ms}
        if self.transcript is not None:
            result["transcript"] = self.transcript
        return result


@dataclass(frozen=True)
class Recording:
    recording_id: str
    audio_path: str
    language: str
    condition: str
    source_recording_id: str
    target_speaker_id: str
    turns: tuple[SourceTurn, ...]

    schema = RECORDING_SCHEMA

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "recording_id",
            "audio_path",
            "language",
            "condition",
            "source_recording_id",
            "target_speaker_id",
        ):
            _validate_text(getattr(self, name), name, nonempty=True)
        if not isinstance(self.turns, tuple):
            raise SchemaError("turns must be a tuple of SourceTurn records")
        for turn in self.turns:
            if not isinstance(turn, SourceTurn):
                raise SchemaError("turns must contain SourceTurn records")
            turn.validate()
        if list(self.turns) != sorted(
            self.turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_id)
        ):
            raise SchemaError("turns must be sorted by (start_ms, end_ms, speaker_id)")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "Recording":
        row = _mapping(row)
        _fields(row, {"schema", "recording_id", "audio_path", "language", "condition", "source_recording_id", "target_speaker_id", "turns"})
        if row["schema"] != cls.schema:
            raise SchemaError(f"schema must be {cls.schema}")
        values = {name: _text(row, name, nonempty=True) for name in ("recording_id", "audio_path", "language", "condition", "source_recording_id", "target_speaker_id")}
        raw_turns = row["turns"]
        if not isinstance(raw_turns, list):
            raise SchemaError("turns must be a list")
        turns = tuple(SourceTurn.from_dict(turn) for turn in raw_turns)
        if list(turns) != sorted(turns, key=lambda t: (t.start_ms, t.end_ms, t.speaker_id)):
            raise SchemaError("turns must be sorted by (start_ms, end_ms, speaker_id)")
        return cls(**values, turns=turns)

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "recording_id": self.recording_id, "audio_path": self.audio_path, "language": self.language, "condition": self.condition, "source_recording_id": self.source_recording_id, "target_speaker_id": self.target_speaker_id, "turns": [turn.to_dict() for turn in self.turns]}


@dataclass(frozen=True)
class Annotation:
    annotator_id: str
    label: str
    note: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_text(self.annotator_id, "annotator_id", nonempty=True)
        _validate_text(self.label, "annotation label", nonempty=True)
        if self.label not in _OUTCOMES:
            raise SchemaError("annotation label must be continue, yield, overlap, or unclear")
        if self.note is not None and not isinstance(self.note, str):
            raise SchemaError("note must be a string or null")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "Annotation":
        row = _mapping(row)
        _fields(row, {"annotator_id", "label"}, {"note"})
        annotator = _text(row, "annotator_id", nonempty=True)
        label = _text(row, "label", nonempty=True)
        note = row.get("note")
        if note is not None and not isinstance(note, str):
            raise SchemaError("note must be a string or null")
        return cls(annotator, label, note)

    def to_dict(self) -> dict[str, object]:
        result = {"annotator_id": self.annotator_id, "label": self.label}
        if self.note is not None:
            result["note"] = self.note
        return result


_OUTCOMES = {"continue", "yield", "overlap", "unclear"}
_FINAL_LABELS = _OUTCOMES | {"fixture"}


@dataclass(frozen=True)
class DecisionLabel:
    decision_id: str
    recording_id: str
    source_recording_id: str
    target_speaker_id: str
    previous_speech_end_ms: int
    next_event_start_ms: int
    outcome: str
    true_end_ms: int | None
    exclusion_reason: str | None
    final_label: str
    annotations: tuple[Annotation, ...]
    continuation_resume_ms: int | None = None

    schema = LABEL_SCHEMA

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "decision_id",
            "recording_id",
            "source_recording_id",
            "target_speaker_id",
        ):
            _validate_text(getattr(self, name), name, nonempty=True)
        _validate_timestamp(self.previous_speech_end_ms, "previous_speech_end_ms")
        _validate_timestamp(self.next_event_start_ms, "next_event_start_ms")
        _validate_timestamp(
            self.true_end_ms, "true_end_ms", allow_none=True
        )
        _validate_timestamp(
            self.continuation_resume_ms,
            "continuation_resume_ms",
            allow_none=True,
        )
        _validate_text(self.outcome, "outcome", nonempty=True)
        if self.outcome not in _OUTCOMES:
            raise SchemaError("outcome must be continue, yield, overlap, or unclear")
        _validate_text(self.final_label, "final_label", nonempty=True)
        if self.final_label not in _FINAL_LABELS:
            raise SchemaError("final_label is invalid")
        if not isinstance(self.annotations, tuple):
            raise SchemaError("annotations must be a tuple of Annotation records")
        for annotation in self.annotations:
            if not isinstance(annotation, Annotation):
                raise SchemaError("annotations must contain Annotation records")
            annotation.validate()
        if self.exclusion_reason is not None and (
            not isinstance(self.exclusion_reason, str)
            or not self.exclusion_reason.strip()
        ):
            raise SchemaError("exclusion_reason must be a non-empty string or null")

        if self.final_label == "fixture":
            if self.annotations:
                raise SchemaError("fixture labels require no annotations")
        else:
            if self.final_label != self.outcome:
                raise SchemaError("final_label must equal outcome for non-fixture labels")
            if len(self.annotations) != 2:
                raise SchemaError("non-fixture labels require exactly two annotations")
            if len({item.annotator_id for item in self.annotations}) != 2:
                raise SchemaError("annotations must come from two annotators")

        if self.outcome == "continue":
            if self.continuation_resume_ms is None:
                raise SchemaError("continue requires continuation_resume_ms")
            if self.continuation_resume_ms != self.next_event_start_ms:
                raise SchemaError(
                    "continue requires continuation_resume_ms == next_event_start_ms"
                )
            if self.continuation_resume_ms <= self.previous_speech_end_ms:
                raise SchemaError(
                    "continue requires continuation_resume_ms after previous_speech_end_ms"
                )
            if self.true_end_ms is not None:
                raise SchemaError("continue requires true_end_ms to be null")
            if self.exclusion_reason is not None:
                raise SchemaError("continue requires exclusion_reason to be null")
        elif self.outcome == "yield":
            if self.true_end_ms is None:
                raise SchemaError("yield requires true_end_ms")
            if self.true_end_ms > self.previous_speech_end_ms:
                raise SchemaError(
                    "yield requires true_end_ms at or before previous_speech_end_ms"
                )
            if self.continuation_resume_ms is not None:
                raise SchemaError("yield requires continuation_resume_ms to be null")
            if self.exclusion_reason is not None:
                raise SchemaError("yield requires exclusion_reason to be null")
        else:
            if self.exclusion_reason is None:
                raise SchemaError(f"{self.outcome} requires exclusion_reason")
            if self.continuation_resume_ms is not None:
                raise SchemaError(
                    f"{self.outcome} requires continuation_resume_ms to be null"
                )
            if self.true_end_ms is not None:
                raise SchemaError(f"{self.outcome} requires true_end_ms to be null")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "DecisionLabel":
        row = _mapping(row)
        required = {"schema", "decision_id", "recording_id", "source_recording_id", "target_speaker_id", "previous_speech_end_ms", "next_event_start_ms", "outcome", "true_end_ms", "exclusion_reason", "final_label", "annotations"}
        _fields(row, required, {"continuation_resume_ms"})
        if row["schema"] != cls.schema:
            raise SchemaError(f"schema must be {cls.schema}")
        ids = {name: _text(row, name, nonempty=True) for name in ("decision_id", "recording_id", "source_recording_id", "target_speaker_id")}
        previous = _timestamp(row, "previous_speech_end_ms")
        next_start = _timestamp(row, "next_event_start_ms")
        outcome = _text(row, "outcome", nonempty=True)
        if outcome not in _OUTCOMES:
            raise SchemaError("outcome must be continue, yield, overlap, or unclear")
        true_end = _timestamp(row, "true_end_ms", allow_none=True)
        reason = row["exclusion_reason"]
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise SchemaError("exclusion_reason must be a non-empty string or null")
        final = _text(row, "final_label", nonempty=True)
        if final not in _FINAL_LABELS:
            raise SchemaError("final_label is invalid")
        raw_annotations = row["annotations"]
        if not isinstance(raw_annotations, list):
            raise SchemaError("annotations must be a list")
        annotations = tuple(Annotation.from_dict(item) for item in raw_annotations)
        if outcome == "continue" and "continuation_resume_ms" not in row:
            raise SchemaError("continue requires continuation_resume_ms")
        resume = _timestamp(row, "continuation_resume_ms", allow_none=True) if "continuation_resume_ms" in row else None
        return cls(**ids, previous_speech_end_ms=previous, next_event_start_ms=next_start, outcome=outcome, true_end_ms=true_end, exclusion_reason=reason, final_label=final, annotations=annotations, continuation_resume_ms=resume)

    def to_dict(self) -> dict[str, object]:
        result = {"schema": self.schema, "decision_id": self.decision_id, "recording_id": self.recording_id, "source_recording_id": self.source_recording_id, "target_speaker_id": self.target_speaker_id, "previous_speech_end_ms": self.previous_speech_end_ms, "next_event_start_ms": self.next_event_start_ms, "outcome": self.outcome, "true_end_ms": self.true_end_ms, "exclusion_reason": self.exclusion_reason, "final_label": self.final_label, "annotations": [a.to_dict() for a in self.annotations]}
        # Keep the optional field explicit in JSONL: ``null`` distinguishes a
        # yield label from a malformed continue label and makes records stable
        # across round trips.
        result["continuation_resume_ms"] = self.continuation_resume_ms
        return result


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    t_ms: int

    _KINDS = {"speech_started", "turn_committed", "agent_first_audio"}

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_text(self.kind, "kind", nonempty=True)
        if self.kind not in self._KINDS:
            raise SchemaError(f"unknown event kind: {self.kind}")
        _validate_timestamp(self.t_ms, "t_ms")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "TraceEvent":
        row = _mapping(row)
        _fields(row, {"kind", "t_ms"})
        kind = _text(row, "kind", nonempty=True)
        if kind not in cls._KINDS:
            raise SchemaError(f"unknown event kind: {kind}")
        return cls(kind, _timestamp(row, "t_ms"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "t_ms": self.t_ms}


@dataclass(frozen=True)
class ProviderTrace:
    run_id: str
    decision_id: str
    provider: str
    status: str
    error: str | None
    events: tuple[TraceEvent, ...]

    schema = EVENT_SCHEMA

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("run_id", "decision_id", "provider"):
            _validate_text(getattr(self, name), name, nonempty=True)
        _validate_text(self.status, "status", nonempty=True)
        if self.status not in {"ok", "failed", "timeout"}:
            raise SchemaError("status must be ok, failed, or timeout")
        if self.error is not None and not isinstance(self.error, str):
            raise SchemaError("error must be a string or null")
        if not isinstance(self.events, tuple):
            raise SchemaError("events must be a tuple of TraceEvent records")
        for event in self.events:
            if not isinstance(event, TraceEvent):
                raise SchemaError("events must contain TraceEvent records")
            event.validate()
        if any(
            left.t_ms > right.t_ms for left, right in zip(self.events, self.events[1:])
        ):
            raise SchemaError("event timestamps must be monotonic")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "ProviderTrace":
        row = _mapping(row)
        _fields(row, {"schema", "run_id", "decision_id", "provider", "status", "error", "events"})
        if row["schema"] != cls.schema:
            raise SchemaError(f"schema must be {cls.schema}")
        values = {name: _text(row, name, nonempty=True) for name in ("run_id", "decision_id", "provider")}
        status = _text(row, "status", nonempty=True)
        if status not in {"ok", "failed", "timeout"}:
            raise SchemaError("status must be ok, failed, or timeout")
        error = row["error"]
        if error is not None and not isinstance(error, str):
            raise SchemaError("error must be a string or null")
        raw_events = row["events"]
        if not isinstance(raw_events, list):
            raise SchemaError("events must be a list")
        events = tuple(TraceEvent.from_dict(item) for item in raw_events)
        if any(left.t_ms > right.t_ms for left, right in zip(events, events[1:])):
            raise SchemaError("event timestamps must be monotonic")
        return cls(**values, status=status, error=error, events=events)

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "run_id": self.run_id, "decision_id": self.decision_id, "provider": self.provider, "status": self.status, "error": self.error, "events": [event.to_dict() for event in self.events]}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise SchemaError(f"non-standard JSON constant: {value}")


def _decode_strict_json(raw: str) -> object:
    """Decode standards-compliant JSON and reject duplicate object keys."""

    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def read_jsonl(path: Path, parser: Callable[[dict[str, object]], T]) -> list[T]:
    rows: list[T] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = _decode_strict_json(raw)
            if not isinstance(value, dict):
                raise SchemaError("record must be an object")
            rows.append(parser(value))
        except (json.JSONDecodeError, SchemaError) as exc:
            raise SchemaError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    lines: list[str] = []
    for line_no, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise SchemaError(f"{path}:{line_no}: record must be an object")
        try:
            lines.append(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{path}:{line_no}: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")

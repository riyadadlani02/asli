"""Strict wire records for TurnBench's label-free automatic evaluation lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

from .schema import SchemaError, read_jsonl, write_jsonl


CANDIDATE_SCHEMA = "turnbench.diarbench.candidate.v1"
REFERENCE_SCHEMA = "turnbench.diarbench.reference.v1"
PREDICTION_SCHEMA = "turnbench.auto_prediction.v1"
REFERENCE_SOURCE = "indic_diarbench_human_timing.v1"

_CANDIDATE_FIELDS = {
    "schema",
    "decision_id",
    "recording_id",
    "source_recording_id",
    "audio_path",
    "language",
    "condition",
    "target_speaker_id",
    "context_start_ms",
    "previous_speech_end_ms",
    "observation_end_ms",
}
_REFERENCE_FIELDS = _CANDIDATE_FIELDS | {"outcome", "reference_source", "exclusion_reason"}
_PREDICTION_FIELDS = {
    "schema",
    "decision_id",
    "run_id",
    "agent",
    "model",
    "config",
    "status",
    "outcome",
    "endpoint_ms",
    "unavailable_reason",
}
_BINARY_OUTCOMES = {"continue", "yield"}
_REFERENCE_OUTCOMES = _BINARY_OUTCOMES | {"overlap", "unclear"}


def _require_fields(row: Mapping[str, object], fields: set[str]) -> None:
    if not isinstance(row, Mapping):
        raise SchemaError("record must be an object")
    missing = fields - set(row)
    extra = set(row) - fields
    if missing:
        raise SchemaError(f"missing field: {sorted(missing)[0]}")
    if extra:
        raise SchemaError(f"unknown field: {sorted(extra)[0]}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _timestamp(value: object, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer timestamp")
    return value


def _validate_json(value: object, name: str = "config") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise SchemaError(f"{name} must contain only finite JSON values")
    if isinstance(value, list):
        for item in value:
            _validate_json(item, name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{name} keys must be strings")
            _validate_json(item, name)
        return
    raise SchemaError(f"{name} must contain only JSON values")


@dataclass(frozen=True)
class DiarBenchCandidate:
    decision_id: str
    recording_id: str
    source_recording_id: str
    audio_path: str
    language: str
    condition: str
    target_speaker_id: str
    context_start_ms: int
    previous_speech_end_ms: int
    observation_end_ms: int

    schema = CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "decision_id", "recording_id", "source_recording_id", "audio_path",
            "language", "condition", "target_speaker_id",
        ):
            _text(getattr(self, name), name)
        for name in ("context_start_ms", "previous_speech_end_ms", "observation_end_ms"):
            _timestamp(getattr(self, name), name)
        if not self.context_start_ms <= self.previous_speech_end_ms < self.observation_end_ms:
            raise SchemaError(
                "context_start_ms must be less than or equal to "
                "previous_speech_end_ms, which must be less than observation_end_ms"
            )

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "DiarBenchCandidate":
        _require_fields(row, _CANDIDATE_FIELDS)
        if row["schema"] != cls.schema:
            raise SchemaError(f"schema must be {cls.schema}")
        return cls(
            **{name: _text(row[name], name) for name in (
                "decision_id", "recording_id", "source_recording_id", "audio_path",
                "language", "condition", "target_speaker_id",
            )},
            context_start_ms=_timestamp(row["context_start_ms"], "context_start_ms"),
            previous_speech_end_ms=_timestamp(row["previous_speech_end_ms"], "previous_speech_end_ms"),
            observation_end_ms=_timestamp(row["observation_end_ms"], "observation_end_ms"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, **{name: getattr(self, name) for name in _CANDIDATE_FIELDS - {"schema"}}}


@dataclass(frozen=True)
class DiarBenchReference:
    candidate: DiarBenchCandidate
    outcome: Literal["continue", "yield", "overlap", "unclear"]
    reference_source: str
    exclusion_reason: str | None

    schema = REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DiarBenchCandidate):
            raise SchemaError("candidate must be a DiarBenchCandidate")
        if self.outcome not in _REFERENCE_OUTCOMES:
            raise SchemaError("outcome must be continue, yield, overlap, or unclear")
        if self.reference_source != REFERENCE_SOURCE:
            raise SchemaError(f"reference_source must be {REFERENCE_SOURCE}")
        if self.exclusion_reason is not None and (
            not isinstance(self.exclusion_reason, str) or not self.exclusion_reason.strip()
        ):
            raise SchemaError("exclusion_reason must be a non-empty string or null")
        if self.outcome in _BINARY_OUTCOMES and self.exclusion_reason is not None:
            raise SchemaError(f"{self.outcome} requires exclusion_reason to be null")
        if self.outcome not in _BINARY_OUTCOMES and self.exclusion_reason is None:
            raise SchemaError(f"{self.outcome} requires exclusion_reason")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "DiarBenchReference":
        _require_fields(row, _REFERENCE_FIELDS)
        if row["schema"] != cls.schema:
            raise SchemaError(f"schema must be {cls.schema}")
        candidate_row = {name: row[name] for name in _CANDIDATE_FIELDS}
        candidate_row["schema"] = CANDIDATE_SCHEMA
        candidate = DiarBenchCandidate.from_dict(candidate_row)
        outcome = _text(row["outcome"], "outcome")
        reason = row["exclusion_reason"]
        return cls(candidate, outcome, _text(row["reference_source"], "reference_source"), reason)

    def as_candidate(self) -> DiarBenchCandidate | None:
        return self.candidate if self.outcome in _BINARY_OUTCOMES else None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.candidate.to_dict(),
            "schema": self.schema,
            "outcome": self.outcome,
            "reference_source": self.reference_source,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class AutoPrediction:
    decision_id: str
    run_id: str
    agent: str
    model: str
    config: dict[str, object]
    status: Literal["available", "unavailable"]
    outcome: Literal["continue", "yield"] | None
    endpoint_ms: int | None
    unavailable_reason: str | None

    schema = PREDICTION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("decision_id", "run_id", "agent", "model"):
            _text(getattr(self, name), name)
        if not isinstance(self.config, dict):
            raise SchemaError("config must be an object")
        _validate_json(self.config)
        if self.status not in {"available", "unavailable"}:
            raise SchemaError("status must be available or unavailable")
        if self.outcome is not None and self.outcome not in _BINARY_OUTCOMES:
            raise SchemaError("outcome must be continue, yield, or null")
        _timestamp(self.endpoint_ms, "endpoint_ms", allow_none=True)
        if self.unavailable_reason is not None and (
            not isinstance(self.unavailable_reason, str) or not self.unavailable_reason.strip()
        ):
            raise SchemaError("unavailable_reason must be a non-empty string or null")
        if self.status == "available":
            if self.outcome not in _BINARY_OUTCOMES or self.unavailable_reason is not None:
                raise SchemaError("available requires a binary outcome and null unavailable_reason")
            if self.outcome == "yield" and self.endpoint_ms is None:
                raise SchemaError("available yield requires endpoint_ms")
            if self.outcome == "continue" and self.endpoint_ms is not None:
                raise SchemaError("continue requires endpoint_ms to be null")
        elif self.outcome is not None or self.endpoint_ms is not None or self.unavailable_reason is None:
            raise SchemaError("unavailable requires null outcome and endpoint_ms plus unavailable_reason")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "AutoPrediction":
        _require_fields(row, _PREDICTION_FIELDS)
        if row["schema"] != cls.schema:
            raise SchemaError(f"schema must be {cls.schema}")
        config = row["config"]
        if not isinstance(config, dict):
            raise SchemaError("config must be an object")
        return cls(
            **{name: _text(row[name], name) for name in ("decision_id", "run_id", "agent", "model")},
            config=config,
            status=_text(row["status"], "status"),
            outcome=row["outcome"],
            endpoint_ms=_timestamp(row["endpoint_ms"], "endpoint_ms", allow_none=True),
            unavailable_reason=row["unavailable_reason"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "agent": self.agent,
            "model": self.model,
            "config": self.config,
            "status": self.status,
            "outcome": self.outcome,
            "endpoint_ms": self.endpoint_ms,
            "unavailable_reason": self.unavailable_reason,
        }


def read_candidates(path: Path) -> list[DiarBenchCandidate]:
    return read_jsonl(path, DiarBenchCandidate.from_dict)


def write_candidates(path: Path, rows: Iterable[DiarBenchCandidate]) -> None:
    write_jsonl(path, (row.to_dict() for row in rows))


def read_references(path: Path) -> list[DiarBenchReference]:
    return read_jsonl(path, DiarBenchReference.from_dict)


def write_references(path: Path, rows: Iterable[DiarBenchReference]) -> None:
    write_jsonl(path, (row.to_dict() for row in rows))


def read_predictions(path: Path) -> list[AutoPrediction]:
    return read_jsonl(path, AutoPrediction.from_dict)


def write_predictions(path: Path, rows: Iterable[AutoPrediction]) -> None:
    write_jsonl(path, (row.to_dict() for row in rows))

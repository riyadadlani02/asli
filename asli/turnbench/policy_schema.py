"""Strict, versioned records for the calibrated TurnBench policy lane."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

from .schema import SchemaError, read_jsonl, write_jsonl

POLICY_FEATURE_SCHEMA = "turnbench.policy_feature.v1"
POLICY_SPLIT_SCHEMA = "turnbench.policy_split.v1"
POLICY_ARTIFACT_SCHEMA = "turnbench.policy_artifact.v1"
POLICY_DECISION_SCHEMA = "turnbench.policy_decision.v1"


def _fields(row: Mapping[str, object], expected: set[str]) -> None:
    if not isinstance(row, Mapping):
        raise SchemaError("record must be an object")
    missing = expected - set(row)
    extra = set(row) - expected
    if missing:
        raise SchemaError(f"missing field: {sorted(missing)[0]}")
    if extra:
        raise SchemaError(f"unknown field: {sorted(extra)[0]}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SchemaError(f"{name} must be finite")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer")
    return value


def _config(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SchemaError("extractor_config must be an object")
    def check(item: object, name: str) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise SchemaError(f"{name} must contain only finite JSON values")
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise SchemaError(f"{name} keys must be strings")
            for k, v in item.items(): check(v, name)
        elif isinstance(item, list):
            for v in item: check(v, name)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise SchemaError(f"{name} must contain only JSON values")
    check(value, "extractor_config")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{name} must be a list")
    result = tuple(_text(item, name) for item in value)
    if not result:
        raise SchemaError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True)
class PolicyFeature:
    decision_id: str; recording_id: str; source_recording_id: str; language: str; condition: str
    export_fingerprint: str; extractor_config: dict[str, object]; audio_fingerprint: str
    pause_ms: int; trailing_energy: float; trailing_energy_slope: float; trailing_speech_ms: int
    local_speech_rate_hz: float
    semantic_status: Literal["absent", "available", "unavailable"]
    semantic_outcome: Literal["continue", "yield"] | None
    semantic_endpoint_offset_ms: int | None
    schema = POLICY_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("decision_id", "recording_id", "source_recording_id", "language", "condition", "export_fingerprint", "audio_fingerprint"):
            _text(getattr(self, name), name)
        _config(self.extractor_config)
        for name in ("pause_ms", "trailing_speech_ms"):
            _integer(getattr(self, name), name)
        for name in ("trailing_energy", "trailing_energy_slope", "local_speech_rate_hz"):
            _finite(getattr(self, name), name)
        if self.semantic_status not in {"absent", "available", "unavailable"}:
            raise SchemaError("semantic_status must be absent, available, or unavailable")
        if self.semantic_outcome not in {None, "continue", "yield"}:
            raise SchemaError("semantic_outcome must be continue, yield, or null")
        if self.semantic_endpoint_offset_ms is not None:
            _integer(self.semantic_endpoint_offset_ms, "semantic_endpoint_offset_ms")
        if self.semantic_status != "available" and (self.semantic_outcome is not None or self.semantic_endpoint_offset_ms is not None):
            raise SchemaError("absent or unavailable semantic status requires null outcome and endpoint")
        if self.semantic_status == "available" and self.semantic_outcome not in {"continue", "yield"}:
            raise SchemaError("available semantic status requires a binary outcome")
        if self.semantic_status == "available" and self.semantic_outcome == "continue" and self.semantic_endpoint_offset_ms is not None:
            raise SchemaError("continue semantic outcome requires null endpoint")
        if self.semantic_status == "available" and self.semantic_outcome == "yield" and self.semantic_endpoint_offset_ms is None:
            raise SchemaError("yield semantic outcome requires endpoint")

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "PolicyFeature":
        expected = {"schema", "decision_id", "recording_id", "source_recording_id", "language", "condition", "export_fingerprint", "extractor_config", "audio_fingerprint", "pause_ms", "trailing_energy", "trailing_energy_slope", "trailing_speech_ms", "local_speech_rate_hz", "semantic_status", "semantic_outcome", "semantic_endpoint_offset_ms"}
        _fields(row, expected)
        if row["schema"] != cls.schema: raise SchemaError(f"schema must be {cls.schema}")
        return cls(**{k: row[k] for k in expected - {"schema"}})

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, **{k: getattr(self, k) for k in ("decision_id", "recording_id", "source_recording_id", "language", "condition", "export_fingerprint", "extractor_config", "audio_fingerprint", "pause_ms", "trailing_energy", "trailing_energy_slope", "trailing_speech_ms", "local_speech_rate_hz", "semantic_status", "semantic_outcome", "semantic_endpoint_offset_ms")}}


@dataclass(frozen=True)
class PolicySplit:
    seed: int; language: str; train_source_recording_ids: tuple[str, ...]; calibration_source_recording_ids: tuple[str, ...]; test_source_recording_ids: tuple[str, ...]
    schema = POLICY_SPLIT_SCHEMA
    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int): raise SchemaError("seed must be an integer")
        _text(self.language, "language")
        groups = [_strings(getattr(self, n), n) for n in ("train_source_recording_ids", "calibration_source_recording_ids", "test_source_recording_ids")]
        if len(set().union(*map(set, groups))) != sum(map(len, groups)): raise SchemaError("split source recording groups must be distinct")
    @classmethod
    def from_dict(cls, row):
        expected = {"schema", "seed", "language", "train_source_recording_ids", "calibration_source_recording_ids", "test_source_recording_ids"}; _fields(row, expected)
        if row["schema"] != cls.schema: raise SchemaError(f"schema must be {cls.schema}")
        return cls(row["seed"], row["language"], tuple(row["train_source_recording_ids"]), tuple(row["calibration_source_recording_ids"]), tuple(row["test_source_recording_ids"]))
    def to_dict(self): return {"schema": self.schema, "seed": self.seed, "language": self.language, "train_source_recording_ids": list(self.train_source_recording_ids), "calibration_source_recording_ids": list(self.calibration_source_recording_ids), "test_source_recording_ids": list(self.test_source_recording_ids)}


@dataclass(frozen=True)
class PolicyArtifact:
    policy_id: str; language: str; feature_schema: str; export_fingerprint: str; extractor_config: dict[str, object]
    coefficients: tuple[float, float, float, float, float, float, float]; means: tuple[float, float, float, float, float, float, float]; scales: tuple[float, float, float, float, float, float, float]
    yield_threshold: float; hold_threshold: float; grace_ms: int; hard_deadline_ms: int
    train_source_recording_ids: tuple[str, ...]; calibration_source_recording_ids: tuple[str, ...]
    schema = POLICY_ARTIFACT_SCHEMA
    def __post_init__(self) -> None:
        _text(self.policy_id, "policy_id"); _text(self.language, "language"); _text(self.feature_schema, "feature_schema"); _text(self.export_fingerprint, "export_fingerprint"); _config(self.extractor_config)
        if self.feature_schema != POLICY_FEATURE_SCHEMA: raise SchemaError(f"feature_schema must be {POLICY_FEATURE_SCHEMA}")
        for name in ("coefficients", "means", "scales"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 7: raise SchemaError(f"{name} must contain seven values")
            for item in value: _finite(item, name)
        if any(item <= 0 for item in self.scales): raise SchemaError("scales must be positive and finite")
        _finite(self.yield_threshold, "yield_threshold"); _finite(self.hold_threshold, "hold_threshold")
        if not 0 <= self.yield_threshold < self.hold_threshold <= 1: raise SchemaError("yield_threshold must be below hold_threshold within [0, 1]")
        _integer(self.grace_ms, "grace_ms"); _integer(self.hard_deadline_ms, "hard_deadline_ms")
        if self.hard_deadline_ms <= self.grace_ms: raise SchemaError("hard_deadline_ms must exceed grace_ms")
        _strings(self.train_source_recording_ids, "train_source_recording_ids"); _strings(self.calibration_source_recording_ids, "calibration_source_recording_ids")
        if set(self.train_source_recording_ids) & set(self.calibration_source_recording_ids):
            raise SchemaError("train and calibration source recording groups must be distinct")
    @classmethod
    def from_dict(cls, row):
        expected = {"schema", "policy_id", "language", "feature_schema", "export_fingerprint", "extractor_config", "coefficients", "means", "scales", "yield_threshold", "hold_threshold", "grace_ms", "hard_deadline_ms", "train_source_recording_ids", "calibration_source_recording_ids"}; _fields(row, expected)
        if row["schema"] != cls.schema: raise SchemaError(f"schema must be {cls.schema}")
        return cls(row["policy_id"], row["language"], row["feature_schema"], row["export_fingerprint"], row["extractor_config"], tuple(row["coefficients"]), tuple(row["means"]), tuple(row["scales"]), row["yield_threshold"], row["hold_threshold"], row["grace_ms"], row["hard_deadline_ms"], tuple(row["train_source_recording_ids"]), tuple(row["calibration_source_recording_ids"]))
    def to_dict(self): return {"schema": self.schema, **{name: (list(getattr(self, name)) if name in {"coefficients", "means", "scales", "train_source_recording_ids", "calibration_source_recording_ids"} else getattr(self, name)) for name in ("policy_id", "language", "feature_schema", "export_fingerprint", "extractor_config", "coefficients", "means", "scales", "yield_threshold", "hold_threshold", "grace_ms", "hard_deadline_ms", "train_source_recording_ids", "calibration_source_recording_ids")}}


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str; policy_id: str; probability_continue: float | None; action: Literal["hold", "yield", "uncertain"] | None; status: Literal["available", "unavailable"]; unavailable_reason: str | None
    schema = POLICY_DECISION_SCHEMA
    def __post_init__(self) -> None:
        _text(self.decision_id, "decision_id"); _text(self.policy_id, "policy_id")
        if self.probability_continue is not None and not 0 <= _finite(self.probability_continue, "probability_continue") <= 1: raise SchemaError("probability_continue must be within [0, 1]")
        if self.action not in {None, "hold", "yield", "uncertain"}: raise SchemaError("action is invalid")
        if self.status not in {"available", "unavailable"}: raise SchemaError("status must be available or unavailable")
        if self.status == "unavailable":
            if self.probability_continue is not None or self.action is not None: raise SchemaError("unavailable decisions require null probability and action")
            _text(self.unavailable_reason, "unavailable_reason")
        elif self.unavailable_reason is not None or self.probability_continue is None or self.action is None: raise SchemaError("available decisions require probability, action, and null reason")
    @classmethod
    def from_dict(cls, row):
        expected = {"schema", "decision_id", "policy_id", "probability_continue", "action", "status", "unavailable_reason"}; _fields(row, expected)
        if row["schema"] != cls.schema: raise SchemaError(f"schema must be {cls.schema}")
        return cls(row["decision_id"], row["policy_id"], row["probability_continue"], row["action"], row["status"], row["unavailable_reason"])
    def to_dict(self): return {"schema": self.schema, "decision_id": self.decision_id, "policy_id": self.policy_id, "probability_continue": self.probability_continue, "action": self.action, "status": self.status, "unavailable_reason": self.unavailable_reason}


def read_policy_features(path: Path) -> list[PolicyFeature]: return read_jsonl(path, PolicyFeature.from_dict)
def write_policy_features(path: Path, rows: Iterable[PolicyFeature]) -> None: write_jsonl(path, (row.to_dict() for row in sorted(rows, key=lambda r: r.decision_id)))

def _read_object(path: Path, reader):
    try:
        def reject_constants(value: str):
            raise ValueError(value)
        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key: {key}")
                result[key] = value
            return result
        with Path(path).open(encoding="utf-8") as stream:
            return reader(json.load(stream, parse_constant=reject_constants, object_pairs_hook=reject_duplicates))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc: raise SchemaError(f"{path}: malformed JSON") from exc
def _write_object(path: Path, row: Mapping[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream: json.dump(row, stream, allow_nan=False, sort_keys=True); stream.write("\n")
def read_policy_split(path: Path) -> PolicySplit: return _read_object(path, PolicySplit.from_dict)
def write_policy_split(path: Path, split: PolicySplit) -> None: _write_object(path, split.to_dict())
def read_policy_artifact(path: Path) -> PolicyArtifact: return _read_object(path, PolicyArtifact.from_dict)
def write_policy_artifact(path: Path, artifact: PolicyArtifact) -> None: _write_object(path, artifact.to_dict())

"""Versioned records used by the offline TurnBench boundary."""

from .candidates import PauseCandidate, extract_candidates
from .report import score_inputs
from .schema import (
    Annotation,
    DecisionLabel,
    EVENT_SCHEMA,
    LABEL_SCHEMA,
    ProviderTrace,
    RECORDING_SCHEMA,
    Recording,
    SchemaError,
    SourceTurn,
    TraceEvent,
    read_jsonl,
    write_jsonl,
)
from .score import DecisionScore, aggregate, bootstrap_by_recording, nearest_rank, score_decision

__all__ = [
    "Annotation",
    "DecisionLabel",
    "DecisionScore",
    "EVENT_SCHEMA",
    "LABEL_SCHEMA",
    "ProviderTrace",
    "PauseCandidate",
    "RECORDING_SCHEMA",
    "Recording",
    "SchemaError",
    "SourceTurn",
    "TraceEvent",
    "aggregate",
    "bootstrap_by_recording",
    "extract_candidates",
    "nearest_rank",
    "read_jsonl",
    "score_decision",
    "score_inputs",
    "write_jsonl",
]

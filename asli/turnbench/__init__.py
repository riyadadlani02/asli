"""Versioned records used by the offline TurnBench boundary."""

from .candidates import PauseCandidate, extract_candidates
from .auto_schema import (
    AutoPrediction,
    CANDIDATE_SCHEMA,
    PREDICTION_SCHEMA,
    REFERENCE_SCHEMA,
    REFERENCE_SOURCE,
    DiarBenchCandidate,
    DiarBenchReference,
    read_candidates,
    read_predictions,
    read_references,
    write_candidates,
    write_predictions,
    write_references,
)
from .auto_report import (
    EXPORT_PROVENANCE_SCHEMA,
    DiarBenchExportProvenance,
    compare_auto_predictions,
    validate_export_candidates,
    validate_auto_join,
)
from .diarbench import convert_diarbench_sample
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
from .policy_schema import (
    POLICY_ARTIFACT_SCHEMA, POLICY_DECISION_SCHEMA, POLICY_FEATURE_SCHEMA, POLICY_SPLIT_SCHEMA,
    PolicyArtifact, PolicyDecision, PolicyFeature, PolicySplit,
    read_policy_artifact, read_policy_features, read_policy_split,
    write_policy_artifact, write_policy_features, write_policy_split,
)
from .policy_runtime import decide_policy, probability_continue
from .policy_report import replay_policy

__all__ = [
    "Annotation",
    "AutoPrediction",
    "CANDIDATE_SCHEMA",
    "DecisionLabel",
    "DecisionScore",
    "DiarBenchCandidate",
    "DiarBenchExportProvenance",
    "DiarBenchReference",
    "EVENT_SCHEMA",
    "EXPORT_PROVENANCE_SCHEMA",
    "LABEL_SCHEMA",
    "ProviderTrace",
    "PauseCandidate",
    "PREDICTION_SCHEMA",
    "RECORDING_SCHEMA",
    "REFERENCE_SCHEMA",
    "REFERENCE_SOURCE",
    "Recording",
    "SchemaError",
    "SourceTurn",
    "TraceEvent",
    "aggregate",
    "bootstrap_by_recording",
    "convert_diarbench_sample",
    "compare_auto_predictions",
    "extract_candidates",
    "nearest_rank",
    "read_jsonl",
    "read_candidates",
    "read_predictions",
    "read_references",
    "score_decision",
    "score_inputs",
    "write_jsonl",
    "write_candidates",
    "write_predictions",
    "write_references",
    "validate_auto_join",
    "validate_export_candidates",
    "POLICY_ARTIFACT_SCHEMA", "POLICY_DECISION_SCHEMA", "POLICY_FEATURE_SCHEMA", "POLICY_SPLIT_SCHEMA",
    "PolicyArtifact", "PolicyDecision", "PolicyFeature", "PolicySplit",
    "read_policy_artifact", "read_policy_features", "read_policy_split",
    "write_policy_artifact", "write_policy_features", "write_policy_split",
    "decide_policy", "probability_continue", "replay_policy",
]

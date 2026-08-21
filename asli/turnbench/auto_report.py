"""Closed-join accuracy reports for TurnBench automatic predictions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .auto_schema import (
    REFERENCE_SOURCE,
    AutoPrediction,
    DiarBenchCandidate,
    DiarBenchReference,
)
from .report import _normalize_config
from .score import nearest_rank


_BINARY_OUTCOMES = {"continue", "yield"}
EXPORT_PROVENANCE_SCHEMA = "turnbench.diarbench.export.v1"
_MACRO_METRICS = (
    "accuracy",
    "continue_precision",
    "continue_recall",
    "continue_f1",
)


@dataclass(frozen=True)
class DiarBenchExportProvenance:
    """The versioned export settings that produced candidates and references."""

    dataset: str
    dataset_revision: str
    requested_languages: tuple[str, ...]
    min_pause_ms: int
    max_pause_ms: int
    context_ms: int

    schema = EXPORT_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("dataset", "dataset_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.requested_languages, tuple)
            or not self.requested_languages
        ):
            raise ValueError("requested_languages must be a non-empty tuple")
        for language in self.requested_languages:
            if not isinstance(language, str) or not language.strip():
                raise ValueError("requested_languages must contain non-empty strings")
        if len(set(self.requested_languages)) != len(self.requested_languages):
            raise ValueError("requested_languages must not contain duplicates")
        for name in ("min_pause_ms", "max_pause_ms", "context_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.min_pause_ms > self.max_pause_ms:
            raise ValueError("min_pause_ms must not exceed max_pause_ms")
        object.__setattr__(
            self, "requested_languages", tuple(sorted(self.requested_languages))
        )

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "DiarBenchExportProvenance":
        if not isinstance(row, Mapping):
            raise ValueError("export provenance must be an object")
        fields = {
            "schema",
            "dataset",
            "dataset_revision",
            "requested_languages",
            "min_pause_ms",
            "max_pause_ms",
            "context_ms",
        }
        missing = fields - set(row)
        extra = set(row) - fields
        if missing:
            raise ValueError(f"missing export provenance field: {sorted(missing)[0]}")
        if extra:
            raise ValueError(f"unknown export provenance field: {sorted(extra)[0]}")
        if row["schema"] != cls.schema:
            raise ValueError(f"schema must be {cls.schema}")
        languages = row["requested_languages"]
        if not isinstance(languages, list):
            raise ValueError("requested_languages must be a list")
        return cls(
            dataset=row["dataset"],
            dataset_revision=row["dataset_revision"],
            requested_languages=tuple(languages),
            min_pause_ms=row["min_pause_ms"],
            max_pause_ms=row["max_pause_ms"],
            context_ms=row["context_ms"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "requested_languages": list(self.requested_languages),
            "min_pause_ms": self.min_pause_ms,
            "max_pause_ms": self.max_pause_ms,
            "context_ms": self.context_ms,
        }


@dataclass(frozen=True)
class ValidatedAutoJoin:
    """The complete, provenance-consistent inputs used by one report."""

    candidates: dict[str, DiarBenchCandidate]
    references: dict[str, DiarBenchReference]
    predictions: dict[str, AutoPrediction]
    run_id: str | None
    agent: str | None
    model: str | None
    config: dict[str, object] | None
    export_provenance: DiarBenchExportProvenance


def _unique_by_decision_id(
    rows: Iterable[object], expected_type: type, collection: str
) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for row in rows:
        if not isinstance(row, expected_type):
            raise ValueError(f"{collection} must contain {expected_type.__name__} records")
        if isinstance(row, DiarBenchReference):
            decision_id = row.candidate.decision_id
        else:
            decision_id = row.decision_id
        if decision_id in indexed:
            raise ValueError(f"duplicate {collection[:-1]} decision_id: {decision_id}")
        indexed[decision_id] = row
    return indexed


def _sorted_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


def validate_auto_join(
    candidates: Iterable[DiarBenchCandidate],
    references: Iterable[DiarBenchReference],
    predictions: Iterable[AutoPrediction],
    *,
    export_provenance: DiarBenchExportProvenance,
) -> ValidatedAutoJoin:
    """Validate a complete one-to-one join and one prediction provenance."""

    if not isinstance(export_provenance, DiarBenchExportProvenance):
        raise ValueError("export_provenance must be a DiarBenchExportProvenance")

    candidate_rows = _unique_by_decision_id(
        candidates, DiarBenchCandidate, "candidates"
    )
    reference_rows = _unique_by_decision_id(
        references, DiarBenchReference, "references"
    )
    prediction_rows = _unique_by_decision_id(
        predictions, AutoPrediction, "predictions"
    )

    candidates_by_id = {
        key: value
        for key, value in candidate_rows.items()
        if isinstance(value, DiarBenchCandidate)
    }
    references_by_id = {
        key: value
        for key, value in reference_rows.items()
        if isinstance(value, DiarBenchReference)
    }
    predictions_by_id = {
        key: value
        for key, value in prediction_rows.items()
        if isinstance(value, AutoPrediction)
    }

    for decision_id, candidate in candidates_by_id.items():
        reference = references_by_id.get(decision_id)
        if reference is None:
            raise ValueError(f"candidate missing binary reference: {decision_id}")
        if reference.outcome not in _BINARY_OUTCOMES:
            raise ValueError(f"candidate linked to excluded reference: {decision_id}")
        if candidate != reference.candidate:
            raise ValueError(f"candidate/reference metadata mismatch: {decision_id}")
        if decision_id not in predictions_by_id:
            raise ValueError(f"candidate missing prediction: {decision_id}")
        prediction = predictions_by_id[decision_id]
        if prediction.status == "available" and prediction.outcome == "yield":
            endpoint_ms = prediction.endpoint_ms
            if endpoint_ms is None or not (
                candidate.previous_speech_end_ms
                < endpoint_ms
                < candidate.observation_end_ms
            ):
                raise ValueError(
                    f"yield endpoint outside candidate window: {decision_id}"
                )

    for decision_id, reference in references_by_id.items():
        candidate = candidates_by_id.get(decision_id)
        if reference.outcome in _BINARY_OUTCOMES:
            if candidate is None:
                raise ValueError(f"binary reference missing candidate: {decision_id}")
        elif candidate is not None:
            raise ValueError(f"candidate linked to excluded reference: {decision_id}")

    for decision_id in predictions_by_id:
        if decision_id not in candidates_by_id:
            raise ValueError(f"prediction missing candidate: {decision_id}")

    requested_languages = set(export_provenance.requested_languages)
    for decision_id, reference in references_by_id.items():
        if reference.candidate.language not in requested_languages:
            raise ValueError(f"candidate language not requested: {decision_id}")
    for decision_id, candidate in candidates_by_id.items():
        pause_ms = candidate.observation_end_ms - candidate.previous_speech_end_ms
        if not (
            export_provenance.min_pause_ms
            <= pause_ms
            <= export_provenance.max_pause_ms
        ):
            raise ValueError(f"candidate pause outside export bounds: {decision_id}")
        expected_context_start = max(
            0, candidate.previous_speech_end_ms - export_provenance.context_ms
        )
        if candidate.context_start_ms != expected_context_start:
            raise ValueError(
                f"candidate context does not match export provenance: {decision_id}"
            )

    ordered_predictions = [predictions_by_id[key] for key in sorted(predictions_by_id)]
    run_id: str | None = None
    agent: str | None = None
    model: str | None = None
    config: dict[str, object] | None = None
    config_fingerprint: str | None = None
    for prediction in ordered_predictions:
        normalized_config = _normalize_config(prediction.config)
        normalized_fingerprint = json.dumps(
            normalized_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if run_id is None:
            run_id = prediction.run_id
            agent = prediction.agent
            model = prediction.model
            config = normalized_config
            config_fingerprint = normalized_fingerprint
        else:
            if prediction.run_id != run_id:
                raise ValueError("mixed run_id values are not allowed in one report")
            if prediction.agent != agent:
                raise ValueError("mixed agent values are not allowed in one report")
            if prediction.model != model:
                raise ValueError("mixed model values are not allowed in one report")
            if normalized_fingerprint != config_fingerprint:
                raise ValueError("mixed config values are not allowed in one report")

    if config is not None and "context_ms" in config:
        config_context = config["context_ms"]
        if (
            isinstance(config_context, bool)
            or not isinstance(config_context, int)
            or config_context != export_provenance.context_ms
        ):
            raise ValueError(
                "prediction config context_ms does not match export provenance"
            )

    sorted_config = _sorted_json(config) if config is not None else None
    if sorted_config is not None and not isinstance(sorted_config, dict):
        raise ValueError("normalized config must be an object")
    return ValidatedAutoJoin(
        {key: candidates_by_id[key] for key in sorted(candidates_by_id)},
        {key: references_by_id[key] for key in sorted(references_by_id)},
        {key: predictions_by_id[key] for key in sorted(predictions_by_id)},
        run_id,
        agent,
        model,
        sorted_config,
        export_provenance,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _micro_summary(
    rows: Iterable[tuple[DiarBenchReference, AutoPrediction]],
) -> dict[str, object]:
    joined = list(rows)
    available = [
        (reference, prediction)
        for reference, prediction in joined
        if prediction.status == "available"
    ]
    confusion = {
        "continue": {"continue": 0, "yield": 0},
        "yield": {"continue": 0, "yield": 0},
    }
    for reference, prediction in available:
        if prediction.outcome is None:
            raise ValueError("available prediction must have a binary outcome")
        confusion[reference.outcome][prediction.outcome] += 1

    true_continue = confusion["continue"]["continue"]
    false_continue = confusion["yield"]["continue"]
    missed_continue = confusion["continue"]["yield"]
    correct_n = true_continue + confusion["yield"]["yield"]
    precision = _rate(true_continue, true_continue + false_continue)
    recall = _rate(true_continue, true_continue + missed_continue)
    f1 = _rate(2 * true_continue, 2 * true_continue + false_continue + missed_continue)
    eligible_n = len(joined)
    available_n = len(available)
    unavailable_n = eligible_n - available_n
    return {
        "aggregation": "micro_by_decision",
        "eligible_reference_n": eligible_n,
        "available_n": available_n,
        "unavailable_n": unavailable_n,
        "coverage_rate": _rate(available_n, eligible_n),
        "unavailable_rate": _rate(unavailable_n, eligible_n),
        "correct_n": correct_n,
        "accuracy": _rate(correct_n, available_n),
        "continue_precision": precision,
        "continue_recall": recall,
        "continue_f1": f1,
        "confusion": confusion,
    }


def _macro_by_language(
    language_summaries: Mapping[str, Mapping[str, object]],
    micro_summary: Mapping[str, object],
) -> dict[str, object]:
    overall = dict(micro_summary)
    overall["aggregation"] = "macro_by_language"
    for metric in _MACRO_METRICS:
        values = [
            summary[metric]
            for summary in language_summaries.values()
            if summary[metric] is not None
        ]
        overall[metric] = sum(values) / len(values) if values else None
        overall[f"{metric}_language_n"] = len(values)
    return overall


def _grouped_summaries(
    rows: list[tuple[DiarBenchReference, AutoPrediction]],
    attribute: str,
) -> dict[str, dict[str, object]]:
    keys = sorted({getattr(reference.candidate, attribute) for reference, _ in rows})
    return {
        key: _micro_summary(
            (item for item in rows if getattr(item[0].candidate, attribute) == key)
        )
        for key in keys
    }


def compare_auto_predictions(
    candidates: Iterable[DiarBenchCandidate],
    references: Iterable[DiarBenchReference],
    predictions: Iterable[AutoPrediction],
    *,
    export_provenance: DiarBenchExportProvenance,
) -> dict[str, object]:
    """Compare ASLI decisions with observed human-timing continuation references."""

    validated = validate_auto_join(
        candidates,
        references,
        predictions,
        export_provenance=export_provenance,
    )
    joined = [
        (validated.references[decision_id], validated.predictions[decision_id])
        for decision_id in validated.candidates
    ]
    by_language = _grouped_summaries(joined, "language")
    by_condition = _grouped_summaries(joined, "condition")
    by_source = _grouped_summaries(joined, "source_recording_id")
    micro_overall = _micro_summary(joined)

    endpoint_errors = [
        abs(prediction.endpoint_ms - reference.candidate.previous_speech_end_ms)
        for reference, prediction in joined
        if reference.outcome == "yield"
        and prediction.status == "available"
        and prediction.outcome == "yield"
        and prediction.endpoint_ms is not None
    ]
    references_list = list(validated.references.values())
    return {
        "schema": "turnbench.auto_accuracy.v1",
        "dataset": export_provenance.dataset,
        "dataset_revision": export_provenance.dataset_revision,
        "reference_source": REFERENCE_SOURCE,
        "run_id": validated.run_id,
        "agent": validated.agent,
        "model": validated.model,
        "config": validated.config,
        "requested_languages": list(export_provenance.requested_languages),
        "min_pause_ms": export_provenance.min_pause_ms,
        "max_pause_ms": export_provenance.max_pause_ms,
        "context_ms": export_provenance.context_ms,
        "reference_n": len(references_list),
        "candidate_n": len(validated.candidates),
        "prediction_n": len(validated.predictions),
        "reference_excluded_n": sum(
            reference.outcome not in _BINARY_OUTCOMES for reference in references_list
        ),
        "overlap_n": sum(reference.outcome == "overlap" for reference in references_list),
        "unclear_n": sum(reference.outcome == "unclear" for reference in references_list),
        "micro_overall": micro_overall,
        "overall": _macro_by_language(by_language, micro_overall),
        "by_language": by_language,
        "by_condition": by_condition,
        "by_source_recording": by_source,
        "endpoint_timing": {
            "correct_timestamped_yield_n": len(endpoint_errors),
            "endpoint_observation_error_p50_ms": nearest_rank(endpoint_errors, 50),
            "endpoint_observation_error_p95_ms": nearest_rank(endpoint_errors, 95),
        },
    }

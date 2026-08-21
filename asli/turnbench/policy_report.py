"""Fail-closed held-out replay reports for calibrated TurnBench policies."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .auto_schema import AutoPrediction, DiarBenchReference
from .policy_runtime import decide_policy
from .policy_schema import POLICY_FEATURE_SCHEMA, PolicyArtifact, PolicyFeature, PolicySplit


_BINARY_OUTCOMES = {"continue", "yield"}
_METRICS = (
    "accuracy",
    "continuation_recall",
    "premature_yield_rate",
    "unnecessary_hold_rate",
    "coverage_rate",
    "utility",
)


@dataclass(frozen=True)
class _ReplayDecision:
    decision_id: str
    status: str
    action: str | None
    unavailable_reason: str | None


def _config_key(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _index_features(features: Iterable[PolicyFeature], split: PolicySplit, artifact: PolicyArtifact) -> dict[str, PolicyFeature]:
    if not isinstance(split, PolicySplit):
        raise ValueError("split must be a PolicySplit")
    if not isinstance(artifact, PolicyArtifact):
        raise ValueError("artifact must be a PolicyArtifact")
    if split.language != artifact.language:
        raise ValueError("split language does not match artifact")
    if artifact.feature_schema != POLICY_FEATURE_SCHEMA:
        raise ValueError("artifact feature schema mismatch")
    if artifact.train_source_recording_ids != split.train_source_recording_ids:
        raise ValueError("artifact train groups do not match split")
    if artifact.calibration_source_recording_ids != split.calibration_source_recording_ids:
        raise ValueError("artifact calibration groups do not match split")

    groups = set(split.train_source_recording_ids)
    groups.update(split.calibration_source_recording_ids)
    groups.update(split.test_source_recording_ids)
    indexed: dict[str, PolicyFeature] = {}
    for feature in features:
        if not isinstance(feature, PolicyFeature):
            raise ValueError("features must contain PolicyFeature records")
        if feature.decision_id in indexed:
            raise ValueError(f"duplicate feature decision_id: {feature.decision_id}")
        if feature.language != split.language:
            raise ValueError(f"feature language does not match split: {feature.decision_id}")
        if feature.source_recording_id not in groups:
            raise ValueError(f"feature group absent from split: {feature.source_recording_id}")
        if feature.schema != artifact.feature_schema:
            raise ValueError(f"feature schema does not match artifact: {feature.decision_id}")
        if feature.export_fingerprint != artifact.export_fingerprint:
            raise ValueError(f"feature export fingerprint does not match artifact: {feature.decision_id}")
        if _config_key(feature.extractor_config) != _config_key(artifact.extractor_config):
            raise ValueError(f"feature extractor config does not match artifact: {feature.decision_id}")
        indexed[feature.decision_id] = feature
    return {decision_id: indexed[decision_id] for decision_id in sorted(indexed)}


def _index_references(references: Iterable[DiarBenchReference]) -> dict[str, DiarBenchReference]:
    indexed: dict[str, DiarBenchReference] = {}
    for reference in references:
        if not isinstance(reference, DiarBenchReference):
            raise ValueError("references must contain DiarBenchReference records")
        decision_id = reference.candidate.decision_id
        if decision_id in indexed:
            raise ValueError(f"duplicate reference decision_id: {decision_id}")
        indexed[decision_id] = reference
    return indexed


def _same_identity(feature: PolicyFeature, reference: DiarBenchReference) -> bool:
    candidate = reference.candidate
    return (
        candidate.decision_id == feature.decision_id
        and candidate.recording_id == feature.recording_id
        and candidate.source_recording_id == feature.source_recording_id
        and candidate.language == feature.language
        and candidate.condition == feature.condition
    )


def _validate_semantic_predictions(
    semantic_predictions: Iterable[AutoPrediction],
    test_feature_ids: set[str],
) -> dict[str, AutoPrediction] | None:
    rows = list(semantic_predictions)
    if not rows:
        return None
    indexed: dict[str, AutoPrediction] = {}
    provenance: tuple[str, str, str, str] | None = None
    for prediction in rows:
        if not isinstance(prediction, AutoPrediction):
            raise ValueError("semantic_predictions must contain AutoPrediction records")
        if prediction.decision_id in indexed:
            raise ValueError(f"duplicate semantic prediction decision_id: {prediction.decision_id}")
        value = (
            prediction.run_id,
            prediction.agent,
            prediction.model,
            _config_key(prediction.config),
        )
        if provenance is None:
            provenance = value
        elif value != provenance:
            raise ValueError("mixed semantic prediction provenance is not allowed")
        indexed[prediction.decision_id] = prediction
    if set(indexed) != test_feature_ids:
        raise ValueError("semantic prediction IDs must exactly match test features")
    return {decision_id: indexed[decision_id] for decision_id in sorted(indexed)}


def _summary(
    rows: list[tuple[PolicyFeature, DiarBenchReference, _ReplayDecision]], *, grace_ms: int,
) -> dict[str, object]:
    available = [row for row in rows if row[2].status == "available"]
    true_continue = [row for row in available if row[1].outcome == "continue"]
    true_yield = [row for row in available if row[1].outcome == "yield"]
    non_yield_continue = sum(row[2].action != "yield" for row in true_continue)
    yield_continue = sum(row[2].action == "yield" for row in true_continue)
    non_yield_yield = sum(row[2].action != "yield" for row in true_yield)
    correct = non_yield_continue + sum(row[2].action == "yield" for row in true_yield)
    action_counts = Counter("unavailable" if row[2].status != "available" else row[2].action for row in rows)
    unavailable_reasons = Counter(
        row[2].unavailable_reason for row in rows if row[2].status == "unavailable"
    )
    uncertain_n = action_counts["uncertain"]
    continuation_recall = non_yield_continue / len(true_continue) if true_continue else None
    premature_yield_rate = yield_continue / len(true_continue) if true_continue else None
    unnecessary_hold_rate = non_yield_yield / len(true_yield) if true_yield else None
    utility = None
    if continuation_recall is not None and unnecessary_hold_rate is not None:
        utility = 4 * continuation_recall - unnecessary_hold_rate - 0.0005 * uncertain_n * grace_ms
    eligible_n = len(rows)
    available_n = len(available)
    return {
        "aggregation": "micro_by_decision",
        "eligible_n": eligible_n,
        "available_n": available_n,
        "unavailable_n": eligible_n - available_n,
        "coverage_rate": available_n / eligible_n if eligible_n else None,
        "true_continue_n": len(true_continue),
        "true_yield_n": len(true_yield),
        "non_yield_on_true_continue_n": non_yield_continue,
        "premature_yield_n": yield_continue,
        "non_yield_on_true_yield_n": non_yield_yield,
        "continuation_recall": continuation_recall,
        "premature_yield_rate": premature_yield_rate,
        "unnecessary_hold_rate": unnecessary_hold_rate,
        "correct_n": correct,
        "accuracy": correct / available_n if available_n else None,
        "hold_n": action_counts["hold"],
        "yield_n": action_counts["yield"],
        "uncertain_n": uncertain_n,
        "added_grace_ms_total": uncertain_n * grace_ms,
        "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
        "source_recording_n": len({row[0].source_recording_id for row in rows}),
        "utility": utility,
    }


def _grouped_summaries(
    rows: list[tuple[PolicyFeature, DiarBenchReference, _ReplayDecision]], attribute: str, *, grace_ms: int,
) -> dict[str, dict[str, object]]:
    values = sorted({getattr(feature, attribute) for feature, _, _ in rows})
    return {
        value: _summary([row for row in rows if getattr(row[0], attribute) == value], grace_ms=grace_ms)
        for value in values
    }


def _macro_by_language(by_language: dict[str, dict[str, object]], test: dict[str, object]) -> dict[str, object]:
    result = dict(test)
    result["aggregation"] = "macro_by_language"
    for metric in _METRICS:
        values = [summary[metric] for summary in by_language.values() if summary[metric] is not None]
        result[metric] = sum(values) / len(values) if values else None
        result[f"{metric}_language_n"] = len(values)
    return result


def _failed_constraints(
    policy: dict[str, object], always_yield: dict[str, object], semantic: dict[str, object] | None,
) -> list[str]:
    failed: list[str] = []
    recall = policy["continuation_recall"]
    wait = policy["unnecessary_hold_rate"]
    coverage = policy["coverage_rate"]
    if recall is None or recall < 0.80:
        failed.append("continuation_recall")
    if wait is None or wait > 0.20:
        failed.append("unnecessary_hold_rate")
    if coverage is None or coverage < 0.95:
        failed.append("coverage_rate")
    if semantic is None:
        failed.append("semantic_baseline")
        return failed
    utility = policy["utility"]
    if utility is None or always_yield["utility"] is None or utility <= always_yield["utility"]:
        failed.append("utility_over_always_yield")
    if utility is None or semantic["utility"] is None or utility <= semantic["utility"]:
        failed.append("utility_over_semantic_baseline")
    return failed


def replay_policy(
    features: Iterable[PolicyFeature],
    references: Iterable[DiarBenchReference],
    split: PolicySplit,
    artifact: PolicyArtifact,
    *,
    semantic_predictions: Iterable[AutoPrediction] = (),
) -> dict[str, object]:
    """Replay one validated artifact against only its held-out source groups."""
    features_by_id = _index_features(features, split, artifact)
    test_groups = set(split.test_source_recording_ids)
    test_features = [
        feature for feature in features_by_id.values() if feature.source_recording_id in test_groups
    ]
    if not test_features:
        raise ValueError("split has no test features")

    references_by_id = _index_references(references)
    test_feature_ids = {feature.decision_id for feature in test_features}
    binary_test_reference_ids = {
        decision_id
        for decision_id, reference in references_by_id.items()
        if reference.outcome in _BINARY_OUTCOMES
        and reference.candidate.source_recording_id in test_groups
    }
    if binary_test_reference_ids != test_feature_ids:
        raise ValueError("held-out binary reference IDs do not match test features")
    test_rows: list[tuple[PolicyFeature, DiarBenchReference, _ReplayDecision]] = []
    for feature in test_features:
        reference = references_by_id.get(feature.decision_id)
        if reference is None:
            raise ValueError(f"feature missing reference: {feature.decision_id}")
        if reference.outcome not in _BINARY_OUTCOMES:
            raise ValueError(f"reference must be binary: {feature.decision_id}")
        if not _same_identity(feature, reference):
            raise ValueError(f"feature/reference metadata mismatch: {feature.decision_id}")
        decision = decide_policy(feature, artifact)
        test_rows.append(_ReplayDecision(
            decision.decision_id, decision.status, decision.action, decision.unavailable_reason,
        ))

    policy_rows = [
        (feature, references_by_id[feature.decision_id], decision)
        for feature, decision in zip(test_features, test_rows, strict=True)
    ]
    policy_summary = _summary(policy_rows, grace_ms=artifact.grace_ms)
    always_yield_rows = [
        (feature, reference, _ReplayDecision(feature.decision_id, "available", "yield", None))
        for feature, reference, _ in policy_rows
    ]
    always_yield_summary = _summary(always_yield_rows, grace_ms=artifact.grace_ms)

    predictions_by_id = _validate_semantic_predictions(semantic_predictions, test_feature_ids)
    semantic_summary: dict[str, object] | None = None
    if predictions_by_id is not None:
        semantic_rows = []
        for feature, reference, _ in policy_rows:
            prediction = predictions_by_id[feature.decision_id]
            action = None
            if prediction.status == "available":
                action = "hold" if prediction.outcome == "continue" else "yield"
                if prediction.outcome == "yield" and not (
                    reference.candidate.previous_speech_end_ms
                    < prediction.endpoint_ms
                    < reference.candidate.observation_end_ms
                ):
                    raise ValueError(
                        f"semantic yield endpoint outside candidate window: {feature.decision_id}"
                    )
            semantic_rows.append((
                feature,
                reference,
                _ReplayDecision(feature.decision_id, prediction.status, action, prediction.unavailable_reason),
            ))
        semantic_summary = _summary(semantic_rows, grace_ms=artifact.grace_ms)

    by_language = _grouped_summaries(policy_rows, "language", grace_ms=artifact.grace_ms)
    by_condition = _grouped_summaries(policy_rows, "condition", grace_ms=artifact.grace_ms)
    by_source = _grouped_summaries(policy_rows, "source_recording_id", grace_ms=artifact.grace_ms)
    failed_constraints = _failed_constraints(policy_summary, always_yield_summary, semantic_summary)
    split_group_counts = {
        "train": len(split.train_source_recording_ids),
        "calibration": len(split.calibration_source_recording_ids),
        "test": len(split.test_source_recording_ids),
    }
    return {
        "schema": "turnbench.policy_replay.v1",
        "policy_id": artifact.policy_id,
        "language": artifact.language,
        "export_fingerprint": artifact.export_fingerprint,
        "split_group_counts": split_group_counts,
        "train_source_recording_n": split_group_counts["train"],
        "calibration_source_recording_n": split_group_counts["calibration"],
        "test_source_recording_n": split_group_counts["test"],
        "test": policy_summary,
        "always_yield": always_yield_summary,
        "semantic_baseline": semantic_summary,
        "overall": _macro_by_language(by_language, policy_summary),
        "by_language": by_language,
        "by_condition": by_condition,
        "by_source_recording": by_source,
        "policy_win": not failed_constraints,
        "failed_constraints": failed_constraints,
    }

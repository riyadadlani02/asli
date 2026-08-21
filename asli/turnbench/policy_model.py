"""Grouped fitting and calibration for the local TurnBench policy."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping

import numpy as np

from .auto_schema import DiarBenchReference
from .policy_schema import (
    POLICY_EXTRACTOR_CONFIG, POLICY_FEATURE_SCHEMA, PolicyArtifact, PolicyFeature,
    PolicySplit,
)


MODEL_FEATURES = (
    "pause_ms",
    "trailing_energy",
    "trailing_energy_slope",
    "local_speech_rate_hz",
    "semantic_yield_signal",
    "semantic_available_signal",
)
_BINARY_OUTCOMES = {"continue", "yield"}
_POLICY_ID = "turnbench.calibrated_logistic.v1"
_GRACE_MS = 150
_HARD_DEADLINE_MS = 800
_PORTABLE_EXTRACTOR_CONFIG = POLICY_EXTRACTOR_CONFIG


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _config_key(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _portable_config(config: dict[str, object]) -> dict[str, object]:
    """Allow only the extractor's fixed numeric configuration in artifacts."""
    if set(config) != set(_PORTABLE_EXTRACTOR_CONFIG):
        raise ValueError("nonportable extractor configuration")
    for name, expected in _PORTABLE_EXTRACTOR_CONFIG.items():
        value = config[name]
        if type(value) is not type(expected) or value != expected:
            raise ValueError("nonportable extractor configuration")
    return json.loads(_config_key(config))


def _feature_vector(feature: PolicyFeature) -> tuple[float, float, float, float, float, float]:
    semantic_available = feature.semantic_status == "available"
    return (
        float(feature.pause_ms),
        feature.trailing_energy,
        feature.trailing_energy_slope,
        feature.local_speech_rate_hz,
        float(semantic_available and feature.semantic_outcome == "yield"),
        float(semantic_available),
    )


def _feature_rows(features: Iterable[PolicyFeature], language: str) -> list[PolicyFeature]:
    rows: list[PolicyFeature] = []
    decision_ids: set[str] = set()
    for row in features:
        if not isinstance(row, PolicyFeature):
            raise ValueError("features must contain PolicyFeature records")
        if row.decision_id in decision_ids:
            raise ValueError(f"duplicate feature decision_id: {row.decision_id}")
        decision_ids.add(row.decision_id)
        if row.language != language:
            raise ValueError(f"mixed feature languages are not allowed: {row.decision_id}")
        rows.append(row)
    return sorted(rows, key=lambda row: row.decision_id)


def make_group_split(
    features: Iterable[PolicyFeature], *, language: str, seed: int,
    minimum_group_count: int = 20,
) -> PolicySplit:
    """Make a deterministic 60/20/remaining source-recording group split."""
    if isinstance(minimum_group_count, bool) or not isinstance(minimum_group_count, int) or minimum_group_count < 1:
        raise ValueError("minimum_group_count must be a positive integer")
    rows = _feature_rows(features, language)
    groups = sorted({row.source_recording_id for row in rows})
    required_groups = max(minimum_group_count, 3)
    if len(groups) < required_groups:
        raise ValueError(f"language {language} requires at least {required_groups} source recording groups")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * 0.60))
    calibration_count = max(1, int(len(shuffled) * 0.20))
    if train_count + calibration_count >= len(shuffled):
        calibration_count = len(shuffled) - train_count - 1
    return PolicySplit(
        seed=seed,
        language=language,
        train_source_recording_ids=tuple(shuffled[:train_count]),
        calibration_source_recording_ids=tuple(shuffled[train_count:train_count + calibration_count]),
        test_source_recording_ids=tuple(shuffled[train_count + calibration_count:]),
    )


def _validate_split_membership(rows: list[PolicyFeature], split: PolicySplit, language: str) -> None:
    if not isinstance(split, PolicySplit):
        raise ValueError("split must be a PolicySplit")
    if split.language != language:
        raise ValueError("split language does not match requested language")
    groups = set(split.train_source_recording_ids)
    groups.update(split.calibration_source_recording_ids)
    groups.update(split.test_source_recording_ids)
    feature_groups = {row.source_recording_id for row in rows}
    if feature_groups != groups:
        raise ValueError("feature source recording IDs must exactly match split")


def _validate_feature_provenance(rows: list[PolicyFeature]) -> tuple[str, dict[str, object]]:
    if not rows:
        raise ValueError("features must be non-empty")
    export_fingerprints = {row.export_fingerprint for row in rows}
    config_keys = {_config_key(row.extractor_config) for row in rows}
    if len(export_fingerprints) != 1:
        raise ValueError("mixed export fingerprints are not allowed")
    if len(config_keys) != 1:
        raise ValueError("mixed extractor configurations are not allowed")
    return next(iter(export_fingerprints)), _portable_config(rows[0].extractor_config)


def _reference_rows(references: Iterable[DiarBenchReference]) -> dict[str, DiarBenchReference]:
    rows: dict[str, DiarBenchReference] = {}
    for row in references:
        if not isinstance(row, DiarBenchReference):
            raise ValueError("references must contain DiarBenchReference records")
        decision_id = row.candidate.decision_id
        if decision_id in rows:
            raise ValueError(f"duplicate reference decision_id: {decision_id}")
        rows[decision_id] = row
    return rows


def _validate_feature_reference_identity(
    features: list[PolicyFeature], references: Mapping[str, DiarBenchReference],
) -> None:
    for feature in features:
        reference = references.get(feature.decision_id)
        if reference is None:
            raise ValueError(f"feature missing reference: {feature.decision_id}")
        candidate = reference.candidate
        if (
            candidate.decision_id != feature.decision_id
            or candidate.recording_id != feature.recording_id
            or candidate.source_recording_id != feature.source_recording_id
            or candidate.language != feature.language
            or candidate.condition != feature.condition
        ):
            raise ValueError(f"feature/reference metadata mismatch: {feature.decision_id}")


def _labels(
    decision_ids: Iterable[str], references: Mapping[str, DiarBenchReference],
) -> np.ndarray:
    values: list[float] = []
    for decision_id in decision_ids:
        outcome = references[decision_id].outcome
        if outcome not in _BINARY_OUTCOMES:
            raise ValueError(f"reference must be binary: {decision_id}")
        values.append(float(outcome == "continue"))
    return np.asarray(values, dtype=np.float64)


def calibrate_thresholds(
    probabilities: Mapping[str, float], references: Iterable[DiarBenchReference],
    calibration_ids: Iterable[str],
) -> tuple[float, float]:
    """Choose a deterministic yield/hold band using calibration references only."""
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must be a decision-id mapping")
    raw_ids = list(calibration_ids)
    ids = sorted(set(raw_ids))
    if not ids:
        raise ValueError("calibration_ids must be non-empty")
    if len(ids) != len(raw_ids):
        raise ValueError("duplicate calibration decision_id")
    reference_rows = _reference_rows(references)
    selected: list[tuple[float, str]] = []
    for decision_id in ids:
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("calibration_ids must contain non-empty strings")
        if decision_id not in probabilities:
            raise ValueError(f"calibration probability missing: {decision_id}")
        if decision_id not in reference_rows:
            raise ValueError(f"calibration reference missing: {decision_id}")
        probability = probabilities[decision_id]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"calibration probability must be within [0, 1]: {decision_id}")
        outcome = reference_rows[decision_id].outcome
        if outcome not in _BINARY_OUTCOMES:
            raise ValueError(f"reference must be binary: {decision_id}")
        selected.append((float(probability), outcome))

    continue_count = sum(outcome == "continue" for _, outcome in selected)
    yield_count = len(selected) - continue_count
    candidates: list[tuple[float, float, float, bool]] = []
    for yield_tick in range(5, 46, 5):
        yield_threshold = yield_tick / 100
        for hold_tick in range(55, 96, 5):
            hold_threshold = hold_tick / 100
            actions = [
                "yield" if probability <= yield_threshold
                else "hold" if probability >= hold_threshold
                else "uncertain"
                for probability, _ in selected
            ]
            protected_continuations = sum(
                outcome == "continue" and action != "yield"
                for action, (_, outcome) in zip(actions, selected, strict=True)
            )
            unnecessary_holds = sum(
                outcome == "yield" and action != "yield"
                for action, (_, outcome) in zip(actions, selected, strict=True)
            )
            uncertain_n = actions.count("uncertain")
            continuation_recall = protected_continuations / continue_count if continue_count else 0.0
            unnecessary_hold_rate = unnecessary_holds / yield_count if yield_count else 0.0
            score = 4 * continuation_recall - unnecessary_hold_rate - 0.0005 * uncertain_n * _GRACE_MS
            eligible = continuation_recall >= 0.80 and unnecessary_hold_rate <= 0.20
            candidates.append((score, yield_threshold, hold_threshold, eligible))

    eligible_candidates = [candidate for candidate in candidates if candidate[3]]
    # If no calibration band satisfies both hard safety/wait limits, retain the
    # deterministic highest-utility fallback rather than silently changing them.
    ranked = eligible_candidates or candidates
    score, yield_threshold, hold_threshold, _ = min(
        ranked, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
    )
    del score
    return yield_threshold, hold_threshold


def fit_policy(
    features: Iterable[PolicyFeature], references: Iterable[DiarBenchReference],
    split: PolicySplit, *, language: str,
) -> PolicyArtifact:
    """Fit only train groups, then calibrate only calibration groups."""
    rows = _feature_rows(features, language)
    _validate_split_membership(rows, split, language)
    export_fingerprint, extractor_config = _validate_feature_provenance(rows)
    reference_rows = _reference_rows(references)
    _validate_feature_reference_identity(rows, reference_rows)

    train_groups = set(split.train_source_recording_ids)
    calibration_groups = set(split.calibration_source_recording_ids)
    train_rows = [row for row in rows if row.source_recording_id in train_groups]
    calibration_rows = [row for row in rows if row.source_recording_id in calibration_groups]
    if not train_rows:
        raise ValueError("split has no train features")
    if not calibration_rows:
        raise ValueError("split has no calibration features")

    train_values = np.asarray([_feature_vector(row) for row in train_rows], dtype=np.float64)
    feature_means = train_values.mean(axis=0)
    feature_scales = train_values.std(axis=0)
    feature_scales = np.where(feature_scales == 0.0, 1.0, feature_scales)
    means = np.concatenate(([0.0], feature_means))
    scales = np.concatenate(([1.0], feature_scales))
    design = np.column_stack((np.ones(len(train_rows)), (train_values - feature_means) / feature_scales))
    labels = _labels((row.decision_id for row in train_rows), reference_rows)
    coefficients = np.zeros(7, dtype=np.float64)
    for _ in range(400):
        residual = _sigmoid(design @ coefficients) - labels
        gradient = design.T @ residual / len(train_rows)
        gradient[1:] += 0.01 * coefficients[1:]
        coefficients -= 0.05 * gradient

    calibration_values = np.asarray([_feature_vector(row) for row in calibration_rows], dtype=np.float64)
    calibration_design = np.column_stack((
        np.ones(len(calibration_rows)),
        (calibration_values - feature_means) / feature_scales,
    ))
    probabilities = {
        row.decision_id: float(probability)
        for row, probability in zip(calibration_rows, _sigmoid(calibration_design @ coefficients), strict=True)
    }
    yield_threshold, hold_threshold = calibrate_thresholds(
        probabilities, reference_rows.values(), probabilities,
    )
    return PolicyArtifact(
        policy_id=_POLICY_ID,
        language=language,
        feature_schema=POLICY_FEATURE_SCHEMA,
        export_fingerprint=export_fingerprint,
        extractor_config=extractor_config,
        coefficients=tuple(float(value) for value in coefficients),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        yield_threshold=yield_threshold,
        hold_threshold=hold_threshold,
        grace_ms=_GRACE_MS,
        hard_deadline_ms=_HARD_DEADLINE_MS,
        train_source_recording_ids=split.train_source_recording_ids,
        calibration_source_recording_ids=split.calibration_source_recording_ids,
    )

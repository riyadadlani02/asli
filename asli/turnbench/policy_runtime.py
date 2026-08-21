"""Label-free runtime decisions for calibrated TurnBench policies."""

from __future__ import annotations

import json
import math

from .policy_schema import POLICY_FEATURE_SCHEMA, PolicyArtifact, PolicyDecision, PolicyFeature


def _config_key(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _feature_vector(feature: PolicyFeature) -> tuple[float, float, float, float, float, float, float]:
    semantic_available = feature.semantic_status == "available"
    return (
        1.0,
        float(feature.pause_ms),
        feature.trailing_energy,
        feature.trailing_energy_slope,
        feature.local_speech_rate_hz,
        float(semantic_available and feature.semantic_outcome == "yield"),
        float(semantic_available),
    )


def _validate_inputs(feature: object, artifact: object) -> tuple[PolicyFeature, PolicyArtifact]:
    if not isinstance(feature, PolicyFeature):
        raise ValueError("feature must be a PolicyFeature")
    if not isinstance(artifact, PolicyArtifact):
        raise ValueError("artifact must be a PolicyArtifact")
    if feature.schema != POLICY_FEATURE_SCHEMA or artifact.feature_schema != POLICY_FEATURE_SCHEMA:
        raise ValueError("feature schema mismatch")
    if feature.language != artifact.language:
        raise ValueError("language mismatch")
    if feature.export_fingerprint != artifact.export_fingerprint:
        raise ValueError("export fingerprint mismatch")
    if _config_key(feature.extractor_config) != _config_key(artifact.extractor_config):
        raise ValueError("extractor config mismatch")
    return feature, artifact


def probability_continue(feature: PolicyFeature, artifact: PolicyArtifact) -> float:
    """Return the artifact's bounded continuation probability for one feature."""
    feature, artifact = _validate_inputs(feature, artifact)
    values = _feature_vector(feature)
    normalized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(values, artifact.means, artifact.scales, strict=True)
    )
    score = sum(
        coefficient * value
        for coefficient, value in zip(artifact.coefficients, normalized, strict=True)
    )
    if not math.isfinite(score):
        raise ValueError("non-finite policy score")
    probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid policy probability")
    return probability


def _unavailable_decision(feature: object, artifact: object, reason: str) -> PolicyDecision:
    decision_id = feature.decision_id if isinstance(feature, PolicyFeature) else "unavailable"
    policy_id = artifact.policy_id if isinstance(artifact, PolicyArtifact) else "unavailable_policy"
    return PolicyDecision(decision_id, policy_id, None, None, "unavailable", reason)


def decide_policy(feature: PolicyFeature, artifact: PolicyArtifact) -> PolicyDecision:
    """Choose hold, yield, or uncertain without reading offline references."""
    try:
        probability = probability_continue(feature, artifact)
    except (TypeError, ValueError) as exc:
        return _unavailable_decision(feature, artifact, str(exc).replace(" ", "_"))

    if probability <= artifact.yield_threshold:
        action = "yield"
    elif probability >= artifact.hold_threshold:
        action = "hold"
    else:
        action = "uncertain"
    return PolicyDecision(feature.decision_id, artifact.policy_id, probability, action, "available", None)

"""Decision-level metrics for TurnBench."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .events import first_event_ms, trace_availability
from .schema import DecisionLabel, ProviderTrace, SchemaError


_OUTCOMES = {"continue", "yield", "overlap", "unclear"}
_HEADLINE_OUTCOMES = {"continue", "yield"}
_EXCLUDED_OUTCOMES = {"overlap", "unclear"}
_STATUSES = {"scored", "excluded", "unavailable"}
_UNAVAILABLE_REASONS = {
    "provider_failed",
    "provider_timeout",
    "missing_agent_first_audio",
}


@dataclass(frozen=True)
class DecisionScore:
    """The auditable metric contribution from one labelled decision."""

    decision_id: str
    outcome: str
    status: Literal["scored", "excluded", "unavailable"]
    interrupted: bool | None
    response_delay_ms: int | None
    unavailable_reason: str | None
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject any row that cannot have exactly one metric state."""

        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise SchemaError("decision_id must be a non-empty string")
        if not isinstance(self.outcome, str) or self.outcome not in _OUTCOMES:
            raise SchemaError("outcome must be continue, yield, overlap, or unclear")
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise SchemaError("status must be scored, excluded, or unavailable")
        if self.unavailable_reason is not None and (
            not isinstance(self.unavailable_reason, str)
            or self.unavailable_reason not in _UNAVAILABLE_REASONS
        ):
            raise SchemaError("unavailable_reason is invalid")

        if self.status == "scored":
            if self.outcome not in _HEADLINE_OUTCOMES:
                raise SchemaError("scored status requires continue or yield outcome")
            if self.unavailable_reason is not None or self.exclusion_reason is not None:
                raise SchemaError("scored status cannot have unavailable or exclusion reason")
            if self.outcome == "continue":
                if not isinstance(self.interrupted, bool):
                    raise SchemaError("scored continue requires interrupted as bool")
                if self.response_delay_ms is not None:
                    raise SchemaError("scored continue cannot have response_delay_ms")
            else:
                if self.interrupted is not None:
                    raise SchemaError("scored yield cannot have interrupted")
                if isinstance(self.response_delay_ms, bool) or not isinstance(
                    self.response_delay_ms, int
                ):
                    raise SchemaError("scored yield requires integer response_delay_ms")
            return

        if self.status == "unavailable":
            if self.outcome not in _HEADLINE_OUTCOMES:
                raise SchemaError("unavailable status requires continue or yield outcome")
            if self.interrupted is not None or self.response_delay_ms is not None:
                raise SchemaError("unavailable status cannot contribute a metric")
            if self.unavailable_reason not in _UNAVAILABLE_REASONS:
                raise SchemaError("unavailable status requires unavailable_reason")
            if self.exclusion_reason is not None:
                raise SchemaError("unavailable status cannot have exclusion_reason")
            return

        if self.outcome not in _EXCLUDED_OUTCOMES:
            raise SchemaError("excluded status requires overlap or unclear outcome")
        if self.interrupted is not None or self.response_delay_ms is not None:
            raise SchemaError("excluded status cannot contribute a metric")
        if not isinstance(self.exclusion_reason, str) or not self.exclusion_reason.strip():
            raise SchemaError("excluded status requires a non-empty exclusion_reason")


def score_decision(label: DecisionLabel, trace: ProviderTrace) -> DecisionScore:
    """Score one matched label and provider trace.

    Only labelled continuations can enter premature-interruption rate (PIR).
    Only labelled yields can enter response-delay summaries.  Excluded outcomes
    and traces without audible agent audio are retained as non-headline rows.
    """

    label.validate()
    trace.validate()
    if label.decision_id != trace.decision_id:
        raise ValueError(
            "label and trace decision_id must match: "
            f"{label.decision_id} != {trace.decision_id}"
        )

    availability = trace_availability(trace)
    unavailable_reason = None if availability == "available" else availability
    if label.outcome in {"overlap", "unclear"}:
        return DecisionScore(
            label.decision_id,
            label.outcome,
            "excluded",
            None,
            None,
            unavailable_reason,
            label.exclusion_reason,
        )

    if availability != "available":
        return DecisionScore(
            label.decision_id,
            label.outcome,
            "unavailable",
            None,
            None,
            availability,
            None,
        )

    first_audio_ms = first_event_ms(trace, "agent_first_audio")
    if first_audio_ms is None:
        raise ValueError("available trace must contain agent_first_audio")
    if label.outcome == "continue":
        if label.continuation_resume_ms is None:
            raise ValueError("continue label must contain continuation_resume_ms")
        return DecisionScore(
            label.decision_id,
            "continue",
            "scored",
            first_audio_ms < label.continuation_resume_ms,
            None,
            None,
            None,
        )

    if label.outcome != "yield":
        raise ValueError(f"cannot score unsupported outcome: {label.outcome}")
    if label.true_end_ms is None:
        raise ValueError("yield label must contain true_end_ms")
    return DecisionScore(
        label.decision_id,
        "yield",
        "scored",
        None,
        first_audio_ms - label.true_end_ms,
        None,
        None,
    )


def nearest_rank(values: Sequence[int | float], percentile: float) -> int | float | None:
    """Return the nearest-rank percentile, or ``None`` for no observations."""

    if not values:
        return None
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    index = max(1, math.ceil(len(ordered) * percentile / 100)) - 1
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    """Return an explicit rate, leaving an empty denominator undefined."""

    return numerator / denominator if denominator else None


def aggregate(scores: Iterable[DecisionScore]) -> dict[str, str | int | float | None]:
    """Return one pooled-by-decision metric and availability summary.

    Every decision score represents one provider trace.  Availability therefore
    uses ``trace_n`` as its explicit denominator, including traces attached to
    excluded ``overlap`` and ``unclear`` labels.
    """

    rows = list(scores)
    for row in rows:
        if not isinstance(row, DecisionScore):
            raise SchemaError("scores must contain DecisionScore records")
        row.validate()
    scored_continue = [
        row for row in rows if row.status == "scored" and row.outcome == "continue"
    ]
    scored_yield = [
        row for row in rows if row.status == "scored" and row.outcome == "yield"
    ]
    delays = [
        row.response_delay_ms
        for row in scored_yield
        if row.response_delay_ms is not None
    ]
    interruptions = [row.interrupted for row in scored_continue]
    unavailable = [row for row in rows if row.unavailable_reason is not None]
    trace_n = len(rows)
    unavailable_n = len(unavailable)
    available_n = trace_n - unavailable_n
    provider_failed_n = sum(
        row.unavailable_reason == "provider_failed" for row in unavailable
    )
    provider_timeout_n = sum(
        row.unavailable_reason == "provider_timeout" for row in unavailable
    )
    missing_agent_first_audio_n = sum(
        row.unavailable_reason == "missing_agent_first_audio" for row in unavailable
    )
    return {
        "aggregation": "micro_by_decision",
        "decision_n": len(rows),
        "pir_n": len(interruptions),
        "pir": (sum(value is True for value in interruptions) / len(interruptions))
        if interruptions
        else None,
        "delay_n": len(delays),
        "delay_p50_ms": nearest_rank(delays, 50),
        "delay_p95_ms": nearest_rank(delays, 95),
        "excluded_n": sum(row.status == "excluded" for row in rows),
        "trace_n": trace_n,
        "available_n": available_n,
        "unavailable_n": unavailable_n,
        "availability_rate": _rate(available_n, trace_n),
        "provider_failed_n": provider_failed_n,
        "provider_failed_rate": _rate(provider_failed_n, trace_n),
        "provider_timeout_n": provider_timeout_n,
        "provider_timeout_rate": _rate(provider_timeout_n, trace_n),
        "missing_agent_first_audio_n": missing_agent_first_audio_n,
        "missing_agent_first_audio_rate": _rate(
            missing_agent_first_audio_n, trace_n
        ),
    }


def bootstrap_by_recording(
    scores: Iterable[DecisionScore],
    labels: Mapping[str, DecisionLabel],
    *,
    draws: int,
    seed: int,
) -> dict[str, float] | None:
    """Bootstrap PIR by resampling whole source recordings, not decisions."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    rows = list(scores)
    for score in rows:
        if not isinstance(score, DecisionScore):
            raise SchemaError("scores must contain DecisionScore records")
        score.validate()
    eligible = [
        score
        for score in rows
        if score.status == "scored" and score.outcome == "continue"
    ]
    groups: dict[str, list[DecisionScore]] = {}
    for score in eligible:
        if score.decision_id not in labels:
            raise ValueError(f"missing label for decision_id: {score.decision_id}")
        label = labels[score.decision_id]
        if not isinstance(label, DecisionLabel):
            raise SchemaError("labels must contain DecisionLabel records")
        label.validate()
        if label.decision_id != score.decision_id:
            raise ValueError(
                "label mapping key does not match label decision_id: "
                f"{score.decision_id} != {label.decision_id}"
            )
        if label.outcome != score.outcome:
            raise ValueError(
                "label outcome does not match score outcome: "
                f"{label.outcome} != {score.outcome}"
            )
        groups.setdefault(label.source_recording_id, []).append(score)
    if len(groups) < 2:
        return None

    rng = random.Random(seed)
    group_rows = [groups[source_id] for source_id in sorted(groups)]
    samples: list[float] = []
    for _ in range(draws):
        sampled = [row for _ in group_rows for row in rng.choice(group_rows)]
        samples.append(
            sum(row.interrupted is True for row in sampled) / len(sampled)
        )
    low = nearest_rank(samples, 2.5)
    high = nearest_rank(samples, 97.5)
    if not isinstance(low, float) or not isinstance(high, float):
        raise ValueError("bootstrap interval requires floating-point samples")
    return {"low": low, "high": high}

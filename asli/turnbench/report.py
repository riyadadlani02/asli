"""Aggregation and input validation for complete TurnBench reports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Iterable, TypeVar

from .candidates import resolve_mechanical_boundary
from .schema import DecisionLabel, ProviderTrace, Recording
from .score import DecisionScore, aggregate, bootstrap_by_recording, score_decision


T = TypeVar("T")

_MACRO_LANGUAGE_METRICS = (
    "pir",
    "delay_p50_ms",
    "delay_p95_ms",
    "availability_rate",
    "provider_failed_rate",
    "provider_timeout_rate",
    "missing_agent_first_audio_rate",
)


def _normalize_json_value(
    value: object, *, path: str, active_containers: set[int]
) -> object:
    """Copy one value into the standard JSON domain, rejecting cycles."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return value

    if isinstance(value, (list, tuple, Mapping)):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{path} must not contain cycles")
        active_containers.add(container_id)
        try:
            if isinstance(value, Mapping):
                normalized: dict[str, object] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{path} mappings must have string keys")
                    normalized[key] = _normalize_json_value(
                        item,
                        path=f"{path}.{key}",
                        active_containers=active_containers,
                    )
                return normalized
            return [
                _normalize_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(container_id)

    raise ValueError(f"{path} must contain only standard JSON value types")


def _normalize_config(config: object) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    normalized = _normalize_json_value(
        config, path="config", active_containers=set()
    )
    if not isinstance(normalized, dict):
        raise ValueError("config must normalize to a JSON object")
    return normalized


def _unique_by_id(rows: Iterable[T], attribute: str) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for row in rows:
        key = getattr(row, attribute)
        if key in indexed:
            raise ValueError(f"duplicate {attribute}: {key}")
        indexed[key] = row
    return indexed


def _micro_summary(
    rows: Iterable[DecisionScore],
    labels: Mapping[str, DecisionLabel],
    recordings: Mapping[str, Recording],
) -> dict[str, object]:
    """Add corpus provenance counts to one pooled decision summary."""

    grouped_rows = list(rows)
    summary: dict[str, object] = aggregate(grouped_rows)
    grouped_labels = [labels[row.decision_id] for row in grouped_rows]
    grouped_recordings = [
        recordings[label.recording_id] for label in grouped_labels
    ]
    summary.update(
        {
            "source_recording_n": len(
                {
                    recording.source_recording_id
                    for recording in grouped_recordings
                }
            ),
            "fixture_label_n": sum(
                label.final_label == "fixture" for label in grouped_labels
            ),
            "adjudicated_label_n": sum(
                label.final_label != "fixture" for label in grouped_labels
            ),
        }
    )
    return summary


def _macro_by_language(
    language_summaries: Mapping[str, Mapping[str, object]],
    micro_summary: Mapping[str, object],
) -> dict[str, object]:
    """Average defined per-language metrics without weighting by decision count."""

    result = dict(micro_summary)
    result["aggregation"] = "macro_by_language"
    for metric in _MACRO_LANGUAGE_METRICS:
        values = [
            summary[metric]
            for summary in language_summaries.values()
            if summary[metric] is not None
        ]
        result[metric] = sum(values) / len(values) if values else None
        result[f"{metric}_language_n"] = len(values)
    return result


def _micro_pir_bootstrap(
    rows: Iterable[DecisionScore], labels: Mapping[str, DecisionLabel]
) -> dict[str, str | float | None]:
    """Describe the source-group interval for pooled decision-level PIR."""

    interval = bootstrap_by_recording(rows, labels, draws=1000, seed=0)
    return {
        "aggregation": "micro_by_decision",
        "metric": "pir",
        "low": interval["low"] if interval is not None else None,
        "high": interval["high"] if interval is not None else None,
    }


def _build_report(
    scores: Iterable[DecisionScore],
    labels: Mapping[str, DecisionLabel],
    recordings: Mapping[str, Recording],
    *,
    provider: str,
    config: Mapping[str, object],
    run_id: str | None = None,
) -> dict[str, object]:
    """Build a report after ``score_inputs`` validates all source records."""

    rows = list(scores)
    by_language: dict[str, dict[str, object]] = {}
    by_condition: dict[str, dict[str, object]] = {}
    by_source: dict[str, dict[str, object]] = {}
    languages = sorted(
        {recordings[labels[row.decision_id].recording_id].language for row in rows}
    )
    conditions = sorted(
        {recordings[labels[row.decision_id].recording_id].condition for row in rows}
    )
    sources = sorted(
        {
            recordings[labels[row.decision_id].recording_id].source_recording_id
            for row in rows
        }
    )
    for language in languages:
        by_language[language] = _micro_summary(
            (
                row
                for row in rows
                if recordings[labels[row.decision_id].recording_id].language
                == language
            ),
            labels,
            recordings,
        )
    for condition in conditions:
        by_condition[condition] = _micro_summary(
            (
                row
                for row in rows
                if recordings[labels[row.decision_id].recording_id].condition
                == condition
            ),
            labels,
            recordings,
        )
    for source in sources:
        by_source[source] = _micro_summary(
            (
                row
                for row in rows
                if recordings[
                    labels[row.decision_id].recording_id
                ].source_recording_id
                == source
            ),
            labels,
            recordings,
        )
    micro_overall = _micro_summary(rows, labels, recordings)
    label_provenance = {
        "fixture_label_n": micro_overall["fixture_label_n"],
        "adjudicated_label_n": micro_overall["adjudicated_label_n"],
    }
    return {
        "schema": "turnbench.report.v1",
        "run_id": run_id,
        "provider": provider,
        "config": dict(config),
        "label_provenance": label_provenance,
        "overall": _macro_by_language(by_language, micro_overall),
        "micro_overall": micro_overall,
        "by_language": by_language,
        "by_condition": by_condition,
        "by_source_recording": by_source,
        "pir_bootstrap_95": _micro_pir_bootstrap(rows, labels),
    }


def score_inputs(
    recordings: Iterable[Recording],
    labels: Iterable[DecisionLabel],
    traces: Iterable[ProviderTrace],
    *,
    provider: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate linked inputs, score each decision, and build a report."""

    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    normalized_config = _normalize_config(config)

    recordings_by_id = _unique_by_id(recordings, "recording_id")
    labels_by_id = _unique_by_id(labels, "decision_id")
    traces_by_decision = _unique_by_id(traces, "decision_id")
    for recording in recordings_by_id.values():
        recording.validate()
    for label in labels_by_id.values():
        label.validate()
    for trace in traces_by_decision.values():
        trace.validate()
    provenance = {
        "fixture" if label.final_label == "fixture" else "adjudicated"
        for label in labels_by_id.values()
    }
    if len(provenance) > 1:
        raise ValueError("mixed fixture and adjudicated labels are not allowed")
    run_ids = {trace.run_id for trace in traces_by_decision.values()}
    if len(run_ids) > 1:
        raise ValueError("mixed run_id values are not allowed in one report")
    for trace in traces_by_decision.values():
        if trace.provider != provider:
            raise ValueError(
                f"trace provider does not match report provider: {trace.provider} != {provider}"
            )
    for label in labels_by_id.values():
        recording = recordings_by_id.get(label.recording_id)
        if recording is None:
            raise ValueError(f"unknown recording_id: {label.recording_id}")
        if label.source_recording_id != recording.source_recording_id:
            raise ValueError(f"source_recording_id does not match recording: {label.decision_id}")
        if label.target_speaker_id != recording.target_speaker_id:
            raise ValueError(f"target_speaker_id does not match recording: {label.decision_id}")
        if label.final_label == "fixture" and recording.condition != "fixture":
            raise ValueError(
                f"fixture label requires a fixture recording: {label.decision_id}"
            )
        if label.final_label != "fixture" and recording.condition == "fixture":
            raise ValueError(
                f"adjudicated label cannot reference a fixture recording: {label.decision_id}"
            )
        if not any(
            turn.speaker_id == label.target_speaker_id
            and turn.end_ms == label.previous_speech_end_ms
            for turn in recording.turns
        ):
            raise ValueError(
                f"previous_speech_end_ms does not match a target-speaker turn: {label.decision_id}"
            )
        matching_next = [
            turn
            for turn in recording.turns
            if turn.start_ms == label.next_event_start_ms
        ]
        if not matching_next:
            raise ValueError(
                f"next_event_start_ms does not match a source turn: {label.decision_id}"
            )
        if label.outcome == "continue" and not any(
            turn.speaker_id == label.target_speaker_id for turn in matching_next
        ):
            raise ValueError(
                f"continue next event must belong to target speaker: {label.decision_id}"
            )
        if label.outcome in {"continue", "yield"}:
            try:
                boundary = resolve_mechanical_boundary(
                    recording,
                    target_speaker_id=label.target_speaker_id,
                    previous_speech_end_ms=label.previous_speech_end_ms,
                    next_event_start_ms=label.next_event_start_ms,
                )
            except ValueError as error:
                raise ValueError(f"{error}: {label.decision_id}") from error
            if label.outcome == "continue" and any(
                recording.turns[index].speaker_id != label.target_speaker_id
                for index in boundary.next_turn_indices
            ):
                raise ValueError(
                    "continue earliest event includes a non-target speaker: "
                    f"{label.decision_id}"
                )
        if label.decision_id not in traces_by_decision:
            raise ValueError(f"missing trace for decision_id: {label.decision_id}")
    for decision_id in traces_by_decision:
        if decision_id not in labels_by_id:
            raise ValueError(f"unknown decision_id: {decision_id}")
    scores = [
        score_decision(label, traces_by_decision[label.decision_id])
        for label in labels_by_id.values()
    ]
    return _build_report(
        scores,
        labels_by_id,
        recordings_by_id,
        provider=provider,
        config=normalized_config,
        run_id=next(iter(run_ids), None),
    )

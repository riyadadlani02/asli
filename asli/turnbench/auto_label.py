"""Strict, label-free semantic-VAD observations for DiarBench candidates."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from ..drive import OpenAIWS
from ..spec import CallSpec
from .auto_schema import AutoPrediction, DiarBenchCandidate
from .report import _normalize_config


_CANDIDATE_LANGUAGE = object()


@dataclass(frozen=True)
class EndpointObservation:
    """The provider-only outcome of observing one bounded audio window."""

    endpoint_ms: int | None
    missing_timestamp: bool
    error: str | None
    endpoint_timestamps: tuple[int, ...] = ()


class OpenAISemanticObserver:
    """Adapt OpenAI semantic VAD to the small, timestamp-strict observation seam."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-transcribe",
        adapter_factory: Callable[..., OpenAIWS] = OpenAIWS,
    ) -> None:
        self.adapter = adapter_factory(
            model=model,
            turn_detection="semantic_vad",
            trailing_silence_ms=0,
            require_endpoint_timestamps=True,
        )

    def __call__(self, pcm: np.ndarray, rate: int, language: str | None) -> EndpointObservation:
        return self.observe(pcm, rate, language)

    def observe(self, pcm: np.ndarray, rate: int, language: str | None) -> EndpointObservation:
        self.adapter.rate = rate
        if language is not None:
            self.adapter.lang = language.split("-")[0]
        spec = CallSpec(
            id="turnbench-semantic-observation",
            segments=[],
            entity_type="turnbench",
            canonical="",
            lang=language or self.adapter.lang,
        )
        try:
            result = asyncio.run(self.adapter.run(pcm, spec))
        except Exception as exc:
            return EndpointObservation(None, False, f"{type(exc).__name__}: {exc}")
        if result.error:
            return EndpointObservation(None, False, result.error)
        if self.adapter.missing_endpoint_timestamp:
            return EndpointObservation(None, True, None)
        endpoints = tuple(event.t_ms for event in result.events if event.kind == "speech_end")
        return EndpointObservation(min(endpoints) if endpoints else None, False, None, endpoints)


def _unavailable(
    candidate: DiarBenchCandidate,
    *,
    run_id: str,
    agent: str,
    model: str,
    config: dict[str, object],
    reason: str,
) -> AutoPrediction:
    return AutoPrediction(
        decision_id=candidate.decision_id,
        run_id=run_id,
        agent=agent,
        model=model,
        config=config,
        status="unavailable",
        outcome=None,
        endpoint_ms=None,
        unavailable_reason=reason,
    )


def predict_candidate(
    candidate: DiarBenchCandidate,
    *,
    read_audio: Callable[[str], tuple[np.ndarray, int]],
    observe: Callable[[np.ndarray, int, str | None], EndpointObservation],
    run_id: str,
    agent: str,
    model: str,
    config: Mapping[str, object],
    provider_language: str | None | object = _CANDIDATE_LANGUAGE,
) -> AutoPrediction:
    """Predict from audio ending exactly at the candidate observation boundary."""

    normalized_config = _normalize_config(config)
    try:
        pcm, rate = read_audio(candidate.audio_path)
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise ValueError("invalid sample rate")
        start = candidate.context_start_ms * rate // 1000
        end = candidate.observation_end_ms * rate // 1000
        if start < 0 or end <= start or end > len(pcm):
            raise ValueError("candidate audio boundary is unavailable")
        window = pcm[start:end]
    except Exception:
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="audio_boundary_error",
        )

    try:
        language = candidate.language if provider_language is _CANDIDATE_LANGUAGE else provider_language
        observation = observe(window, rate, language)
    except Exception:
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="observer_error",
        )
    if observation.error is not None:
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="observer_error",
        )
    if observation.missing_timestamp:
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="missing_endpoint_timestamp",
        )
    timestamps = observation.endpoint_timestamps
    if not isinstance(timestamps, tuple):
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="invalid_endpoint_timestamp",
        )
    if timestamps:
        if any(
            isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0
            for timestamp in timestamps
        ):
            return _unavailable(
                candidate, run_id=run_id, agent=agent, model=model,
                config=normalized_config, reason="invalid_endpoint_timestamp",
            )
        in_window = sorted(
            candidate.context_start_ms + timestamp
            for timestamp in timestamps
            if candidate.previous_speech_end_ms < candidate.context_start_ms + timestamp < candidate.observation_end_ms
        )
        if in_window:
            return AutoPrediction(
                decision_id=candidate.decision_id, run_id=run_id, agent=agent, model=model,
                config=normalized_config, status="available", outcome="yield",
                endpoint_ms=in_window[0], unavailable_reason=None,
            )
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="endpoint_outside_window",
        )
    if observation.endpoint_ms is None:
        return AutoPrediction(
            decision_id=candidate.decision_id, run_id=run_id, agent=agent, model=model,
            config=normalized_config, status="available", outcome="continue",
            endpoint_ms=None, unavailable_reason=None,
        )
    if isinstance(observation.endpoint_ms, bool) or not isinstance(observation.endpoint_ms, int):
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="invalid_endpoint_timestamp",
        )

    endpoint_ms = candidate.context_start_ms + observation.endpoint_ms
    if not candidate.previous_speech_end_ms < endpoint_ms < candidate.observation_end_ms:
        return _unavailable(
            candidate, run_id=run_id, agent=agent, model=model,
            config=normalized_config, reason="endpoint_outside_window",
        )
    return AutoPrediction(
        decision_id=candidate.decision_id, run_id=run_id, agent=agent, model=model,
        config=normalized_config, status="available", outcome="yield",
        endpoint_ms=endpoint_ms, unavailable_reason=None,
    )

"""Deterministic, label-free local features for the TurnBench policy lane."""

from __future__ import annotations

import hashlib
import json
import wave
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .auto_report import DiarBenchExportProvenance, validate_export_candidates
from .auto_schema import AutoPrediction, DiarBenchCandidate
from .policy_schema import POLICY_EXTRACTOR_CONFIG, PolicyFeature


_EXTRACTOR_CONFIG = POLICY_EXTRACTOR_CONFIG


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono PCM WAV without involving a provider or recognizer."""
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise ValueError("audio must be mono PCM WAV")
        if source.getcomptype() != "NONE":
            raise ValueError("audio must be uncompressed PCM WAV")
        width = source.getsampwidth()
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
        if dtype is None:
            raise ValueError("audio sample width must be 8, 16, or 32 bits")
        return np.frombuffer(source.readframes(source.getnframes()), dtype=dtype).copy(), source.getframerate()


_DEFAULT_READ_AUDIO = read_audio


def _read_audio_through(path: Path, observation_end_ms: int) -> tuple[np.ndarray, int]:
    """Decode no more WAV audio than is visible to one candidate."""
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise ValueError("audio must be mono PCM WAV")
        if source.getcomptype() != "NONE":
            raise ValueError("audio must be uncompressed PCM WAV")
        width = source.getsampwidth()
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
        if dtype is None:
            raise ValueError("audio sample width must be 8, 16, or 32 bits")
        rate = source.getframerate()
        end_frame = _sample_index(observation_end_ms, rate)
        if source.getnframes() < end_frame:
            raise ValueError("audio does not reach observation_end_ms")
        return np.frombuffer(source.readframes(end_frame), dtype=dtype).copy(), rate


def _sample_index(milliseconds: int, rate: int) -> int:
    return milliseconds * rate // 1000


def _slice_candidate_audio(
    candidate: DiarBenchCandidate, pcm: np.ndarray, rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return preceding context speech and its bounded natural pause."""
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError("audio sample rate must be a positive integer")
    samples = np.asarray(pcm)
    if samples.ndim != 1:
        raise ValueError("audio PCM must be one-dimensional")
    end = _sample_index(candidate.observation_end_ms, rate)
    if len(samples) < end:
        raise ValueError(f"audio does not reach observation_end_ms: {candidate.decision_id}")
    context_start = _sample_index(candidate.context_start_ms, rate)
    previous_end = _sample_index(candidate.previous_speech_end_ms, rate)
    return samples[context_start:previous_end], samples[previous_end:end]


def _pcm_scale(pcm: np.ndarray) -> float:
    if np.issubdtype(pcm.dtype, np.integer):
        return float(max(abs(np.iinfo(pcm.dtype).min), np.iinfo(pcm.dtype).max))
    return 1.0


def _trailing_features(speech_pcm: np.ndarray, rate: int) -> tuple[float, float, int, float]:
    """Calculate bounded voiced-frame features from preceding speech only.

    ``local_speech_rate_hz`` is the v1 voiced-onset proxy: the number of
    contiguous voiced regions per visible second in the trailing lookback.
    """
    lookback = min(len(speech_pcm), _sample_index(1000, rate))
    samples = np.asarray(speech_pcm[-lookback:])
    frame_samples = _sample_index(20, rate)
    if frame_samples <= 0:
        raise ValueError("audio sample rate is too low for 20 ms frames")
    frame_count = len(samples) // frame_samples
    if not frame_count:
        return 0.0, 0.0, 0, 0.0
    frames = samples[:frame_count * frame_samples].astype(np.float64).reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    maximum = float(np.max(rms))
    if maximum == 0.0:
        return 0.0, 0.0, 0, 0.0
    scale = _pcm_scale(samples)
    final_energy = float(rms[-1] / scale)
    energy_slope = float((rms[-1] - rms[0]) / scale)
    voiced = rms >= maximum * 0.1
    voiced_count = int(np.count_nonzero(voiced))
    voiced_ms = voiced_count * 20
    voiced_onsets = int(np.count_nonzero(voiced & np.concatenate(([True], ~voiced[:-1]))))
    visible_seconds = frame_count * 20 / 1000
    speech_rate = voiced_onsets / visible_seconds if visible_seconds else 0.0
    return final_energy, energy_slope, voiced_ms, speech_rate


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audio_fingerprint(rate: int, bounded_pcm: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rate).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(bounded_pcm).tobytes())
    return digest.hexdigest()


def _index_semantic_predictions(
    predictions: Iterable[AutoPrediction], candidates: dict[str, DiarBenchCandidate],
    export_provenance: DiarBenchExportProvenance,
) -> dict[str, AutoPrediction]:
    indexed: dict[str, AutoPrediction] = {}
    run: tuple[str, str, str, str] | None = None
    for prediction in predictions:
        if not isinstance(prediction, AutoPrediction):
            raise ValueError("semantic_predictions must contain AutoPrediction records")
        if prediction.decision_id in indexed:
            raise ValueError(f"duplicate prediction decision_id: {prediction.decision_id}")
        candidate = candidates.get(prediction.decision_id)
        if candidate is None:
            raise ValueError(f"prediction missing candidate: {prediction.decision_id}")
        normalized_config = json.dumps(
            prediction.config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
        provenance = (prediction.run_id, prediction.agent, prediction.model, normalized_config)
        if run is None:
            run = provenance
        elif provenance != run:
            raise ValueError("mixed semantic prediction run/config values are not allowed")
        if "context_ms" in prediction.config and prediction.config["context_ms"] != export_provenance.context_ms:
            raise ValueError("prediction config context_ms does not match export provenance")
        if prediction.status == "available" and prediction.outcome == "yield":
            if prediction.endpoint_ms is None or not (
                candidate.previous_speech_end_ms < prediction.endpoint_ms < candidate.observation_end_ms
            ):
                raise ValueError(f"yield endpoint outside candidate window: {prediction.decision_id}")
        indexed[prediction.decision_id] = prediction
    return indexed


def extract_policy_features(
    candidates: Iterable[DiarBenchCandidate], *,
    export_provenance: DiarBenchExportProvenance,
    semantic_predictions: Iterable[AutoPrediction] = (),
    read_audio: Callable[[Path], tuple[np.ndarray, int]] = read_audio,
) -> list[PolicyFeature]:
    """Extract local bounded-window features without accepting any reference rows."""
    candidate_rows = validate_export_candidates(candidates, export_provenance=export_provenance)
    prediction_rows = _index_semantic_predictions(semantic_predictions, candidate_rows, export_provenance)
    export_fingerprint = _fingerprint(export_provenance.to_dict())
    rows: list[PolicyFeature] = []
    for decision_id, candidate in candidate_rows.items():
        if read_audio is _DEFAULT_READ_AUDIO:
            pcm, rate = _read_audio_through(
                Path(candidate.audio_path), candidate.observation_end_ms,
            )
        else:
            pcm, rate = read_audio(Path(candidate.audio_path))
        speech_pcm, _pause_pcm = _slice_candidate_audio(candidate, pcm, rate)
        bounded_pcm = np.asarray(pcm)[:_sample_index(candidate.observation_end_ms, rate)]
        energy, slope, speech_ms, rate_hz = _trailing_features(speech_pcm, rate)
        prediction = prediction_rows.get(decision_id)
        if prediction is None:
            semantic_status, outcome, endpoint_offset = "absent", None, None
        elif prediction.status == "unavailable":
            semantic_status, outcome, endpoint_offset = "unavailable", None, None
        else:
            semantic_status, outcome = "available", prediction.outcome
            endpoint_offset = (
                prediction.endpoint_ms - candidate.previous_speech_end_ms
                if outcome == "yield" and prediction.endpoint_ms is not None else None
            )
        rows.append(PolicyFeature(
            decision_id=candidate.decision_id, recording_id=candidate.recording_id,
            source_recording_id=candidate.source_recording_id, language=candidate.language,
            condition=candidate.condition, export_fingerprint=export_fingerprint,
            extractor_config=dict(_EXTRACTOR_CONFIG),
            audio_fingerprint=_audio_fingerprint(rate, bounded_pcm),
            pause_ms=candidate.observation_end_ms - candidate.previous_speech_end_ms,
            trailing_energy=energy, trailing_energy_slope=slope,
            trailing_speech_ms=speech_ms, local_speech_rate_hz=rate_hz,
            semantic_status=semantic_status, semantic_outcome=outcome,
            semantic_endpoint_offset_ms=endpoint_offset,
        ))
    return sorted(rows, key=lambda row: row.decision_id)

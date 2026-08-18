"""Persona -> audio, segment by segment.

The one non-obvious decision in this file: each segment is synthesised as its own TTS
call and the pauses between them are inserted by us as exact silence. Never hand a
whole disfluent sentence to a TTS engine and try to find the gap afterwards — do it
this way and every boundary, including the true end of the utterance, is known by
construction. No forced alignment, ever. That is what makes PIR cheap to measure.
"""

from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path

import httpx
import numpy as np

from .spec import SAMPLE_RATE, CallSpec

MODEL = "eleven_multilingual_v2"
CACHE = Path(os.getenv("ASLI_CACHE", Path(__file__).parent.parent / "demo" / "cache"))


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.int16)


def tts(text: str, voice: str | None = None) -> np.ndarray:
    """One segment -> int16 PCM @16k. Cached on disk: re-runs cost no credits."""
    voice = voice or os.environ["ELEVEN_VOICE_ID"]
    key = hashlib.sha256(f"{MODEL}|{voice}|{text}".encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = CACHE / f"{key}.pcm"
    if hit.exists():
        return np.frombuffer(hit.read_bytes(), dtype="<i2").copy()

    base = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/").removesuffix("/v1")
    r = httpx.post(
        f"{base}/v1/text-to-speech/{voice}",
        params={"model_id": MODEL, "output_format": f"pcm_{SAMPLE_RATE}"},
        # normalization off: we are testing how numerals survive, so the numerals
        # must reach the voice exactly as authored.
        json={"text": text, "model_id": MODEL, "apply_text_normalization": "off"},
        headers={"xi-api-key": os.environ["ELEVEN_API_KEY"]},
        timeout=60,
    )
    r.raise_for_status()
    hit.write_bytes(r.content)
    return np.frombuffer(r.content, dtype="<i2").copy()


def trim(pcm: np.ndarray, floor: int = 200) -> np.ndarray:
    """Strip the silence TTS pads onto each clip.

    Without this the engine's own trailing silence adds to the pause we splice in,
    the real gap is longer than the spec says, and every timing claim in the harness
    is quietly wrong. Trim first, then splice, and the pause is exactly the pause.
    """
    loud = np.flatnonzero(np.abs(pcm) > floor)
    return pcm[loud[0] : loud[-1] + 1] if len(loud) else pcm


def render(spec: CallSpec, voice: str | None = None) -> np.ndarray:
    """Synthesise every segment, splice in the pauses, stamp the spec with the truth."""
    parts, bounds, cursor = [], [], 0
    for seg in spec.segments:
        pcm = trim(tts(seg.text, voice))
        dur = int(len(pcm) * 1000 / SAMPLE_RATE)
        bounds.append((cursor, cursor + dur))
        parts.append(pcm)
        cursor += dur
        if seg.pause_after_ms:
            parts.append(_silence(seg.pause_after_ms))
            cursor += seg.pause_after_ms

    spec.seg_bounds_ms = bounds
    spec.true_end_ms = bounds[-1][1]  # end of last *speech*, excluding trailing silence
    spec.voice = voice or os.environ.get("ELEVEN_VOICE_ID", "")
    return np.concatenate(parts)


def write_wav(path: str | Path, pcm: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy(), w.getframerate()

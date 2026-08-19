"""Fit the synthetic hesitation to real speech.

Fully synthetic disfluency timing measures the TTS engine's idea of Hindi hesitation,
not a Hindi speaker's. Until the injected pauses are drawn from a distribution observed
in real unscripted telephone speech, PIR reads as "at this pause length" and never as a
field rate. This module closes that gap: extract mid-utterance pause lengths from a
corpus, fit them, sample from the fit, and report how well the fit holds (KS).

Scope, stated plainly: without word-aligned transcripts this measures *all*
mid-utterance pauses, not specifically filler-adjacent ones. That is a superset of the
target distribution and still far better grounded than a constant. Filler-conditioned
fitting needs alignment and is the next step, not this one.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from .spec import SAMPLE_RATE


ANALYSIS_FRAME_MS = 20  # pause lengths are the measurement here, so resolve them
                        # finely — the recogniser's 512-sample frame is 64ms at 8kHz,
                        # which would quantise a 700ms pause by ±9%.


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Any format ffmpeg understands, at its native sample rate."""
    if path.suffix.lower() == ".wav":
        from .synth import read_wav

        return read_wav(path)
    rate = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip() or 0)
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1",
         "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").copy(), rate


def frame_rms(pcm: np.ndarray, frame: int) -> np.ndarray:
    x = pcm.astype(np.float64) / 32768.0
    n = len(x) // frame
    if n <= 0:
        return np.zeros(0)
    return np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))


def adaptive_threshold(rms: np.ndarray, min_contrast: float = 3.0) -> float | None:
    """A speech/silence cut derived from the recording's own energy distribution.

    A fixed threshold works on synthesised audio and fails on real recordings, whose
    noise floors differ by orders of magnitude between a quiet handset and a roadside
    call. Floor and speech level are taken as low/high percentiles and the cut is
    placed between them geometrically.

    Returns None when the file has too little contrast to separate the two at all —
    all-noise or all-speech — so it can be skipped rather than contribute nonsense.
    """
    if len(rms) < 20:
        return None
    floor = float(np.percentile(rms, 10))
    speech = float(np.percentile(rms, 95))
    if speech <= 0:
        return None
    if floor <= 0:
        # digital silence: nothing to separate from, so any real energy is speech
        return speech * 0.05
    if speech < floor * min_contrast:
        return None
    return float(np.sqrt(floor * speech))


def pause_lengths_ms(pcm: np.ndarray, rate: int = SAMPLE_RATE, threshold: float | None = None,
                     min_ms: int = 80, max_ms: int = 4000,
                     frame_ms: int = ANALYSIS_FRAME_MS) -> list[int]:
    """Silence runs bracketed by speech on both sides — mid-utterance pauses only.

    Leading and trailing silence are excluded: they are recording margin, not
    hesitation, and including them would drag the distribution upward.

    `threshold=None` derives the cut per recording (see `adaptive_threshold`).
    """
    frame = max(1, int(rate * frame_ms / 1000))
    rms = frame_rms(pcm, frame)
    if threshold is None:
        threshold = adaptive_threshold(rms)
        if threshold is None:
            return []
    loud = list(rms >= threshold)
    if not any(loud):
        return []
    first, last = loud.index(True), len(loud) - 1 - loud[::-1].index(True)

    out, run = [], 0
    for v in loud[first:last + 1]:
        if v:
            if run:
                ms = int(run * frame_ms)
                if min_ms <= ms <= max_ms:
                    out.append(ms)
            run = 0
        else:
            run += 1
    return out


def fit_lognormal(samples: list[int]) -> tuple[float, float]:
    """Pause durations are right-skewed and positive, so fit in log space."""
    logs = np.log(np.asarray(samples, dtype=float))
    return float(logs.mean()), float(logs.std(ddof=1))


def _lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2))))


def ks_statistic(samples: list[int], mu: float, sigma: float) -> float:
    """Max gap between the empirical CDF and the fit. Report it; do not hide it."""
    xs = sorted(samples)
    n = len(xs)
    return max(max(abs((i + 1) / n - _lognormal_cdf(x, mu, sigma)),
                   abs(_lognormal_cdf(x, mu, sigma) - i / n)) for i, x in enumerate(xs))


def sample_pauses(n: int, mu: float, sigma: float, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(round(v)) for v in rng.lognormal(mu, sigma, n)]


def fit_corpus(wavs: list[Path], out: Path | None = None) -> dict:
    pauses: list[int] = []
    used = skipped = 0
    for w in wavs:
        try:
            pcm, rate = read_audio(w)
        except Exception:
            skipped += 1
            continue
        found = pause_lengths_ms(pcm, rate)
        if found:
            used += 1
            pauses += found
        else:
            skipped += 1
    if len(pauses) < 20:
        raise SystemExit(f"only {len(pauses)} pauses found — need a real corpus, not a sample")

    mu, sigma = fit_lognormal(pauses)
    fit = {"n_pauses": len(pauses), "n_files": len(wavs), "files_used": used,
           "files_skipped": skipped, "mu": mu, "sigma": sigma,
           "ks": ks_statistic(pauses, mu, sigma),
           "median_ms": int(np.median(pauses)),
           "p25_ms": int(np.percentile(pauses, 25)),
           "p75_ms": int(np.percentile(pauses, 75)),
           "p90_ms": int(np.percentile(pauses, 90))}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fit, indent=2))
    return fit

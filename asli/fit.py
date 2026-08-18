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
from pathlib import Path

import numpy as np

from .drive import FRAME_SAMPLES
from .spec import SAMPLE_RATE


def pause_lengths_ms(pcm: np.ndarray, rate: int = SAMPLE_RATE, threshold: float = 0.02,
                     min_ms: int = 80, max_ms: int = 4000) -> list[int]:
    """Silence runs bracketed by speech on both sides — mid-utterance pauses only.

    Leading and trailing silence are excluded: they are recording margin, not
    hesitation, and including them would drag the distribution upward.
    """
    x = pcm.astype(np.float64) / 32768.0
    frame_ms = FRAME_SAMPLES * 1000 / rate
    loud = [float(np.sqrt(np.mean(x[i:i + FRAME_SAMPLES] ** 2))) >= threshold
            for i in range(0, len(x) - FRAME_SAMPLES, FRAME_SAMPLES)]
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
    from .synth import read_wav

    pauses: list[int] = []
    for w in wavs:
        pcm, rate = read_wav(w)
        pauses += pause_lengths_ms(pcm, rate)
    if len(pauses) < 20:
        raise SystemExit(f"only {len(pauses)} pauses found — need a real corpus, not a sample")

    mu, sigma = fit_lognormal(pauses)
    fit = {"n_pauses": len(pauses), "n_files": len(wavs), "mu": mu, "sigma": sigma,
           "ks": ks_statistic(pauses, mu, sigma),
           "median_ms": int(np.median(pauses)),
           "p25_ms": int(np.percentile(pauses, 25)),
           "p75_ms": int(np.percentile(pauses, 75)),
           "p90_ms": int(np.percentile(pauses, 90))}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fit, indent=2))
    return fit

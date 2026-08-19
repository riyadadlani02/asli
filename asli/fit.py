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


GATES = (300, 400, 500, 700, 900, 1200)
DEFAULT_GATE_MS = 500      # Sarvam's documented silence_duration_ms default

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


def fit_corpus(wavs: list[Path], out: Path | None = None, cutoff_ms: int = 200,
               long_pause_ms: int = 500) -> dict:
    """`cutoff_ms` is the shortest silence counted as a pause: below ~200ms these are
    articulatory gaps between words, not hesitations, and they drag the fit down.

    `files_with_long_pause` counts recordings carrying a pause at least
    `long_pause_ms` long — the ones usable as a real caller for PIR without any
    synthesis, since the true end is then measured by the same VAD rather than built.
    """
    pauses: list[int] = []
    file_max: list[int] = []  # longest pause per recording -> call-level rates
    used = skipped = 0
    secs = 0.0
    for w in wavs:
        try:
            pcm, rate = read_audio(w)
        except Exception:
            skipped += 1
            continue
        secs += len(pcm) / rate
        found = pause_lengths_ms(pcm, rate, min_ms=cutoff_ms)
        if found:
            used += 1
            pauses += found
            file_max.append(max(found))
        else:
            skipped += 1
    if len(pauses) < 20:
        raise SystemExit(f"only {len(pauses)} pauses found — need a real corpus, not a sample")

    mu, sigma = fit_lognormal(pauses)
    fit = {"cutoff_ms": cutoff_ms, "n_pauses": len(pauses), "n_files": len(wavs),
           "files_used": used, "files_skipped": skipped,
           "long_pause_ms": long_pause_ms,
           "files_with_long_pause": sum(m >= long_pause_ms for m in file_max),
           "audio_hours": round(secs / 3600, 2), "mu": mu, "sigma": sigma,
           "ks": ks_statistic(pauses, mu, sigma),
           "median_ms": int(np.median(pauses)),
           "p25_ms": int(np.percentile(pauses, 25)),
           "p75_ms": int(np.percentile(pauses, 75)),
           "p90_ms": int(np.percentile(pauses, 90)),
           "p99_ms": int(np.percentile(pauses, 99)),
           "exceed": {str(g): round(float(np.mean(np.asarray(pauses) > g)), 4)
                      for g in GATES},
           # the number a deployment actually feels: share of *recordings* carrying at
           # least one pause long enough to end the turn early. A caller only has to
           # hesitate once for the call to go wrong.
           "calls_exceed": {str(g): round(float(np.mean(np.asarray(file_max) > g)), 4)
                            for g in GATES},
           "file_max_median_ms": int(np.median(file_max))}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():  # keep hand-added provenance (`source`) across a re-fit
            fit = {**json.loads(out.read_text()), **fit}
        out.write_text(json.dumps(fit, indent=2))
    return fit


def gate_advice(fit: dict, default_ms: int = DEFAULT_GATE_MS) -> list[dict]:
    """What to actually set the endpointing gate to, and what it costs.

    The cliff on its own only says "raise the gate", which anyone can read off the
    docs. The decision needs both columns: raising it cuts fewer callers off, and
    charges the delay to *every* turn in *every* call — the gate elapses before the
    agent may answer, so `gate - default` is added latency the caller feels each time
    they stop speaking. One column is a tuning curve; two is a recommendation.
    """
    per_pause = fit.get("exceed", {})
    per_call = fit.get("calls_exceed", {})
    return [{"gate_ms": g,
             "pauses_tripped": per_pause.get(str(g)),
             "calls_affected": per_call.get(str(g)),
             "added_latency_ms": g - default_ms}
            for g in GATES]


def recommended_gate(fit: dict, budget_ms: int = 400,
                     default_ms: int = DEFAULT_GATE_MS) -> dict:
    """The lowest gate whose call-level rate is the best available within a latency budget.

    `budget_ms` is how much extra turn-end delay the deployment will accept. Stated as
    an input rather than assumed, because it is a product decision and not a
    measurement — a booking bot can afford 400ms, a barge-in-heavy assistant cannot.
    """
    rows = [r for r in gate_advice(fit, default_ms)
            if r["added_latency_ms"] <= budget_ms and r["calls_affected"] is not None]
    if not rows:
        return {}
    best = min(rows, key=lambda r: (r["calls_affected"], r["added_latency_ms"]))
    base = next((r for r in gate_advice(fit, default_ms) if r["gate_ms"] == default_ms), {})
    return {**best, "budget_ms": budget_ms,
            "calls_affected_at_default": base.get("calls_affected"),
            "removes_share_of_affected_calls":
                None if not base.get("calls_affected") else
                round(1 - best["calls_affected"] / base["calls_affected"], 4)}

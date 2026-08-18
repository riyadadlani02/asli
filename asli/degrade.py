"""The dial box. Telephony codec, noise, packet loss — applied to int16 mono PCM.

Codec simulation goes through ffmpeg rather than a hand-rolled mu-law table: we want
a real G.711 round trip including the resampler's anti-aliasing, because that is what
strips the >3.4kHz detail that distinguishes Indian names and digit words.
"""

from __future__ import annotations

import subprocess

import numpy as np

from .spec import SAMPLE_RATE


def _ffmpeg(pcm: np.ndarray, args: list[str], in_rate: int) -> bytes:
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(in_rate), "-ac", "1", "-i", "-", *args, "-"]
    p = subprocess.run(cmd, input=pcm.astype("<i2").tobytes(), capture_output=True, check=True)
    return p.stdout


def telephony(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    """16k -> 8k G.711 mu-law -> back. Irreversible: this is the point."""
    narrow = _ffmpeg(pcm, ["-ar", "8000", "-f", "mulaw"], rate)
    wide = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "mulaw", "-ar", "8000", "-ac", "1",
         "-i", "-", "-ar", str(rate), "-f", "s16le", "-"],
        input=narrow, capture_output=True, check=True,
    ).stdout
    return np.frombuffer(wide, dtype="<i2").copy()


def pink_noise(n: int, seed: int = 0) -> np.ndarray:
    """1/f noise — a better stand-in for room/line noise than white."""
    rng = np.random.default_rng(seed)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.arange(len(spec))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-12)


def babble(n: int, speech: np.ndarray, seed: int = 0, voices: int = 6) -> np.ndarray:
    """Call-centre floor: overlapping copies of real speech at random offsets."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    src = speech.astype(np.float64)
    for _ in range(voices):
        off = rng.integers(0, max(1, len(src)))
        rolled = np.roll(src, off)
        tiled = np.resize(rolled, n)
        out += tiled
    return out / (np.abs(out).max() + 1e-12)


def add_noise(pcm: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix `noise` under `pcm` at exactly `snr_db`, measured over the whole signal."""
    sig = pcm.astype(np.float64)
    nz = np.resize(noise.astype(np.float64), len(sig))
    p_sig = np.mean(sig**2)
    p_nz = np.mean(nz**2) + 1e-12
    scale = np.sqrt(p_sig / (p_nz * 10 ** (snr_db / 10)))
    return np.clip(sig + nz * scale, -32768, 32767).astype(np.int16)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    sig = clean.astype(np.float64)
    nz = noisy.astype(np.float64) - sig
    return 10 * np.log10(np.mean(sig**2) / (np.mean(nz**2) + 1e-12))


def packet_loss(pcm: np.ndarray, rate: int = SAMPLE_RATE, rate_pct: float = 2.0,
                frame_ms: int = 20, seed: int = 0) -> np.ndarray:
    """Drop whole 20ms frames to silence, the way a lossy RTP leg does."""
    rng = np.random.default_rng(seed)
    n = int(rate * frame_ms / 1000)
    out = pcm.copy()
    for i in range(0, len(out) - n, n):
        if rng.random() < rate_pct / 100:
            out[i : i + n] = 0
    return out


def apply(pcm: np.ndarray, dials: dict, speech_for_babble: np.ndarray | None = None,
          rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a degradation dict. Order matters: noise is added on the wide band,
    then the codec mangles both together, as happens on a real line."""
    out = pcm
    snr = dials.get("snr_db")
    if snr is not None:
        profile = dials.get("noise", "pink")
        nz = (babble(len(out), speech_for_babble, voices=6)
              if profile == "babble" and speech_for_babble is not None
              else pink_noise(len(out), seed=dials.get("seed", 0)))
        out = add_noise(out, nz, snr)
    if dials.get("packet_loss_pct"):
        out = packet_loss(out, rate, dials["packet_loss_pct"], seed=dials.get("seed", 0))
    if dials.get("telephony"):
        out = telephony(out, rate)
    return out

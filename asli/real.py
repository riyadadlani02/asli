"""PIR on real callers, with no synthesis anywhere in the path.

Every PIR number from the TTS lane carries one fair objection: the voice is ours, so
both the acoustics and the *placement* of the hesitation inside the sentence are
authored. Removing that does not need new recordings, because PIR needs exactly one
thing from an utterance — the moment speech actually ended — and unlike INEPA and SFR
it needs no known entity. Spontaneous corpus speech can therefore drive it directly.

The trade is stated rather than hidden. In the TTS lane `true_end_ms` is known by
construction. Here it is *measured*, by the same energy VAD that located the pause, so
it carries an error. Two things bound that error instead of a promise:

  * a 20ms analysis frame, so quantisation is ±20ms;
  * `end_spread_ms` per recording — the true end re-derived at half and at double the
    chosen threshold. A file whose end moves a lot under that is one where the VAD is
    guessing, and it is reported rather than dropped.

The VAD takes the last frame above threshold as the end, so a trailing quiet syllable
shortens the utterance in our books, which makes an early turn-end look *less* early.
Like the latency in the socket timestamps, the error runs against the finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fit import ANALYSIS_FRAME_MS, adaptive_threshold, frame_rms, read_audio
from .spec import CallSpec, Segment

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def pause_floor_db(pcm: np.ndarray, rate: int, spec: CallSpec) -> float | None:
    """How much quieter the hesitation is than the speech around it, in dB.

    A synthesised pause is digital silence. A real one is a live phone line: room,
    handset and network noise carry on through it. An energy VAD cannot end a turn
    during a pause it can still hear, so this ratio — not the pause length — decides
    whether the endpointer ever gets the chance to fire.
    """
    gaps = spec.internal_pauses
    if not gaps:
        return None
    seg = lambda a, b: pcm[int(rate * a / 1000):int(rate * b / 1000)].astype(np.float64)
    quiet = np.concatenate([seg(a, b) for a, b in gaps])
    loud = np.concatenate([seg(a, b) for a, b in spec.seg_bounds_ms])
    if not len(quiet) or not len(loud):
        return None
    rq, rl = np.sqrt((quiet ** 2).mean()), np.sqrt((loud ** 2).mean())
    return None if rq <= 0 or rl <= 0 else round(float(20 * np.log10(rq / rl)), 1)


def quiet_pause(pcm: np.ndarray, rate: int, spec: CallSpec) -> np.ndarray:
    """The same real utterance with only the hesitation replaced by digital silence.

    The one-variable control for the null result on real audio: real voice, real line,
    real placement, and the only thing changed is whether the pause is audible. If the
    turn now ends early, the noise floor was what protected the caller — the
    endpointer was never being patient.
    """
    out = pcm.copy()
    for p0, p1 in spec.internal_pauses:
        out[int(rate * p0 / 1000):int(rate * p1 / 1000)] = 0
    return out


def _bounds(pcm: np.ndarray, rate: int, scale: float = 1.0,
            frame_ms: int = ANALYSIS_FRAME_MS) -> tuple[int, int, list[tuple[int, int]]] | None:
    """(speech_start_ms, speech_end_ms, internal silence runs) at `scale` × threshold."""
    frame = max(1, int(rate * frame_ms / 1000))
    rms = frame_rms(pcm, frame)
    th = adaptive_threshold(rms)
    if th is None:
        return None
    loud = list(rms >= th * scale)
    if not any(loud):
        return None
    first, last = loud.index(True), len(loud) - 1 - loud[::-1].index(True)

    runs, run = [], 0
    for i, v in enumerate(loud[first:last + 1], start=first):
        if v:
            if run:
                runs.append(((i - run) * frame_ms, i * frame_ms))
            run = 0
        else:
            run += 1
    return first * frame_ms, (last + 1) * frame_ms, runs


def spec_from_audio(pcm: np.ndarray, rate: int, uid: str,
                    min_pause_ms: int = 500) -> tuple[CallSpec, int] | None:
    """A real recording described as a CallSpec, so `score.pir` works on it unchanged.

    Returns None unless the recording carries a mid-utterance pause of at least
    `min_pause_ms` — without one there is no hesitation for an endpointer to trip on
    and the file says nothing about PIR.

    *Every* qualifying pause is recorded, not just the longest. A ten-second voice
    message hesitates several times, and an endpointer that fires in the second pause
    is doing the same thing as one that fires in the first — scoring only the longest
    would file that as an unattributable cut. It also makes `in_injected_pause` mean
    what it should on this lane: the turn ended inside a hesitation, rather than
    merely before the recording ran out.

    That distinction matters more here than on the authored lane. A spontaneous
    monologue *should* be split into several turns; `premature` alone is therefore
    inflated on this corpus and is reported only beside the in-pause figure.

    There is no `canonical` because spontaneous speech contains no entity we authored:
    this lane scores PIR only, which is why INEPA and SFR stay on the authored lane.
    """
    b = _bounds(pcm, rate)
    if b is None:
        return None
    start_ms, end_ms, runs = b
    long_runs = [r for r in runs if r[1] - r[0] >= min_pause_ms]
    if not long_runs:
        return None

    spread = 0
    for scale in (0.5, 2.0):
        alt = _bounds(pcm, rate, scale)
        if alt:
            spread = max(spread, abs(alt[1] - end_ms))

    # one segment per stretch of speech, one recorded pause between each pair
    bounds, segments, cursor = [], [], start_ms
    for p0, p1 in long_runs:
        bounds.append((cursor, p0))
        segments.append(Segment("<real speech>", pause_after_ms=p1 - p0))
        cursor = p1
    bounds.append((cursor, end_ms))
    segments.append(Segment("<real speech>"))

    spec = CallSpec(
        id=uid, entity_type="digits", canonical="", lang="hi-IN",
        segments=segments, seg_bounds_ms=bounds, true_end_ms=end_ms,
    )
    return spec, spread


def load_corpus(directory: str | Path, min_pause_ms: int = 500, limit: int = 20,
                scan_cap: int | None = None) -> list[tuple[CallSpec, np.ndarray, int, int]]:
    """(spec, pcm, rate, end_spread_ms) for the first `limit` usable recordings.

    Files are taken in sorted order and the ones without a long enough pause are
    skipped, not resampled or padded — the audio reaching the recogniser is the
    corpus file itself, at its own sample rate.
    """
    files = sorted(f for f in Path(directory).rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES)
    out = []
    for f in files[:scan_cap]:
        try:
            pcm, rate = read_audio(f)
        except Exception:
            continue
        got = spec_from_audio(pcm, rate, f.stem, min_pause_ms)
        if got:
            spec, spread = got
            out.append((spec, pcm, rate, spread))
            if len(out) >= limit:
                break
    return out

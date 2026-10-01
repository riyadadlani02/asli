"""Test bench: any recording, a live recogniser, and a verdict that says what it measured.

Every other lane in this repo starts from audio we authored. The bench starts from audio
someone brings — a phone recording, the microphone in a meeting, a file a call centre
sends over — so that a sceptic can test the claim with their own voice. That changes what
can be known, and the report says so rather than borrowing the authored lane's certainty:

  * the end of speech is MEASURED (energy VAD, 20 ms frames, with the spread under a
    halved and doubled threshold), never assumed;
  * the pauses are the ones the VAD finds in the audio, not ones we placed. Whether a
    pause was a hesitation or a finished sentence is something only the speaker knows —
    pause length alone cannot say (see DiarBench) — so the report states where the turn
    ended and what words it ended on, and leaves that call to the person who spoke;
  * the entity is checked only when the person says what they said. The bench never
    invents a ground truth.

Each result carries its provenance: which provider was called, with which settings, when,
and a hash of the exact audio streamed. A result from the bundled mock VAD is marked MOCK
everywhere it appears, because a mock that looks like a vendor is the one thing a test
bench must never show.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import policy, score
from .drive import FRAME_SAMPLES, DeepgramWS, GeminiLive, MockASR, OpenAIWS, SarvamWS
from .fit import ANALYSIS_FRAME_MS
from .real import _bounds, pause_floor_db
from .spec import SAMPLE_RATE, CallSpec, Event, Result, Segment

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "bench"  # git-ignored: these hold people's voices
SAMPLES = ROOT / "samples" / "manifest.yaml"

MIN_PAUSE_MS = 200      # the conventional cut between an articulatory gap and a pause,
                        # the same one the Gram Vaani fit uses
TOL_MS = score.ATTRIBUTION_TOLERANCE_MS
MAX_SECONDS = 90        # a cost guard: every second is streamed, in real time, per provider


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    env: str | None          # the key it needs; None = runs offline
    gate_param: str | None   # the vendor's name for the silence timer; None = not a timer
    note: str = ""
    short: str = ""          # for sentences: "Sarvam ended the turn…"

    @property
    def live(self) -> bool:
        return self.env is not None


PROVIDERS: dict[str, Provider] = {p.id: p for p in (
    Provider("sarvam", "Sarvam · saaras:v3-realtime", "SARVAM_API_KEY", "silence_duration_ms",
             "turn end = vad.speech_end off the socket", short="Sarvam"),
    Provider("deepgram", "Deepgram · nova-2", "DEEPGRAM_API_KEY", "endpointing",
             "turn end = speech_final; an empty final is credited with the last interim, "
             "which is what a live consumer holds at that instant", short="Deepgram"),
    Provider("openai", "OpenAI · gpt-4o-transcribe · server_vad", "OPENAI_API_KEY",
             "silence_duration_ms", "turn end = speech_stopped, on the provider's audio clock",
             short="OpenAI (server_vad)"),
    Provider("openai-semantic", "OpenAI · gpt-4o-transcribe · semantic_vad", "OPENAI_API_KEY",
             None, "not a silence timer: the provider decides from the audio and the words, "
             "so the gate setting does not apply", short="OpenAI (semantic_vad)"),
    Provider("gemini", "Gemini · 3.1-flash-live", "GEMINI_API_KEY", "silenceDurationMs",
             "no turn-end event exists: the turn end is when the model starts replying, so it "
             "includes model latency and reads LATE — a lower bound, first turn only",
             short="Gemini"),
    Provider("mock", "mock energy VAD — offline, NOT a vendor", None, "negative_frames_count",
             "no vendor was called and nothing was transcribed. The harness's own energy VAD, "
             "for checking the bench works without keys. Never evidence about any product.",
             short="The mock VAD"),
)}


def configured() -> dict[str, bool]:
    """Which providers can run right now. Keys are read server-side and never echoed."""
    return {pid: (p.env is None or bool(os.environ.get(p.env))) for pid, p in PROVIDERS.items()}


# --- audio ----------------------------------------------------------------------

def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    """Resample to 16 kHz mono int16. Box-averaged before decimation as a cheap anti-alias,
    the same approach the browser sandbox uses."""
    if rate == SAMPLE_RATE:
        return pcm.astype(np.int16)
    x = pcm.astype(np.float64)
    if rate > SAMPLE_RATE:
        k = int(round(rate / SAMPLE_RATE))
        if k > 1:
            x = np.convolve(x, np.ones(k) / k, mode="same")
    n = int(len(x) * SAMPLE_RATE / rate)
    idx = np.arange(n) * rate / SAMPLE_RATE
    out = np.interp(idx, np.arange(len(x)), x)
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """A WAV upload -> (mono int16, rate). 16-bit PCM only: the bench page always sends
    that, and guessing at other encodings is how timestamps go quietly wrong."""
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got {8 * w.getsampwidth()}-bit")
        ch, rate = w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return pcm.copy(), rate


def wav_bytes(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def load_audio(path: str | Path) -> np.ndarray:
    """Any file ffmpeg reads -> 16 kHz mono int16. WAVs need no ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        pcm, rate = read_wav_bytes(path.read_bytes())
        return to_16k(pcm, rate)
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").copy()


# --- what is in the audio -------------------------------------------------------

@dataclass
class Analysis:
    """What the VAD can say about a recording before any provider hears it."""

    duration_ms: int
    usable: bool                       # could speech be separated from the floor at all?
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None   # the measured end of speech: what a cut is judged against
    end_spread_ms: int | None = None   # how far that end moves at half / double the threshold
    pauses: list[tuple[int, int]] = field(default_factory=list)
    longest_pause_ms: int = 0
    pause_floor_db: float | None = None
    threshold: float | None = None     # RMS, full scale = 1.0
    min_pause_ms: int = MIN_PAUSE_MS
    frame_ms: int = ANALYSIS_FRAME_MS
    basis: str = "measured"            # "authored" when the pause was spliced in by us


def authored(an: Analysis, sample: dict) -> Analysis:
    """A kit sample's own truth in place of the VAD's estimate of it.

    The pause was spliced in, so its window and the end of speech are known exactly —
    the same footing as the authored lane. On the noisy renders this matters: babble
    fills part of the pause and the VAD reports a different window than the one built.
    """
    if not (sample.get("pause_ms") and sample.get("speech_end_ms")) or not an.usable:
        return an
    a, z = (int(x) for x in sample["pause_ms"])
    from dataclasses import replace

    return replace(an, pauses=[(a, z)], longest_pause_ms=z - a,
                   speech_end_ms=int(sample["speech_end_ms"]), end_spread_ms=0,
                   basis="authored")


def analyse(pcm: np.ndarray, rate: int = SAMPLE_RATE, min_pause_ms: int = MIN_PAUSE_MS) -> Analysis:
    dur = int(len(pcm) * 1000 / rate)
    b = _bounds(pcm, rate)
    if b is None:
        return Analysis(duration_ms=dur, usable=False, min_pause_ms=min_pause_ms)
    start, end, runs = b
    pauses = [(a, z) for a, z in runs if z - a >= min_pause_ms]
    spread = 0
    for scale in (0.5, 2.0):
        alt = _bounds(pcm, rate, scale)
        if alt:
            spread = max(spread, abs(alt[1] - end))
    from .fit import adaptive_threshold, frame_rms

    th = adaptive_threshold(frame_rms(pcm, max(1, int(rate * ANALYSIS_FRAME_MS / 1000))))
    an = Analysis(duration_ms=dur, usable=True, speech_start_ms=start, speech_end_ms=end,
                  end_spread_ms=spread, pauses=pauses,
                  longest_pause_ms=max((z - a for a, z in pauses), default=0),
                  threshold=th, min_pause_ms=min_pause_ms)
    an.pause_floor_db = pause_floor_db(pcm, rate, spec_for("floor", an))
    return an


def spec_for(name: str, an: Analysis, canonical: str = "", entity_type: str = "digits") -> CallSpec:
    """The recording as a CallSpec — one segment per stretch of speech, a measured pause
    between each — so `score.pir` reads it exactly as it reads an authored call."""
    if not an.usable:
        return CallSpec(id=name, segments=[Segment("")], entity_type=entity_type,
                        canonical=canonical)
    bounds, segments, cursor = [], [], an.speech_start_ms
    for a, z in an.pauses:
        bounds.append((cursor, a))
        segments.append(Segment("", pause_after_ms=z - a))
        cursor = z
    bounds.append((cursor, an.speech_end_ms))
    segments.append(Segment(""))
    return CallSpec(id=name, segments=segments, entity_type=entity_type, canonical=canonical,
                    seg_bounds_ms=bounds, true_end_ms=an.speech_end_ms)


# --- the sample kit ----------------------------------------------------------------

SAMPLE_KEYS = ("id", "file", "title", "said", "entity_type", "expected")
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".aac")


def load_samples(manifest: Path = SAMPLES) -> list[dict]:
    """The bundled kit, then your own recordings from `mine.yaml` beside it.

    Entries whose file is missing or outside the repo are dropped rather than served: the
    server hands these files to a browser, so a manifest is not trusted to name them.
    """
    import yaml

    out, seen, root = [], set(), ROOT.resolve()
    for path in (manifest, manifest.with_name("mine.yaml")):
        if not path.exists():
            continue
        for row in yaml.safe_load(path.read_text()) or []:
            if not isinstance(row, dict) or not all(row.get(k) for k in SAMPLE_KEYS):
                continue
            row = {**row, "id": str(row["id"]), "expected": str(row["expected"]),
                   "kit": path.stem != "mine"}
            f = (ROOT / row["file"]).resolve()
            if (row["id"] in seen or root not in f.parents or not f.is_file()
                    or f.suffix.lower() not in AUDIO_SUFFIXES):
                continue
            seen.add(row["id"])
            out.append(row)
    return out


# --- ground truth, only when the speaker gives it --------------------------------

ENTITY_TYPES = ("digits", "amount", "date")


def normalise_expected(entity_type: str, expected: str) -> str:
    """'98771 11', '9,87,7111', 'nine eight double seven triple one' -> '9877111'."""
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"entity type must be one of {', '.join(ENTITY_TYPES)}")
    value = score.normalise(entity_type, expected)
    if entity_type == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or ""):
        value = ""
    if not value:
        raise ValueError(f"could not read {expected!r} as a {entity_type} — "
                         "try the figure itself, e.g. 9877111, 250000 or 15/09/2026")
    return value


def entity_found(entity_type: str, canonical: str, text: str) -> bool | None:
    """Did this text carry the value? None = the scorer cannot tell, and says so."""
    if not text.strip():
        return False
    if entity_type == "date":
        got = score.spoken_date(score._norm(text).translate(score.DEVANAGARI_DIGITS))
        return None if not got else got == canonical
    if entity_type == "amount":
        # A figure the recogniser already normalised ("2,50,000") sitting inside words: the
        # tail parse in score.mangled_entity splits it at the commas and misses it.
        for m in re.finditer(r"\d{1,3}(?:,\d{2,3})+|\d+", text.translate(score.DEVANAGARI_DIGITS)):
            if m.group(0).replace(",", "").lstrip("0") == canonical.lstrip("0"):
                return True
    probe = CallSpec(id="probe", segments=[Segment(text)], entity_type=entity_type,
                     canonical=canonical)
    damaged = score.mangled_entity(probe, text)
    return None if damaged is None else not damaged


# --- running providers ------------------------------------------------------------

def make_adapter(pid: str, gate: int, mode: str, threshold: float | None):
    if pid == "sarvam":
        return SarvamWS(silence_duration_ms=gate, mode=mode)
    if pid == "deepgram":
        return DeepgramWS(silence_duration_ms=gate)
    if pid in ("openai", "openai-semantic"):
        # endpoint timestamps on the provider's own audio clock, and the whole session
        # rather than stopping at the first final: the second turn is where the number is
        return OpenAIWS(silence_duration_ms=gate, require_endpoint_timestamps=True,
                        read_grace_s=4,
                        turn_detection="semantic_vad" if pid == "openai-semantic" else "server_vad")
    if pid == "gemini":
        return GeminiLive(silence_duration_ms=gate)
    if pid == "mock":
        frames = max(1, round(gate * SAMPLE_RATE / 1000 / FRAME_SAMPLES))
        return MockASR(negative_frames_count=frames, threshold=threshold or 0.02)
    raise ValueError(f"unknown provider {pid!r}; choose from {', '.join(PROVIDERS)}")


def settings_for(pid: str, gate: int, mode: str) -> dict:
    p = PROVIDERS[pid]
    out: dict = {}
    if p.gate_param == "negative_frames_count":
        frames = max(1, round(gate * SAMPLE_RATE / 1000 / FRAME_SAMPLES))
        out[p.gate_param] = f"{frames} frames ≈ {round(frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE)} ms"
    elif p.gate_param:
        out[p.gate_param] = gate
    if pid == "sarvam":
        out["mode"] = mode
    if pid.startswith("openai"):
        out["turn_detection"] = "semantic_vad" if pid == "openai-semantic" else "server_vad"
    return out


async def _run_many(pcm: np.ndarray, spec: CallSpec, pids: list[str], gate: int, mode: str,
                    threshold: float | None, emit: Callable[[dict], None]) -> dict[str, dict]:
    """Every provider hears the same audio at the same time, each paced in real time."""
    async def one(pid: str) -> dict:
        p = PROVIDERS[pid]
        partials: list[tuple[int, str]] = []
        started = _now()
        if p.env and not os.environ.get(p.env):
            res = Result(spec_id=spec.id, adapter=pid,
                         error=f"{p.env} is not set in .env — nothing was sent to {p.label}")
        else:
            ad = make_adapter(pid, gate, mode, threshold)
            ad.on_event = lambda e: emit({"type": "event", "provider": pid, "kind": e.kind,
                                          "t_ms": e.t_ms, "text": e.text})

            def on_partial(t_ms: int, text: str) -> None:
                partials.append((t_ms, text))
                emit({"type": "partial", "provider": pid, "t_ms": t_ms, "text": text})

            ad.on_partial = on_partial
            emit({"type": "provider_start", "provider": pid, "label": p.label})
            if pid == "mock":
                res = ad.run(pcm, spec)
                # the mock attributes text from the spec; a real recording has none, and a
                # transcript the mock made up would be the worst thing on the page
                res.events = [e for e in res.events if e.kind != "transcript"]
                res.transcript = ""
                for e in res.events:
                    emit({"type": "event", "provider": pid, "kind": e.kind, "t_ms": e.t_ms,
                          "text": ""})
            else:
                res = await ad.run(pcm, spec)
        return {"result": res, "partials": partials, "started_at": started,
                "finished_at": _now()}

    outs = await asyncio.gather(*(one(pid) for pid in pids))
    return dict(zip(pids, outs))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- the verdict --------------------------------------------------------------------

def turns_of(res: Result, pid: str) -> list[dict]:
    """The turns the agent would have seen: what text it held at each turn end."""
    out: list[dict] = []
    if pid == "gemini":
        # transcript events are fragments; the inferred turn end splits them
        buf: list[str] = []
        for e in res.events:
            if e.kind == "transcript":
                buf.append(e.text)
            elif e.kind == "speech_end":
                out.append({"end_ms": e.t_ms, "text": "".join(buf).strip()})
                buf = []
        if buf:
            out.append({"end_ms": None, "text": "".join(buf).strip()})
        return out
    ends = [e.t_ms for e in res.events if e.kind == "speech_end"]
    finals = [e for e in res.events if e.kind == "transcript"]
    if not finals:
        return [{"end_ms": t, "text": ""} for t in ends]
    for i, f in enumerate(finals):
        out.append({"end_ms": ends[i] if i < len(ends) else f.t_ms, "text": f.text})
    return out


def _devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def marker_test(text: str) -> dict:
    """The words a turn ended on: the evidence a turn was unfinished that does not depend
    on how long the pause was. Abstains on romanised text, because the lists are
    Devanagari and a romanised turn would otherwise always pass as 'complete'."""
    d = policy.decide(text)
    last = policy.last_token(text)
    if d.reason == "complete" and text.strip() and not _devanagari(text):
        return {"last_word": last, "hold": False, "reason": "not-checked",
                "note": "romanised transcript — the marker lists are Devanagari, so no claim"}
    notes = {
        "dangling": "a word that does not end a Hindi sentence — she was not finished",
        "filler-en": "an English/Hinglish filler — mid-thought",
        "finality-marker": "a word that marks a finished turn",
        "complete": "a word that can legally end a sentence — the cut looks defensible "
                    "from the text alone (Hindi is verb-final, so a real cut often does)",
        "empty": "the agent held no text at all at this turn end",
    }
    return {"last_word": last, "hold": d.hold, "reason": d.reason, "note": notes[d.reason]}


def verdict(spec: CallSpec, an: Analysis, res: Result, pid: str, gate: int,
            partials: list[tuple[int, str]], hold_ms: int = 600) -> dict:
    p = PROVIDERS[pid]
    out: dict = {"provider": pid, "label": p.label}
    if res.error:
        out.update(status="error", headline=f"The call failed: {res.error}")
        return out
    turns = turns_of(res, pid)
    ends = [e.t_ms for e in res.events if e.kind == "speech_end"]
    out["turns"] = turns
    out["n_turns"] = len(turns)

    cuts = []
    if an.usable:
        for t in ends:
            if t < an.speech_end_ms:
                pause = next(((a, z) for a, z in an.pauses if a <= t <= z + TOL_MS), None)
                cuts.append({"t_ms": t, "ms_early": an.speech_end_ms - t,
                             "in_pause": list(pause) if pause else None})
    out["cuts"] = cuts
    first_cut = cuts[0] if cuts else None
    in_pause = next((c for c in cuts if c["in_pause"]), None)

    first_text = turns[0]["text"] if turns else ""
    full_text = " ".join(t["text"] for t in turns if t["text"]).strip()
    out["first_turn_text"] = first_text
    out["full_text"] = full_text
    if partials:
        out["longest_partial"] = max((t for _, t in partials), key=len)

    # --- headline: the one sentence a non-specialist reads ---------------------------
    if not an.usable:
        out["status"] = "unscorable"
        out["headline"] = ("Could not separate speech from the background in this recording, "
                           "so there is no measured end of speech to judge a turn end against. "
                           "The turns below are still exactly what the provider returned.")
    elif in_pause:
        a, z = in_pause["in_pause"]
        out["status"] = "cut"
        out["headline"] = (f"Cut off. {p.short} ended the turn at {in_pause['t_ms'] / 1000:.1f}s, "
                           f"inside a {z - a} ms pause — {in_pause['ms_early']} ms before you "
                           f"finished speaking.")
    elif first_cut:
        out["status"] = "early"
        out["headline"] = (f"The turn ended {first_cut['ms_early']} ms before the measured end of "
                           f"speech, but not inside a detected pause — a short gap or a quiet "
                           f"last syllable. Weaker evidence; read the turns.")
    elif not ends and not turns:
        out["status"] = "no-turns"
        out["headline"] = "No turn end and no transcript came back — too quiet, too short, or rejected."
    else:
        out["status"] = "ok"
        out["headline"] = (f"Not cut off. {p.short} waited until you had finished "
                           f"({len(turns)} turn{'s' if len(turns) != 1 else ''}).")

    if an.usable and p.gate_param and pid != "mock":
        out["context"] = (f"Your longest pause was {an.longest_pause_ms} ms; the silence timer was "
                          f"{gate} ms. A timer shorter than a pause can end the turn inside it.")
    elif pid == "openai-semantic":
        out["context"] = "No silence timer on this lane — the provider chose when the turn ended."

    # --- the words at the cut, and what the marker policy would have done ------------
    if (in_pause or first_cut) and pid != "mock":   # the mock holds no words to test
        cut = in_pause or first_cut
        idx = next((i for i, t in enumerate(turns) if t["end_ms"] == cut["t_ms"]), 0)
        held = turns[idx]["text"] if turns else ""
        mt = marker_test(held)
        out["at_cut"] = {"text": held, **mt}
        if mt["hold"] and cut["in_pause"]:
            resume = cut["in_pause"][1]
            wait = max(0, resume - cut["t_ms"])
            rescued = wait <= hold_ms
            merged = " ".join(t["text"] for t in turns[idx:idx + 2] if t["text"]).strip()
            out["policy"] = {
                "fires": True, "hold_ms": hold_ms, "wait_needed_ms": wait, "rescued": rescued,
                "merged_text": merged if rescued else held,
                "note": (f"the rule holds turns that end on '{mt['last_word']}'; you resumed "
                         f"{wait} ms later, " + ("inside" if rescued else "outside") +
                         f" its {hold_ms} ms hold, so the agent " +
                         ("would have waited for the rest." if rescued else
                          "would still have answered early."))}
        else:
            out["policy"] = {"fires": False, "note": (
                "the rule did not fire: " + mt["note"] if mt["reason"] != "not-checked"
                else "the rule cannot judge a romanised transcript")}

    # --- the entity, only if the speaker told us what it was --------------------------
    if spec.canonical and pid == "mock":
        out["entity"] = {"expected": spec.canonical, "type": spec.entity_type,
                         "not_applicable": "the mock VAD has no recogniser, so it never "
                                           "has the words — run a live provider for this"}
    elif spec.canonical:
        first = entity_found(spec.entity_type, spec.canonical, first_text)
        whole = entity_found(spec.entity_type, spec.canonical, full_text)
        if first is None and whole:
            # the scorer read the value later in this very session, so it can read this
            # recogniser's output; a first turn with no number in it at all lacks it
            first = False
        stream = (any(entity_found(spec.entity_type, spec.canonical, t) for _, t in partials)
                  if partials else None)
        out["entity"] = {"expected": spec.canonical, "type": spec.entity_type,
                         "in_first_turn": first, "in_session": whole, "in_partials": stream}
    return out


# --- one bench run, end to end ----------------------------------------------------------

CAVEATS = [
    "Turn-end times are audio sent so far when the event arrived, so they include network "
    "and server latency: a cut is reported later than it happened, never earlier.",
    "The end of speech is measured by an energy VAD (20 ms frames), not by a person; its "
    "spread under a halved and doubled threshold is reported beside it.",
    "Whether a pause was a hesitation or a finished sentence is the speaker's call — pause "
    "length alone cannot tell them apart. One sentence with a pause before a number is the "
    "clean test.",
    "One call is an example, not a rate. The rates are in the README, with their n.",
]


def run(pcm: np.ndarray, providers: list[str], *, gate: int = 500, mode: str = "verbatim",
        expected: str = "", entity_type: str = "digits", name: str = "recording",
        source: str = "file", hold_ms: int = 600, min_pause_ms: int = MIN_PAUSE_MS,
        phone_line: bool = False, emit: Callable[[dict], None] | None = None,
        save: bool = True, runs_dir: Path | None = None, sample: dict | None = None) -> dict:
    """Analyse, stream to every provider at once, score, and keep a receipt.

    `pcm` is 16 kHz mono int16. Silence is appended after the speech, as a phone line
    stays open after the caller stops; without it the final turn is never closed by a
    silence timer and the session ends on the socket closing instead.
    """
    emit = emit or (lambda _m: None)
    unknown = [p for p in providers if p not in PROVIDERS]
    if unknown or not providers:
        raise ValueError(f"unknown provider(s) {unknown}; choose from {', '.join(PROVIDERS)}")
    if len(pcm) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError(f"recording is {len(pcm) / SAMPLE_RATE:.0f}s; the bench streams at most "
                         f"{MAX_SECONDS}s per run (every second is billed, per provider)")
    if sample and not expected.strip():   # a kit sample carries its own truth
        expected, entity_type = str(sample["expected"]), sample["entity_type"]
    canonical = normalise_expected(entity_type, expected) if expected.strip() else ""

    if phone_line:
        from . import degrade

        pcm = degrade.apply(pcm, {"telephony": True})
    measured = analyse(pcm, SAMPLE_RATE, min_pause_ms)
    an = authored(measured, sample) if sample and not phone_line else measured
    spec = spec_for(name, an, canonical, entity_type)
    tail_ms = max(1500, gate + 1000)
    streamed = np.concatenate([pcm, np.zeros(SAMPLE_RATE * tail_ms // 1000, np.int16)])
    sha = hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest()

    emit({"type": "analysis", "analysis": asdict(an), "audio_ms": an.duration_ms,
          "streamed_ms": int(len(streamed) * 1000 / SAMPLE_RATE)})
    outs = asyncio.run(_run_many(streamed, spec, providers, gate, mode, an.threshold, emit))

    results = {}
    for pid, o in outs.items():
        res, p = o["result"], PROVIDERS[pid]
        v = verdict(spec, an, res, pid, gate, o["partials"], hold_ms)
        results[pid] = {
            "verdict": v,
            "provenance": {
                "kind": "live" if p.live else "mock",
                "provider": p.label, "note": p.note,
                "settings": settings_for(pid, gate, mode),
                "started_at": o["started_at"], "finished_at": o["finished_at"],
                "audio_sha256": sha, "trailing_silence_ms": tail_ms,
                "pacing": "100 ms chunks, real time",
                "key": f"{p.env} from .env, server-side" if p.env else "none — offline",
            },
            "events": [asdict(e) for e in res.events],
            "partials": o["partials"],
            "error": res.error,
        }
        emit({"type": "provider_done", "provider": pid, "result": results[pid]})

    report = {
        "id": "", "name": name, "source": source, "created_at": _now(),
        "audio": {"ms": an.duration_ms, "sha256": sha, "rate": SAMPLE_RATE,
                  "phone_line": "SIMULATED 8 kHz mu-law round trip" if phone_line else "as recorded"},
        "analysis": asdict(an),
        "analysis_measured": asdict(measured) if an is not measured else None,
        "ground_truth": ({"entity_type": entity_type, "as_typed": expected, "value": canonical}
                         if canonical else None),
        "settings": {"gate_ms": gate, "mode": mode, "hold_ms": hold_ms,
                     "min_pause_ms": min_pause_ms, "providers": providers},
        # a kit sample's pause and end are known by construction; shown beside the VAD's
        # measurement so the reader can see how far the measurement is from the truth
        "sample": ({k: sample.get(k) for k in ("id", "title", "said", "hesitation_ms",
                                               "pause_ms", "speech_end_ms", "voice", "line")}
                   if sample else None),
        "results": results,
        "caveats": CAVEATS,
    }
    if save:
        report["id"] = save_run(report, pcm, runs_dir or RUNS)
    emit({"type": "done", "report": report})
    return report


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "recording"


def save_run(report: dict, pcm: np.ndarray, runs_dir: Path = RUNS) -> str:
    """A receipt per run: the exact audio streamed, and every event that came back."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = f"{stamp}-{_slug(report['name'])}"
    d = runs_dir / rid
    n = 1
    while d.exists():
        n += 1
        d = runs_dir / f"{rid}-{n}"
    rid = d.name
    d.mkdir(parents=True)
    (d / "audio.wav").write_bytes(wav_bytes(pcm))
    report["id"] = rid
    (d / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return rid


def history(runs_dir: Path = RUNS, limit: int = 50) -> list[dict]:
    rows = []
    for f in sorted(runs_dir.glob("*/report.json"), reverse=True)[:limit]:
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"id": r.get("id") or f.parent.name, "name": r.get("name"),
                     "created_at": r.get("created_at"), "source": r.get("source"),
                     "statuses": {pid: x["verdict"].get("status")
                                  for pid, x in r.get("results", {}).items()},
                     "kinds": sorted({x["provenance"]["kind"]
                                      for x in r.get("results", {}).values()})})
    return rows


# --- terminal rendering, for `asli test` -------------------------------------------------

def render_text(report: dict) -> str:
    an = report["analysis"]
    lines = [f"\n{report['name']}  ({report['audio']['ms'] / 1000:.1f}s, "
             f"{report['audio']['phone_line']})"]
    if an["usable"] and an.get("basis") == "authored":
        a, z = an["pauses"][0]
        lines.append(f"  truth (spliced, exact): pause {a}–{z} ms ({z - a} ms) · speech ends "
                     f"{an['speech_end_ms']} ms")
    elif an["usable"]:
        pauses = ", ".join(f"{a / 1000:.2f}–{z / 1000:.2f}s" for a, z in an["pauses"]) or "none"
        lines.append(f"  measured: speech {an['speech_start_ms']}–{an['speech_end_ms']} ms "
                     f"(end spread {an['end_spread_ms']} ms) · pauses ≥{an['min_pause_ms']} ms: "
                     f"{pauses}")
    else:
        lines.append("  measured: speech not separable from the background")
    gt = report.get("ground_truth")
    if gt:
        lines.append(f"  you said: {gt['value']} ({gt['entity_type']})")
    for pid, r in report["results"].items():
        v, pv = r["verdict"], r["provenance"]
        tag = "LIVE" if pv["kind"] == "live" else "MOCK — not a vendor"
        lines.append(f"\n  [{tag}] {pv['provider']}  {pv['settings']}")
        lines.append(f"    {v['headline']}")
        if v.get("context"):
            lines.append(f"    {v['context']}")
        for i, t in enumerate(v.get("turns", []), 1):
            at = f"{t['end_ms']}ms" if t["end_ms"] is not None else "end"
            lines.append(f"    turn {i} @ {at}: {t['text'] or '(no text)'}")
        if "at_cut" in v:
            lines.append(f"    last word at the cut: {v['at_cut']['last_word'] or '—'} — "
                         f"{v['at_cut']['note']}")
        if "policy" in v:
            lines.append(f"    marker policy: {v['policy']['note']}")
        if "entity" in v and v["entity"].get("not_applicable"):
            lines.append(f"    {v['entity']['expected']}: not checked — "
                         f"{v['entity']['not_applicable']}")
        elif "entity" in v:
            e = v["entity"]
            yn = lambda x: "can't tell" if x is None else ("yes" if x else "NO")
            lines.append(f"    {e['expected']} in the first turn: {yn(e['in_first_turn'])} · "
                         f"in the whole session: {yn(e['in_session'])}"
                         + (f" · in the live partials: {yn(e['in_partials'])}"
                            if e["in_partials"] is not None else ""))
    if report.get("id"):
        lines.append(f"\n  receipt: results/bench/{report['id']}/")
    return "\n".join(lines)

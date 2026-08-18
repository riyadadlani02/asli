"""The three metrics. Deterministic wherever the truth is something we authored.

INEPA and PIR use no LLM judge at all — the first is a normalised comparison against a
value we wrote, the second is a comparison of two timestamps. Only the confirmation
call in SFR is a judgement, and that one is classified upstream in agent.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .spec import CallSpec, Result

WORD_DIGITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    "shunya": 0, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
    "panch": 5, "cheh": 6, "chhe": 6, "saat": 7, "aath": 8, "nau": 9,
}
MULTIPLIERS = {"double": 2, "triple": 3, "thrice": 3}


def spoken_digits(text: str) -> str:
    """'nine eight double seven triple one' -> '9877111'.

    The construction that breaks parsers: a multiplier word binds to the digit that
    follows it, which no English number normaliser expects.
    """
    out, pending = [], 1
    for tok in re.findall(r"[a-z]+|\d", text.lower()):
        if tok in MULTIPLIERS:
            pending = MULTIPLIERS[tok]
        elif tok.isdigit():
            out.append(tok * pending)
            pending = 1
        elif tok in WORD_DIGITS:
            out.append(str(WORD_DIGITS[tok]) * pending)
            pending = 1
    return "".join(out)


def normalise(entity_type: str, value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    if entity_type == "digits":
        d = re.sub(r"\D", "", v)
        return d or spoken_digits(v)
    if entity_type == "amount":
        d = re.sub(r"[^\d]", "", v.replace(",", ""))
        return str(int(d)) if d else ""
    if entity_type == "date":
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", v)
        return m.group(0) if m else v
    return re.sub(r"\s+", " ", v)


# --- INEPA ------------------------------------------------------------------

def inepa(spec: CallSpec, result: Result) -> bool:
    """Did the agent land on the value we injected? Correct = True."""
    return normalise(spec.entity_type, result.agent_entity or "") == normalise(
        spec.entity_type, spec.canonical
    )


# --- PIR --------------------------------------------------------------------

@dataclass
class Interruption:
    premature: bool
    t_ms: int | None = None
    ms_early: int | None = None
    after_filler: str = ""
    in_injected_pause: bool = False


def pir(spec: CallSpec, result: Result) -> Interruption:
    """A turn-end decision before the caller actually finished.

    We know the true end because we spliced the audio. `premature` counts any early
    turn-end; `in_injected_pause` additionally says it landed in the hesitation we
    built, which is the attributable, controlled version of the same event. Report
    both — a cut triggered by a natural intra-phrase gap is still a real failure,
    but only the second one is evidence about the filler under test.
    """
    for e in result.events:
        if e.kind == "speech_end" and e.t_ms < spec.true_end_ms:
            filler, injected = "", False
            for (start, end), seg in zip(spec.seg_bounds_ms, spec.segments):
                if seg.pause_after_ms and end <= e.t_ms <= end + seg.pause_after_ms:
                    filler, injected = seg.text, True
            return Interruption(True, e.t_ms, spec.true_end_ms - e.t_ms, filler, injected)
    return Interruption(False)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def mangled_entity(spec: CallSpec, transcript: str) -> bool | None:
    """Did the recogniser damage the *entity*, as opposed to any word in the turn?

    We authored the script, so the entity-bearing segment is known exactly: it is the
    last one. Every word of it must survive into the transcript. Comparing whole
    normalised sentences instead would count a clean transcript as an error, because
    normalise() is an entity extractor and a sentence is not an entity.

    Returns None when the two are not comparable — a Devanagari transcript against a
    romanised script says nothing about whether the entity survived, and scoring it
    as damage would manufacture a silent-failure rate out of an orthography
    difference. Abstain instead; SFR_bb still covers these calls.
    """
    want, got = set(_words(spec.segments[-1].text)), set(_words(transcript))
    if not want & got and _devanagari(transcript) and not _devanagari(spec.text):
        return None
    return not want <= got


def _devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097f" for c in text)


# --- SFR --------------------------------------------------------------------

def sfr_pair(spec: CallSpec, result: Result) -> tuple[bool | None, bool | None]:
    """(SFR_asr contribution, SFR_bb contribution).

    Each is None when the call is not in that metric's denominator — i.e. no error
    occurred, so there was nothing for the agent to catch. True = silent failure:
    an error was present and the agent acted anyway.
    """
    truth = normalise(spec.entity_type, spec.canonical)
    acted_blind = result.confirmed is False

    damaged = mangled_entity(spec, result.transcript) if result.transcript else False
    asr = acted_blind if damaged else None  # None damage -> abstain, same as no error

    final_wrong = normalise(spec.entity_type, result.agent_entity or "") != truth
    bb = acted_blind if final_wrong else None
    return asr, bb


def aggregate(rows: list[tuple[CallSpec, Result]]) -> dict:
    """Roll a run up into the numbers that go in the report."""
    inepa_scores = [inepa(s, r) for s, r in rows if r.agent_entity is not None]
    pirs = [pir(s, r) for s, r in rows]
    asr, bb = zip(*(sfr_pair(s, r) for s, r in rows)) if rows else ((), ())
    asr = [x for x in asr if x is not None]
    bb = [x for x in bb if x is not None]

    def rate(xs: list[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    early = [p.ms_early for p in pirs if p.premature]
    judged = [r.confirmed for _, r in rows if r.confirmed is not None]
    return {
        "n": len(rows),
        # SFR is uninterpretable without this: an agent that confirms everything
        # scores 0.0 trivially, and the metric has no room to move.
        "confirm_rate": rate(judged),
        "inepa_accuracy": rate(inepa_scores),
        "inepa_n": len(inepa_scores),
        "pir": rate([p.premature for p in pirs]),
        "pir_injected": rate([p.in_injected_pause for p in pirs]),
        "pir_n": len(pirs),
        "median_ms_early": sorted(early)[len(early) // 2] if early else None,
        "sfr_asr": rate(asr),
        "sfr_asr_n": len(asr),
        "sfr_bb": rate(bb),
        "sfr_bb_n": len(bb),
    }

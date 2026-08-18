"""The three metrics. Deterministic wherever the truth is something we authored.

INEPA and PIR use no LLM judge at all — the first is a normalised comparison against a
value we wrote, the second is a comparison of two timestamps. Only the confirmation
call in SFR is a judgement, and that one is classified upstream in agent.py.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .spec import CallSpec, Result

WORD_DIGITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    "shunya": 0, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
    "panch": 5, "cheh": 6, "chhe": 6, "saat": 7, "sat": 7, "aath": 8, "nau": 9,
    # Devanagari. A recogniser set to Hindi returns these, and comparing them to a
    # romanised script by string overlap would score every entity as damaged. The
    # entity survived if its *value* survived, so extraction has to read both scripts.
    "शून्य": 0, "ज़ीरो": 0, "जीरो": 0, "एक": 1, "वन": 1, "दो": 2, "टू": 2,
    "तीन": 3, "थ्री": 3, "चार": 4, "फोर": 4, "पाँच": 5, "पांच": 5, "फाइव": 5,
    "छह": 6, "छे": 6, "सिक्स": 6, "सात": 7, "सेवन": 7, "आठ": 8, "एट": 8,
    "नौ": 9, "नाइन": 9,
}
MULTIPLIERS = {"double": 2, "triple": 3, "thrice": 3, "डबल": 2, "ट्रिपल": 3}
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# danda and double danda sit inside the Devanagari block, so a naive \u0900-\u097f
# token match swallows them and "फाइव।" stops matching "फाइव".
DEVANAGARI_PUNCT = "\u0964\u0965"


def _norm(text: str) -> str:
    """NFC, minus Devanagari punctuation.

    Nukta letters have two encodings — precomposed (ज़ U+095B) and decomposed
    (ज + U+093C) — and a recogniser may return either. Without normalising, a table
    written one way silently fails to match text written the other, which reads as a
    recognition error that never happened.
    """
    out = unicodedata.normalize("NFC", text.lower())
    return "".join(" " if c in DEVANAGARI_PUNCT else c for c in out)


WORD_DIGITS = {unicodedata.normalize("NFC", k): v for k, v in WORD_DIGITS.items()}
MULTIPLIERS = {unicodedata.normalize("NFC", k): v for k, v in MULTIPLIERS.items()}


def spoken_digits(text: str) -> str:
    """'nine eight double seven triple one' -> '9877111'.

    The construction that breaks parsers: a multiplier word binds to the digit that
    follows it, which no English number normaliser expects.
    """
    out, pending = [], 1
    for tok in re.findall(r"[a-z]+|[\u0900-\u097f]+|\d", _norm(text).translate(DEVANAGARI_DIGITS)):
        if tok in MULTIPLIERS:
            pending = MULTIPLIERS[tok]
        elif tok.isdigit():
            out.append(tok * pending)
            pending = 1
        elif tok in WORD_DIGITS:
            out.append(str(WORD_DIGITS[tok]) * pending)
            pending = 1
    return "".join(out)


SCALES = {
    "hazaar": 1_000, "hazar": 1_000, "hajaar": 1_000, "thousand": 1_000,
    "हज़ार": 1_000, "हजार": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "लाख": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "karod": 10_000_000, "karor": 10_000_000,
    "करोड़": 10_000_000, "करोड": 10_000_000,
}
TENS = {"das": 10, "ten": 10, "bees": 20, "twenty": 20, "pachas": 50, "pachaas": 50,
        "fifty": 50, "sau": 100, "hundred": 100, "पचास": 50, "सौ": 100,
        # romanisation variants a recogniser actually emits. pachees is 25, not 50 —
        # mapped to what it means, so a 50/25 substitution surfaces as an error
        # rather than being silently absorbed.
        "pachees": 25, "pachchees": 25, "twentyfive": 25, "pandra": 15, "pandrah": 15,
        "solah": 16, "chaudah": 14, "terah": 13, "barah": 12, "gyarah": 11}
HALVES = {"saade", "sade", "saadhe", "sadhe", "साढ़े", "साढे"}  # "saade teen" = 3.5
CURRENCY = re.compile(r"(?:\brs\b|\binr\b|rupees?|rupaye|rupaya|₹|रुपये|रुपया)", re.I)


def spoken_amount(text: str) -> str:
    """'do lakh pachas hazaar' -> 250000.

    Indian grouping is not a scaled version of thousands/millions: lakh and crore are
    their own scales, and a fractional prefix ("saade teen lakh" = 3.5 lakh) attaches
    to the number *before* the scale. Neither survives a Western number normaliser,
    which is the whole reason this metric exists.
    """
    toks = re.findall(r"\d+\.\d+|[a-z]+|[\u0900-\u097f]+|\d+",
                      _norm(text).translate(DEVANAGARI_DIGITS))
    total, cur, half, frac = 0.0, None, False, None
    for t in toks:
        if t in HALVES:
            half = True
        elif t in ("point", "dashamlav"):
            frac = ""
        elif "." in t:
            cur = float(t)  # "1.25 lakh" — already a decimal, not two numbers
        elif t.isdigit() or t in WORD_DIGITS or t in TENS:
            d = int(t) if t.isdigit() else WORD_DIGITS.get(t, TENS.get(t))
            if frac is not None:
                frac += str(d)
            elif t in TENS and cur:
                cur += d
            else:
                cur = d if cur is None else cur * 10 + d
        elif t in SCALES:
            if cur is None and not half:
                cur = 1.0  # "lakh rupaye" with no quantity in front
            elif cur is None:
                # "saadhe <unrecognised> hazaar" — the quantity word did not parse.
                # Guessing 1 would assert a wrong number; say nothing instead.
                return ""
            v = float(cur) + (0.5 if half else 0.0)
            if frac:
                v = float(f"{int(v)}.{frac}")
            total += v * SCALES[t]
            cur, half, frac = None, False, None
    if cur is not None:
        total += cur + (0.5 if half else 0)
    return str(int(round(total))) if total else ""


MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
MONTHS |= {m[:3]: i + 1 for m, i in ((k, v - 1) for k, v in MONTHS.items())}


def spoken_date(v: str) -> str:
    """-> YYYY-MM-DD. Day-first throughout: 2/1/2026 is 2 January, not 1 February.

    That ambiguity is not an edge case here, it is one of the things being measured.
    """
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", v):
        return m.group(0)
    if m := re.search(r"\b(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})\b", v):
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    if m := re.search(r"\b(\d{2})(\d{2})(\d{4})\b", v):  # DDMMYYYY run-together
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    # "pandra tarik, september 2026" / "the third of next month"
    if m := re.search(r"\b(" + "|".join(MONTHS) + r")\w*\b", v):
        mo = MONTHS[m.group(1)]
        year = (y.group(0) if (y := re.search(r"\b(20\d{2})\b", v)) else "")
        day = None
        for tok in re.findall(r"[a-z]+|\d{1,2}", v):
            if tok.isdigit() and 1 <= int(tok) <= 31 and tok != year[:2]:
                day = int(tok)
                break
            if tok in TENS and TENS[tok] <= 31:
                day = TENS[tok]
                break
            if tok in WORD_DIGITS:
                day = WORD_DIGITS[tok]
                break
        if day and year:
            return f"{year}-{mo:02d}-{day:02d}"
    return ""


def normalise(entity_type: str, value: str) -> str:
    if value is None:
        return ""
    v = _norm(str(value)).strip().translate(DEVANAGARI_DIGITS)
    if entity_type == "digits":
        d = re.sub(r"\D", "", v)
        return d or spoken_digits(v)
    if entity_type == "amount":
        # a bare figure is the agent's canonical answer; anything wordier is speech.
        # currency markers are not words for this purpose — "Rs 2,50,000" is a figure.
        stripped = CURRENCY.sub("", v)
        bare = re.sub(r"[^\d]", "", stripped.replace(",", ""))
        if bare and not re.search(r"[a-z\u0900-\u097f]", stripped):
            return str(int(bare))
        return spoken_amount(v) or (str(int(bare)) if bare else "")
    if entity_type == "date":
        return spoken_date(v) or v
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
    return re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", _norm(text))


def mangled_entity(spec: CallSpec, transcript: str) -> bool | None:
    """Did the recogniser damage the *entity*, as opposed to any word in the turn?

    Script-independent: compare the *value* the transcript implies against the truth,
    not the words. The recogniser may answer in Devanagari, in roman, with digit words
    or with digits already normalised — all four are the same entity if they carry the
    same number, and only a value comparison sees that.

    The truth is sought as a substring of the digits recovered from the whole turn,
    because non-entity words legitimately contribute digits ("last five digits" yields
    a 5 that belongs to no account number).

    Returns None when nothing numeric can be recovered at all — abstaining beats
    inventing damage. SFR_bb still covers those calls.

    ponytail: substring containment can false-negative on very short entities that
    occur by chance elsewhere in the turn; exact span alignment if the bank grows.
    """
    if spec.entity_type in ("digits", "amount"):
        truth = normalise(spec.entity_type, spec.canonical)
        got = normalise(spec.entity_type, transcript)
        if not got:
            return None
        return truth not in got

    want, got_words = set(_words(spec.segments[-1].text)), set(_words(transcript))
    if not want & got_words and _devanagari(transcript) and not _devanagari(spec.text):
        return None
    return not want <= got_words


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

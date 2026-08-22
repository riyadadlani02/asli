"""The turn-taking intervention, separate from the metrics that score it.

`asli.score` measures what happened. This decides what to do, and the two must not share
a function: a discourse marker means the speaker is unfinished but it is not case
marking, and letting it inflate `cut_dangling_*` would corrupt the measurement with the
intervention.

Seven things this handles that a bare word list did not:

1. HOLDS ARE BOUNDED. "Wait for the rest" has no answer when the rest never comes — the
   speaker really did stop, or hung up, or the line dropped. A hold expires.
2. NEVER MERGE ACROSS AGENT SPEECH. If the agent started talking, what the caller says
   next may be answering the agent, and gluing it onto the previous turn produces a
   sentence neither of them said.
3. FINALITY MARKERS, the opposite direction. Words that mean definitely-finished, so the
   agent can answer sooner instead of waiting out the full gate. Derived from the corpus
   rather than asserted — see experiments/derive_markers.py.
4. READ THE LIVE TRANSCRIPT, NOT THE FINAL. This project measured a recogniser returning
   `speech_final` with an empty transcript while the interim carried the words. Deciding
   off the final means deciding off text this repo has already shown can be truncated, so
   the caller passes whichever is longer.
5. ENGLISH AND HINGLISH FILLERS. Real callers say "so", "I mean", "actually" as readily
   as मतलब.
6. BOTH SIDES ARE COUNTED. `Ledger` tallies holds and expiries together, because a rule
   that only reports prevented interruptions can only produce good news.
7. A MERGED TURN KEEPS ITS SEAM. Gluing two turns into one string walks straight into a
   bug this repo already paid for: `score.spoken_digits` concatenates every digit-like
   token in whatever it is handed, so "मेरा एक नंबर है उसका" merged with "नौ आठ सात सात"
   extracts 19877 rather than 9877 — a wrong number asserted with confidence, which is
   the failure this project is named after. The merged text is for turn-taking. Anything
   pulling an entity out of it reads `parts` and takes the segment, not the join.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .score import DANGLING_FILLER, DANGLING_MARGINAL, DANGLING_STRICT

# Hindi/Hinglish and English discourse markers. The English set is UNVALIDATED: the only
# corpus here is Hindi, so these are reasoned rather than measured, and are kept apart so
# that stays visible. "so" and "like" are content words too — a turn genuinely ending on
# one is rare but not impossible, and that cost lands in the expiry column.
FILLER_EN = frozenset("umm um uh er hmm like actually basically anyway".split())
FILLER_EN_PHRASES = ("i mean", "you know", "sort of", "kind of")

DANGLING = DANGLING_STRICT | DANGLING_MARGINAL | DANGLING_FILLER

# Counted, not reasoned: results/markers.json holds P(final|word) against a random-cut
# baseline with Bonferroni correction, and this is the half of that list that clears the
# threshold. The threshold is the dial — lower it and the agent answers sooner on weaker
# evidence, at the cost of the very interruption this project measures.
FINALITY_MIN_P = 0.5
# धन्यवाद: 107 finals in 140, 19.4x baseline. The only survivor at this threshold.
FINALITY_DERIVED = frozenset("धन्यवाद".split())
# UNMEASURED, kept separate for the same reason FILLER_EN is: शुक्रिया does not occur once
# in the corpus, so it is धन्यवाद's synonym by argument and not by count.
FINALITY_ASSERTED = frozenset("शुक्रिया".split())
FINALITY = FINALITY_DERIVED | FINALITY_ASSERTED
# बस is the case for counting rather than asserting. It reads as "that's it, enough" and
# was in this set on that reading; the corpus ends an utterance on it 1 time in 6, and
# score.DANGLING_FILLER already had it as the other बस, "just/only". Reasoning had the
# sign wrong. It stays a dangler, and the cost asymmetry says so too: a false finality
# produces the interruption this repo exists to measure, a false hold costs one gate.
# है/हैं are derived at ~4x baseline and still excluded — 15% final is not finality, and
# they are exactly the words the real-caller misses ended on. Replying faster there would
# cut off the callers the rule is meant to hold.
assert not FINALITY & DANGLING, "a word cannot mean both finished and not"


@dataclass
class Decision:
    hold: bool
    reason: str
    hold_ms: int = 0


@dataclass
class Ledger:
    """Both columns. A hold that expired is a caller left waiting for nothing."""

    held: int = 0
    expired: int = 0
    merged: int = 0
    refused_agent_spoke: int = 0
    early: int = 0
    by_reason: dict = field(default_factory=dict)

    @property
    def false_hold_rate(self) -> float | None:
        return round(self.expired / self.held, 4) if self.held else None

    def added_latency_ms(self, hold_ms: int, turns: int) -> float | None:
        """Mean latency this cost per turn, which is what a gate raise must be compared to."""
        return round(self.expired * hold_ms / turns, 1) if turns else None


def last_token(text: str) -> str:
    w = (text or "").strip().split()
    return w[-1].strip("।,.?!‍").lower() if w else ""


def best_text(final: str, partial: str = "") -> str:
    """Whichever carries more. A `speech_final` can arrive empty after a hesitation while
    the interim held the whole sentence — measured on Deepgram in this repo."""
    return final if len(final.strip()) >= len(partial.strip()) else partial


def decide(final: str, partial: str = "", *, agent_spoke_since: bool = False,
           hold_ms: int = 600) -> Decision:
    """Hold this turn open, or let it end?"""
    if agent_spoke_since:
        return Decision(False, "agent-spoke")
    text = best_text(final, partial)
    low = " " + text.lower() + " "
    tok = last_token(text)
    if not tok:
        return Decision(False, "empty")
    if tok in FINALITY:
        return Decision(False, "finality-marker")
    if tok in DANGLING:
        return Decision(True, "dangling", hold_ms)
    if tok in FILLER_EN or any(low.endswith(f" {p} ") for p in FILLER_EN_PHRASES):
        return Decision(True, "filler-en", hold_ms)
    return Decision(False, "complete")


def apply(turns: list[dict], *, hold_ms: int = 600, ledger: Ledger | None = None) -> list[dict]:
    """Replay a turn sequence under the policy.

    Each turn needs `text`, `t_ms`, and optionally `partial` and `agent_spoke_before`.
    A held turn merges with the next only if it arrives within `hold_ms` and the agent
    stayed silent; otherwise the hold expires and the turn stands as it was. The merged
    turn carries `parts`, and an entity extractor must use those rather than `text` —
    see point 7 above.
    """
    led = ledger or Ledger()
    out: list[dict] = []
    pending: dict | None = None
    for t in turns:
        if pending is not None:
            gap = t["t_ms"] - pending["t_ms"]
            if t.get("agent_spoke_before"):
                led.refused_agent_spoke += 1
                out.append(pending)
            elif gap <= hold_ms:
                led.merged += 1
                parts = pending.get("parts", [pending["text"]]) + [t["text"]]
                t = {**t, "text": " ".join(p for p in parts if p).strip(),
                     "parts": parts, "t_ms": pending["t_ms"], "merged": True}
            else:
                led.expired += 1
                out.append(pending)
            pending = None
        d = decide(t.get("text", ""), t.get("partial", ""),
                   agent_spoke_since=t.get("agent_spoke_before", False), hold_ms=hold_ms)
        led.by_reason[d.reason] = led.by_reason.get(d.reason, 0) + 1
        if d.reason == "finality-marker":
            led.early += 1
        if d.hold:
            led.held += 1
            pending = t
        else:
            out.append(t)
    if pending is not None:
        led.expired += 1          # nothing followed at all: the hold ran out on silence
        out.append(pending)
    return out

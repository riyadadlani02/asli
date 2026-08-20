"""The turn-taking intervention, separate from the metrics that score it.

`asli.score` measures what happened. This decides what to do, and the two must not share
a function: a discourse marker means the speaker is unfinished but it is not case
marking, and letting it inflate `cut_dangling_*` would corrupt the measurement with the
intervention.

Six things this handles that a bare word list did not:

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

# Derived, not asserted: results/markers.json, P(final|word) against a random-cut
# baseline with Bonferroni correction. है and हैं are here at ~4x baseline, which is the
# same finding the verb-final experiment reached from the other side.
FINALITY = frozenset("धन्यवाद शुक्रिया बस".split())

DANGLING = DANGLING_STRICT | DANGLING_MARGINAL | DANGLING_FILLER


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
    stayed silent; otherwise the hold expires and the turn stands as it was.
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
                t = {**t, "text": (pending["text"] + " " + t["text"]).strip(),
                     "t_ms": pending["t_ms"], "merged": True}
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

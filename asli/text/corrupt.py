"""Authored retrieval errors.

The corruption is known by construction, so `damaged` costs nothing and cannot
drift — the same free-ground-truth trick the audio splice uses on the voice side.

Every class carries a `detectable` flag decided *here*, at authoring time, never by
a judge. That split is the internal-validity check: if the detectable classes and
the undetectable one score the same SFR, the metric is reading the agent's
confirm-rate rather than its sensitivity to error. See docs/sfr-text.md.
"""

from __future__ import annotations

import re

# Only words the amount parser actually knows. A swap that fails to parse would be
# an unparseable string, not a plausible alternative value — and plausibility is the
# entire point of the undetectable class.
WORD_SWAP = {"ek": "do", "do": "teen", "teen": "do", "nau": "paanch",
             "paanch": "nau", "bees": "pachas", "pachas": "bees"}
SCALE_SWAP = {"lakh": "crore", "crore": "lakh"}

FORMAT_CUE = ("System note: dates in this export follow the legacy system's "
              "MM/DD/YYYY convention.")


def _indian(n: int) -> str:
    """2,50,000 — last three digits, then groups of two."""
    s, parts = str(n), []
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + parts + [tail])


def _render(item: dict, surface: str | None = None, drop: bool = False,
            extra: str = "") -> list[str]:
    recs = list(item["records"])
    if not drop:
        recs.insert(1, item["slot"].format(value=item["surface"] if surface is None else surface))
    if extra:
        recs.append(extra)
    return recs


def _swapped(surface: str) -> str | None:
    """One digit changed, or one quantity word changed — still well-formed either way."""
    pos = [i for i, c in enumerate(surface) if c.isdigit()]
    if pos:
        i = pos[len(pos) // 2]  # a middle digit, so no leading zero is ever created
        return surface[:i] + str((int(surface[i]) + 3) % 10) + surface[i + 1:]
    toks = surface.split()
    for i, t in enumerate(toks):
        if t.lower() in WORD_SWAP:
            toks[i] = WORD_SWAP[t.lower()]
            return " ".join(toks)
    return None


def clean(item: dict) -> list[str]:
    return _render(item)


def truncation(item: dict) -> list[str] | None:
    """The value arrives cut mid-token, the way a chunker splits a row."""
    s = item["surface"]
    return _render(item, s[:-2]) if len(s) > 3 else None


def omission(item: dict) -> list[str] | None:
    """The field is simply not in what was retrieved — any answer is unsupported."""
    return _render(item, drop=True)


def contradiction(item: dict) -> list[str] | None:
    other = _swapped(item["surface"])
    if not other:
        return None
    return _render(item, extra="Archived copy of the same record — "
                               + item["slot"].format(value=other))


def scale_swap(item: dict) -> list[str] | None:
    """Right figure, wrong scale: lakh read as crore. Detectable by magnitude."""
    if item["entity_type"] != "amount":
        return None
    s = item["surface"]
    for k, v in SCALE_SWAP.items():
        if re.search(rf"\b{k}\b", s):
            return _render(item, re.sub(rf"\b{k}\b", v, s))
    d = re.sub(r"\D", "", s)
    return _render(item, ("Rs " if "Rs" in s else "") + _indian(int(d) * 100)) if d else None


def format_flip(item: dict) -> list[str] | None:
    """Day and month exchanged, with the locale cue that makes it recoverable.

    A careful agent can get the truth back from the cue; an eager one takes the
    surface at face value. That is the point — detectable does not mean easy.
    """
    if item["entity_type"] != "date":
        return None
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", item["surface"])
    if not m or m.group(1) == m.group(2):
        return None
    d, mo, y = m.groups()
    return _render(item, f"{mo}/{d}/{y}", extra=FORMAT_CUE)


def digit_swap(item: dict) -> list[str] | None:
    """The control. One digit off, perfectly well-formed, nothing in the records
    marks it. Every agent should score ~1.00 here — that is what makes it the row
    that proves the other rows are measuring something."""
    other = _swapped(item["surface"])
    return _render(item, other) if other else None


# name -> (builder, detectable from the records alone)
CLASSES: dict[str, tuple] = {
    "truncation": (truncation, True),
    "omission": (omission, True),
    "contradiction": (contradiction, True),
    "scale_swap": (scale_swap, True),
    "format_flip": (format_flip, True),
    "digit_swap": (digit_swap, False),
}

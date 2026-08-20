"""Derive turn-final markers from the corpus instead of asserting them.

Two directions, and only one of them is learnable from this data.

DANGLERS — words that mean "not finished" — are NOT derivable here. The recordings are
fixed-length segments cut mid-phrase, so a segment ending is close to a random position
in the sentence. Measured: known danglers end segments at 0.039, everything else at
0.045, against a random baseline of 0.039. No signal, and a list built from it would be
noise wearing a number. That half of the rule stays hand-written and says so.

FINALITY markers — words that mean "definitely finished" — are derivable, because a
speaker who says धन्यवाद and stops did that on purpose and a random cut cannot imitate
it. Those are the words worth replying *faster* on, and the same counting works in any
language without knowing the language.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MIN_COUNT = 25


def load(path: Path) -> list[list[str]]:
    out = []
    for line in path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) > 1:
            toks = [t.strip("।,.?!") for t in parts[1].replace("<inaudible>", " ").split()]
            toks = [t for t in toks if t and not t.startswith("<")]
            if len(toks) > 3:
                out.append(toks)
    return out


def main() -> int:
    utts = load(ROOT / "corpus/GV_Dev_5h/text")
    tot, fin = collections.Counter(), collections.Counter()
    for toks in utts:
        tot.update(toks)
        fin[toks[-1]] += 1
    n_tok, n_utt = sum(tot.values()), len(utts)
    base = n_utt / n_tok  # what a random cut position would give any word

    rows = []
    for w, c in tot.items():
        if c < MIN_COUNT:
            continue
        p = fin[w] / c
        # one-sided binomial tail: how surprising is this many finals under a random cut.
        # Computed in log space — comb(1501, k) overflows a float outright.
        def logpmf(k: int) -> float:
            return (math.lgamma(c + 1) - math.lgamma(k + 1) - math.lgamma(c - k + 1)
                    + k * math.log(base) + (c - k) * math.log1p(-base))

        peak = max(logpmf(k) for k in range(fin[w], c + 1))
        tail = math.exp(peak) * sum(math.exp(logpmf(k) - peak)
                                    for k in range(fin[w], c + 1))
        rows.append({"word": w, "count": c, "final": fin[w], "p_final": round(p, 4),
                     "lift": round(p / base, 2), "p_value": tail})
    rows.sort(key=lambda r: -r["lift"])
    keep = [r for r in rows if r["p_value"] < 0.001 / len(rows) and r["lift"] >= 2]

    out = {"n_utterances": n_utt, "n_tokens": n_tok, "baseline_p_final": round(base, 4),
           "min_count": MIN_COUNT, "bonferroni_n": len(rows),
           "finality_markers": keep,
           "dangler_note": "not derivable from this corpus — segment ends are cut "
                           "mid-phrase, so known danglers sit at the random baseline"}
    (ROOT / "results/markers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print(f"{n_utt} utterances, {n_tok} tokens, baseline P(final|w) = {base:.4f}")
    print(f"tested {len(rows)} words with count >= {MIN_COUNT}, "
          f"Bonferroni threshold p < {0.001/len(rows):.2e}\n")
    print(f"  {'word':<14}{'count':>7}{'final':>7}{'P(final)':>10}{'lift':>7}")
    for r in keep:
        print(f"  {r['word']:<14}{r['count']:>7}{r['final']:>7}{r['p_final']:>10.3f}{r['lift']:>7.1f}x")
    print(f"\n{len(keep)} finality markers survive correction. wrote results/markers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

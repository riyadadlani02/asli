"""What the rule costs, against what raising the gate costs. No calls — all stored.

Raising `silence_duration_ms` is paid by every turn of every call. The rule is paid only
when it fires, and it is only *wrongly* paid when it fires on a turn that had actually
finished. So the whole latency argument reduces to one number: the false-positive rate.

The cost model for one false hold is a further endpointing cycle. The turn already ended
after `gate` ms of silence; holding it open means waiting out `gate` ms again before it
can end. So a false hold costs +gate, and expected latency is fp * gate.

The real-speech corpus CANNOT measure this and is not used. Its entries are fixed-length
segments that end mid-phrase — `...अगर`, `...की`, `...लेकिन` — so a dangling final token
there is the recording being cut, not the rule being wrong. Measuring against them would
have produced a 19.9% "false-positive rate" that is nothing of the kind.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from asli import fit as fitmod
from asli.score import policy_holds

ROOT = Path(__file__).parent.parent
GATE_MS = 500


def complete_utterances() -> list[str]:
    """Transcripts whose last word is a true ending, because we built the utterance.

    One turn back from the socket means nothing was split off, so the text is the whole
    thing. This is the only ground truth available for the finished case.
    """
    out = [r["first"] for r in json.loads((ROOT / "results/verbfinal2.json").read_text())
           if r["turns"] == 1 and not r.get("error") and r["first"].strip()]
    for name in ("inepa_clean.jsonl", "inepa.jsonl"):
        path = ROOT / "results" / name
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    t = (json.loads(line).get("result") or {}).get("transcript", "")
                    if t.strip():
                        out.append(t)
    return out


def main() -> int:
    utts = complete_utterances()
    fp = [t for t in utts if policy_holds(t)]
    n, k = len(utts), len(fp)
    # zero events: the 95% upper bound is what carries the claim, not the point estimate
    upper = 1 - 0.05 ** (1 / n) if k == 0 else None
    fit = json.loads((ROOT / "results/pause_fit.json").read_text())
    advice = {r["gate_ms"]: r for r in fitmod.gate_advice(fit)}

    out = {
        "n_complete": n, "false_holds": k,
        "fp_rate": round(k / n, 4),
        "fp_upper95": round(upper, 4) if upper is not None else None,
        "gate_ms": GATE_MS,
        "policy_expected_ms": round((upper if upper is not None else k / n) * GATE_MS, 1),
        "gate_raise_ms": advice[700]["added_latency_ms"],
        "breakeven_fp": round(advice[700]["added_latency_ms"] / GATE_MS, 4),
        "gate_raise_calls_helped": advice[700]["calls_affected"],
        "gate_raise_calls_helped_at_default": advice[500]["calls_affected"],
        "corpus_note": "every utterance here ends on an entity (digits or an amount), "
                       "which is the easiest ending for the rule to get right",
    }
    (ROOT / "results/policy_latency.json").write_text(json.dumps(out, indent=1))

    print(f"  complete utterances       {n}")
    print(f"  rule wrongly holds        {k}   ({out['fp_rate']:.2%})")
    if upper is not None:
        print(f"  95% upper bound on that   {upper:.2%}   (zero events, so the bound is the claim)")
    print(f"\n  expected added latency, rule        <= {out['policy_expected_ms']:.0f} ms per turn")
    print(f"  added latency, gate 500 -> 700        {out['gate_raise_ms']} ms per turn, always")
    print(f"  rule would have to be wrong on        {out['breakeven_fp']:.0%} of finished turns to cost more")
    print(f"\n  wrote results/policy_latency.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

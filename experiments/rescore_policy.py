"""Re-score the shipped policy against stored rows. No calls — the transcripts are here.

The rule lives in asli.score. This walks results/verbfinal2.json, recomputes the policy
verdict on each turn's own text, and prints what changed. Run it after editing the word
lists: the whole point of storing transcripts is that the intervention can be re-measured
without paying for the audio again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from asli import score

ROWS = Path(__file__).parent.parent / "results/verbfinal2.json"
ARMS = ("dangler", "filler", "verb-final")


def main() -> int:
    rows = json.loads(ROWS.read_text())
    before = sum(r["lexical_would_hold"] for r in rows)
    for r in rows:
        r["lexical_would_hold"] = score.policy_holds(r["first"] or "")
    after = sum(r["lexical_would_hold"] for r in rows)

    ok = [r for r in rows if r["turns"] > 0 and not r.get("error")]
    print(f"policy fires on {before} -> {after} of {len(rows)} rows\n")
    print(f"  {'arm':<12}{'cuts':>6}{'rescued':>9}{'residual':>10}")
    for arm in ARMS:
        cut = [r for r in ok if r["mode"] == "server_vad" and r["arm"] == arm and r["split"]]
        held = sum(r["lexical_would_hold"] for r in cut)
        print(f"  {arm:<12}{len(cut):>6}{held:>9}{round(1 - held / len(cut), 3) if cut else '-':>10}")

    sem = [r for r in ok if r["mode"] == "semantic_vad" and r["split"]]
    print(f"\n  on the {len(sem)} cuts semantic detection still made, the policy holds "
          f"{sum(r['lexical_would_hold'] for r in sem)}")
    for r in sem:
        print(f"    {r['id']:<7} {r['arm']:<11} policy="
              f"{'HOLD' if r['lexical_would_hold'] else 'cannot fire':<12} {r['first'][-28:]!r}")

    ROWS.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nrewrote {ROWS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

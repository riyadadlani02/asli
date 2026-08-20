"""Experiment B on the Gemini lane. Same corpus, same arms, same splice.

Two things are NOT the same as the OpenAI run, and neither can be waved away:

  * Gemini Live has one activity-detection mode, not a silence-timer/semantic pair. So
    there is no within-lane baseline here — the silence-timer column stays OpenAI's.
  * Gemini emits no turn-end event. The observable equivalent is the first chunk of its
    reply: the moment an agent built on it would start talking over the caller. That
    carries model response latency, so it lands LATER than the real decision and makes a
    cut look less early than it was. Every number here is therefore a LOWER bound on
    prematurity, and must not be put in one table with a lane that reports its own VAD.

The item set is the 31 that cleared pre-flight on the OpenAI lane. Reused deliberately:
a different corpus per vendor would make the two uncomparable for a second reason.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from asli import score, synth
from asli.cli import load_env
from asli.drive import GeminiLive
from run_verbfinal2 import ARMS, build, cached_preflight

load_env()
GATE = 500
OUT = Path("results/verbfinal_gemini.json")


async def main() -> int:
    items = cached_preflight(yaml.safe_load(
        open(Path(__file__).parent / "verbfinal2.yaml")))
    if not items:
        print("no cached pre-flight — run run_verbfinal2.py first", file=sys.stderr)
        return 2
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["id"], r["arm"]) for r in rows if not r.get("error")}
    print(f"{len(items)} items x {len(ARMS)} arms = {len(items) * len(ARMS)} calls"
          f"{f', {len(done)} already done' if done else ''}\n")

    for item in items:
        for arm in ARMS:
            if (item["id"], arm) in done:
                continue
            spec = build(item, arm)
            pcm = synth.render(spec)
            res = await GeminiLive(language_code="hi", rate=16000,
                                   silence_duration_ms=GATE).run(pcm, spec)
            ends = [e.t_ms for e in res.events if e.kind == "speech_end"]
            p = score.pir(spec, res)
            truth = score.normalise(item["entity_type"], item["canonical"])
            got = score.normalise(item["entity_type"], res.transcript)
            rows.append({"id": item["id"], "arm": arm, "verb": item["verb"],
                         "replied_at_ms": ends[0] if ends else None,
                         "true_end_ms": spec.true_end_ms,
                         "premature": p.premature, "in_pause": p.in_injected_pause,
                         "ms_early": p.ms_early, "has_answer": bool(got) and truth in got,
                         "transcript": res.transcript, "error": res.error})
            OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            r = rows[-1]
            print(f"  {item['id']:<7} {arm:<11} "
                  f"{'ERR' if r['error'] else 'CUT' if r['premature'] else 'held':<5} "
                  f"reply={r['replied_at_ms']} true_end={r['true_end_ms']} "
                  f"ans={'Y' if r['has_answer'] else 'n'} "
                  f"{(r['error'] or r['transcript'])[:38]!r}", flush=True)

    ok = [r for r in rows if not r["error"] and r["replied_at_ms"] is not None]
    print(f"\n\nreplied before the caller finished  (n={len(ok)} usable of {len(rows)})\n")
    print(f"  {'arm':<12}{'cut':>6}{'n':>5}{'rate':>8}{'median ms early':>18}")
    for arm in ARMS:
        sel = [r for r in ok if r["arm"] == arm]
        cut = [r for r in sel if r["premature"]]
        early = sorted(r["ms_early"] for r in cut) if cut else []
        print(f"  {arm:<12}{len(cut):>6}{len(sel):>5}"
              f"{round(len(cut)/len(sel), 3) if sel else '-':>8}"
              f"{early[len(early)//2] if early else '-':>18}")
    ans = [r for r in ok if r["has_answer"]]
    print(f"\n  entity survived in the input transcription: {len(ans)}/{len(ok)}")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

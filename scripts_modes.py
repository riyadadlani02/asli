"""Does a pre-entity hesitation lose the entity, and does it depend on output mode?

Control: the same filler placed at the head of the utterance, where it precedes
nothing. Any difference between the two placements is the hesitation's effect.
"""
import asyncio, json
from asli.spec import CallSpec, Segment
from asli.cli import load_env
from asli import synth, score
from asli.drive import SarvamWS
load_env()

CASES = [("dig-01", "9877111", "Mera mobile number hai", "nine eight double seven, triple one"),
         ("dig-02", "40556", "Account ke last five digits", "char zero double five six"),
         ("dig-03", "8009", "The reference is", "eight, double zero, nine"),
         ("dig-04", "77712345", "Number likhiye,", "triple seven one two three four five")]
rows = []
for mode in ("transcribe", "verbatim", "translit", "codemix"):
    for sid, truth, head, ent in CASES:
        for place, segs in (
            ("control (filler leading)",
             [Segment("matlab", pause_after_ms=700), Segment(f"{head} {ent}")]),
            ("hesitation (pre-entity)",
             [Segment(head), Segment("matlab", pause_after_ms=700), Segment(ent)]),
        ):
            spec = CallSpec(id=sid, entity_type="digits", canonical=truth, segments=segs)
            pcm = synth.render(spec)
            for run in range(2):
                r = asyncio.run(SarvamWS(mode=mode, silence_duration_ms=2000).run(pcm, spec))
                got = score.normalise("digits", r.transcript)
                ok = bool(got) and truth in got
                rows.append({"mode": mode, "id": sid, "placement": place, "run": run,
                             "truth": truth, "extracted": got, "survived": ok,
                             "transcript": r.transcript})
                print(f"{mode:<11} {sid:<7} {place:<25} run{run} "
                      f"{'OK  ' if ok else 'LOST'} {r.transcript!r}", flush=True)
json.dump(rows, open("results/mode_placement.json", "w"), indent=2, ensure_ascii=False)
print("\nwrote results/mode_placement.json")

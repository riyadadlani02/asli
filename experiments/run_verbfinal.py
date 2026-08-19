"""Experiment B — is semantic end-of-turn detection SVO-shaped?

Minimal pair per item: identical head, identical entity, hesitation spliced in the
same place. One arm puts a filler between them, the other lets the pause fall straight
after a finite verb. The lexical policy can only fire on the first.

Both arms go through the same model on the same socket, once with a silence timer and
once with semantic detection, so nothing but the turn-detection mode differs.
"""
from __future__ import annotations

import asyncio, base64, json, os, sys
import numpy as np, websockets, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from asli.cli import load_env
from asli.spec import CallSpec, Segment
from asli import synth, score

load_env()
KEY = os.environ["OPENAI_API_KEY"]
NATIVE = 24000
PAUSE_MS = 700

DANGLING = set("का के की को में से ने तक पर तो और यह वह जो कि क्योंकि इस उस लिए अगर जबकि".split())
FILLERS = set("मतलब यानी वो ऐसा हाँ".split())

def lexical_holds(text: str) -> bool:
    """The shipped policy: would it hold this turn open?"""
    w = (text or "").strip().split()
    if not w:
        return False
    last = w[-1].strip("।,.?!")
    return last in DANGLING or last in FILLERS

def to_native(pcm):
    r = NATIVE / 16000
    n = int(len(pcm) * r); idx = np.arange(n) / r
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, len(pcm) - 1)
    f = idx - lo
    out = (pcm[lo] * (1 - f) + pcm[hi] * f).astype(np.int16)
    return np.concatenate([out, np.zeros(int(NATIVE * 2.5), np.int16)])

async def run_one(pcm, td):
    sess = {"type": "session.update", "session": {"type": "transcription",
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": NATIVE},
                  "transcription": {"model": "gpt-4o-transcribe", "language": "hi"},
                  "turn_detection": td}}}}
    ends, finals, sent = [], [], 0
    async with websockets.connect("wss://api.openai.com/v1/realtime?intent=transcription",
            additional_headers={"Authorization": f"Bearer {KEY}"}, max_size=None) as ws:
        await ws.send(json.dumps(sess))
        async def pump():
            nonlocal sent
            for i in range(0, len(pcm), 2400):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i+2400].astype("<i2").tobytes()).decode()}))
                sent += 100
                await asyncio.sleep(0.1)
        t = asyncio.create_task(pump())
        async def rd():
            async for raw in ws:
                m = json.loads(raw); ty = m.get("type", "")
                if "speech_stopped" in ty: ends.append(sent)
                elif "transcription" in ty and "completed" in ty:
                    finals.append(m.get("transcript", ""))
                    if len(finals) >= 2: return
        try: await asyncio.wait_for(rd(), timeout=len(pcm)/NATIVE + 16)
        except Exception: pass
        t.cancel()
    return ends, finals

def build(item, arm):
    segs = [Segment(item["head"])]
    if arm == "filler":
        segs.append(Segment("matlab", pause_after_ms=PAUSE_MS))
    else:
        segs[-1] = Segment(item["head"], pause_after_ms=PAUSE_MS)
    segs.append(Segment(item["entity"]))
    return CallSpec(id=f"{item['id']}-{arm}", entity_type=item["entity_type"],
                    canonical=item["canonical"], segments=segs)

async def main():
    items = yaml.safe_load(open(Path(__file__).parent / "verbfinal.yaml"))
    rows = []
    for item in items:
        for arm in ("filler", "verb-final"):
            spec = build(item, arm)
            pcm = to_native(synth.render(spec))
            for mode, td in (("server_vad", {"type": "server_vad", "silence_duration_ms": 500}),
                             ("semantic_vad", {"type": "semantic_vad"})):
                ends, finals = await run_one(pcm, td)
                first = finals[0] if finals else ""
                got = score.normalise(item["entity_type"], first)
                truth = score.normalise(item["entity_type"], item["canonical"])
                rows.append({"id": item["id"], "arm": arm, "mode": mode, "verb": item["verb"],
                             "turns": len(finals), "ends": ends[:3], "first": first,
                             "has_answer": bool(got) and truth in got,
                             "lexical_would_hold": lexical_holds(first)})
                print(f"  {item['id']} {arm:<10} {mode:<12} turns={len(finals)} "
                      f"answer={'Y' if rows[-1]['has_answer'] else 'n'} "
                      f"lex={'hold' if rows[-1]['lexical_would_hold'] else '—':<4} {first[:42]!r}",
                      flush=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(rows, open("results/verbfinal.json", "w"), ensure_ascii=False, indent=1)
    print("\nwrote results/verbfinal.json")

asyncio.run(main())

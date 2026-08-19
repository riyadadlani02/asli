"""Experiment B, redesigned. Is semantic end-of-turn detection SVO-shaped?

Three arms, identical entity, 700 ms hesitation spliced in the same place:

    dangler     head ends on a genitive or conjunction   English-SHAPED incompleteness
    filler      head ends on a finite verb + "matlab"    the lexical policy fires
    verb-final  head ends on a finite verb               legal-looking sentence end

The dangler arm is the positive control the first attempt lacked. Without it, a split
on the verb-final arm cannot be told apart from "the model is bad at Hindi".

Two changes from the first attempt, both because it failed:

  * The dependent variable is whether the turn SPLIT, not whether the answer arrived.
    `has_answer` put recognition failure and turn-taking failure in one cell, so
    `pin code` -> `पिंक कोट` killed rows that had nothing to do with endpointing.
  * The heads are pre-flighted alone. Any that comes back as a question is dropped
    before the main run: a question is a complete utterance, so ending the turn there
    is correct behaviour, not the failure under test.

Both gates abort rather than produce an unreportable table.
"""
from __future__ import annotations

import asyncio, base64, json, os, sys
from pathlib import Path

import numpy as np, websockets, yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from asli.cli import load_env
from asli.spec import CallSpec, Event, Result, Segment
from asli import score, synth

load_env()
KEY = os.environ["OPENAI_API_KEY"]
NATIVE = 24000
PAUSE_MS = 700
MODES = {"server_vad": {"type": "server_vad", "silence_duration_ms": 500},
         "semantic_vad": {"type": "semantic_vad"}}
ARMS = ("dangler", "filler", "verb-final")
GATE = 0.80  # both gates: heads that pre-flight clean, and the dangler control

DANGLING = set("का के की को में से ने तक पर तो और यह वह जो कि क्योंकि इस उस लिए अगर जबकि".split())
FILLERS = set("मतलब यानी वो ऐसा हाँ".split())


def lexical_holds(text: str) -> bool:
    """The shipped policy's verdict, recorded per row so the data can contradict me."""
    w = (text or "").strip().split()
    return bool(w) and w[-1].strip("।,.?!") in DANGLING | FILLERS


def to_native(pcm: np.ndarray) -> np.ndarray:
    r = NATIVE / 16000
    idx = np.arange(int(len(pcm) * r)) / r
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(pcm) - 1)
    f = idx - lo
    out = (pcm[lo] * (1 - f) + pcm[hi] * f).astype(np.int16)
    return np.concatenate([out, np.zeros(int(NATIVE * 2.5), np.int16)])


async def transcribe(pcm: np.ndarray, td: dict) -> tuple[list[int], list[str]]:
    sess = {"type": "session.update", "session": {"type": "transcription",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": NATIVE},
                      "transcription": {"model": "gpt-4o-transcribe", "language": "hi"},
                      "turn_detection": td}}}}
    ends: list[int] = []
    finals: list[str] = []
    sent = 0
    async with websockets.connect("wss://api.openai.com/v1/realtime?intent=transcription",
                                  additional_headers={"Authorization": f"Bearer {KEY}"},
                                  max_size=None) as ws:
        await ws.send(json.dumps(sess))

        async def pump():
            nonlocal sent
            for i in range(0, len(pcm), 2400):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i + 2400].astype("<i2").tobytes()).decode()}))
                sent += 100
                await asyncio.sleep(0.1)

        t = asyncio.create_task(pump())

        async def read():
            async for raw in ws:
                m = json.loads(raw)
                ty = m.get("type", "")
                if "speech_stopped" in ty:
                    ends.append(sent)
                elif "transcription" in ty and "completed" in ty:
                    finals.append(m.get("transcript", ""))
                    if len(finals) >= 2:
                        return

        try:
            await asyncio.wait_for(read(), timeout=len(pcm) / NATIVE + 16)
        except Exception:
            pass
        t.cancel()
    return ends, finals


def build(item: dict, arm: str) -> CallSpec:
    """Same entity, same pause, same splice point. Only the token before it moves."""
    if arm == "dangler":
        segs = [Segment(item["dangler_head"], pause_after_ms=PAUSE_MS)]
    elif arm == "filler":
        segs = [Segment(item["verb_head"]), Segment("matlab", pause_after_ms=PAUSE_MS)]
    else:
        segs = [Segment(item["verb_head"], pause_after_ms=PAUSE_MS)]
    return CallSpec(id=f"{item['id']}-{arm}", entity_type=item["entity_type"],
                    canonical=item["canonical"], segments=[*segs, Segment(item["entity"])])


async def preflight(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Each head alone, no pause, no entity. A head that reads as a question is unusable.

    Cross-script recognition fidelity is NOT checked here — comparing a romanised head
    to a Devanagari transcript is not cheap, and it no longer matters: the dependent
    variable is whether the turn split, which does not care what the words came back as.
    A question reading is different, because it makes ending the turn correct.
    """
    ok, report = [], []
    for item in items:
        verdicts = {}
        for which in ("verb_head", "dangler_head"):
            spec = CallSpec(id=f"{item['id']}-{which}", entity_type=item["entity_type"],
                            canonical=item["canonical"], segments=[Segment(item[which])])
            _, finals = await transcribe(to_native(synth.render(spec)), MODES["server_vad"])
            text = finals[0] if finals else ""
            verdicts[which] = ("empty" if not text
                               else "question" if text.strip().rstrip("।").endswith("?")
                               else "ok")
            report.append({"id": item["id"], "head": which, "verdict": verdicts[which],
                           "text": text})
        good = all(v == "ok" for v in verdicts.values())
        print(f"  {item['id']:<7} {'PASS' if good else 'DROP':<5} "
              f"verb={verdicts['verb_head']:<9} dangler={verdicts['dangler_head']}", flush=True)
        if good:
            ok.append(item)
    return ok, report


def row_of(spec: CallSpec, item: dict, arm: str, mode: str,
           ends: list[int], finals: list[str]) -> dict:
    """`split` is primary. PIR timing is descriptive only — `ends` is sampled at the
    pump position when the event arrived, so it is biased late by network latency and
    cannot be trusted to the width of a 700 ms window."""
    res = Result(spec_id=spec.id, adapter=f"openai:{mode}",
                 events=[Event("speech_end", t) for t in ends],
                 transcript=finals[0] if finals else "")
    p = score.pir(spec, res)
    truth = score.normalise(item["entity_type"], item["canonical"])
    got = score.normalise(item["entity_type"], " ".join(finals))
    return {"id": item["id"], "arm": arm, "mode": mode, "verb": item["verb"],
            "turns": len(finals), "split": len(finals) >= 2, "ends": ends[:3],
            "pir_premature": p.premature, "pir_in_pause": p.in_injected_pause,
            "ms_early": p.ms_early, "first": finals[0] if finals else "",
            "has_answer": bool(got) and truth in got,
            "lexical_would_hold": lexical_holds(finals[0] if finals else "")}


def rate(rows: list[dict], **where) -> tuple[float | None, int]:
    sel = [r for r in rows if all(r[k] == v for k, v in where.items()) and r["turns"] > 0]
    return (round(sum(r["split"] for r in sel) / len(sel), 4), len(sel)) if sel else (None, 0)


async def main() -> int:
    items = yaml.safe_load(open(Path(__file__).parent / "verbfinal2.yaml"))
    print(f"pre-flight: {len(items)} items, {len(items) * 2} heads\n")
    ok, pf_report = await preflight(items)

    kept = len(ok) / len(items)
    print(f"\n  {len(ok)}/{len(items)} items usable ({kept:.0%})")
    Path("results").mkdir(exist_ok=True)
    json.dump(pf_report, open("results/verbfinal2_preflight.json", "w"),
              ensure_ascii=False, indent=1)
    if kept < GATE or len(ok) < 30:
        print(f"\nABORT before the main run: need >={GATE:.0%} usable and n>=30.\n"
              f"The corpus is the problem, not the hypothesis. See "
              f"results/verbfinal2_preflight.json.", file=sys.stderr)
        return 2

    print(f"\nmain run: {len(ok)} items x {len(ARMS)} arms x {len(MODES)} modes "
          f"= {len(ok) * len(ARMS) * len(MODES)} calls\n")
    rows = []
    for item in ok:
        for arm in ARMS:
            spec = build(item, arm)
            pcm = to_native(synth.render(spec))
            for mode, td in MODES.items():
                ends, finals = await transcribe(pcm, td)
                rows.append(row_of(spec, item, arm, mode, ends, finals))
                r = rows[-1]
                print(f"  {item['id']:<7} {arm:<11} {mode:<13} turns={r['turns']} "
                      f"{'SPLIT' if r['split'] else 'held ':<6} "
                      f"lex={'hold' if r['lexical_would_hold'] else '—':<4} "
                      f"{r['first'][:38]!r}", flush=True)
    json.dump(rows, open("results/verbfinal2.json", "w"), ensure_ascii=False, indent=1)

    print("\n\nsplit rate (turn ended inside the hesitation)\n")
    print(f"  {'arm':<12}{'server_vad':>14}{'semantic_vad':>16}")
    for arm in ARMS:
        cells = []
        for mode in MODES:
            v, n = rate(rows, arm=arm, mode=mode)
            cells.append(f"{v} (n={n})" if v is not None else "-")
        print(f"  {arm:<12}{cells[0]:>14}{cells[1]:>16}")

    sem_dangler, n_d = rate(rows, arm="dangler", mode="semantic_vad")
    sem_verb, n_v = rate(rows, arm="verb-final", mode="semantic_vad")
    svad = [rate(rows, arm=a, mode="server_vad")[0] for a in ARMS]

    print(f"\n  acoustic-equivalence check (server_vad across arms): {svad}")
    print("    arms are matched stimuli only if these are close to each other\n")

    if sem_dangler is None or sem_dangler > 1 - GATE:
        print(f"  CONTROL FAILED: semantic_vad split the dangler arm {sem_dangler} of the\n"
              f"  time. It cannot hold an English-shaped dangler in Hindi, so it cannot do\n"
              f"  semantic end-of-turn in Hindi at all. The verb-final comparison is NOT\n"
              f"  reportable — but this is itself the finding, and a blunter one.")
        return 0

    print(f"  CONTROL HELD: dangler {sem_dangler} (n={n_d}) — semantic detection does\n"
          f"  work on English-shaped incompleteness in Hindi.")
    print(f"  verb-final: {sem_verb} (n={n_v}).  gap = {round(sem_verb - sem_dangler, 4)}")
    print("\n  A large positive gap is the blind spot: the same model, the same language,\n"
          "  the same pause — split only where the dangling word is a legal sentence end.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

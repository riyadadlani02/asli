"""asli — run a suite, write JSONL, print the table.

    asli demo                     # synthesise, degrade, and write demo/ recordings
    asli run  --suite inepa --llm # parsing accuracy on a clean transcript
    asli sweep --suite pir        # the endpointing curve: PIR vs negative_frames_count
    asli run  --suite sfr --llm   # silent-failure rate under injected ASR error
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import degrade, score, synth
from .drive import MockASR
from .spec import CallSpec, Result, Segment, to_jsonl

ROOT = Path(__file__).parent.parent

# Hindi discourse markers that precede a mid-thought pause. Stage 3 replaces the
# fixed pause length here with one sampled from a distribution fitted to real
# telephone speech (IndicVoices / Voice of India) — until then these are nominal
# and the PIR number must be read as "at this pause length", not as a field rate.
FILLERS = ["matlab", "woh kya bolte hain", "haan toh", "aisa hai ki", "ek minute"]


def load_entities() -> list[CallSpec]:
    rows = yaml.safe_load((ROOT / "entities.yaml").read_text())
    return [CallSpec(id=r["id"], entity_type=r["entity_type"], canonical=r["canonical"],
                     lang=r.get("lang", "hi-IN"),
                     segments=[Segment(**s) for s in r["segments"]]) for r in rows]


def build_pir(base: list[CallSpec], pause_ms: int) -> list[CallSpec]:
    """Insert one filler + pause immediately before the entity in each utterance."""
    out = []
    for i, spec in enumerate(base):
        s = copy.deepcopy(spec)
        filler = FILLERS[i % len(FILLERS)]
        head = s.segments[:-1]
        s.segments = [*head, Segment(filler, pause_after_ms=pause_ms, kind="filler"), s.segments[-1]]
        s.id = f"{spec.id}-pir-{pause_ms}"
        out.append(s)
    return out


def corrupt(text: str, seed: int) -> str:
    """Injected ASR error for the SFR calibration lane. Clearly synthetic, and
    labelled as such wherever the number is reported."""
    rng = np.random.default_rng(seed)
    toks = text.split()
    swaps = {"double": "W", "triple": "W", "nine": "five", "eight": "six",
             "do": "to", "teen": "ten", "lakh": "lack", "char": "car", "saat": "sat"}
    for i, t in enumerate(toks):
        k = t.strip(",.").lower()
        if k in swaps and rng.random() < 0.7:
            toks[i] = swaps[k]
    return " ".join(toks)


def audio_for(spec: CallSpec, dials: dict) -> np.ndarray:
    pcm = synth.render(spec)
    return degrade.apply(pcm, dials, speech_for_babble=pcm) if dials else pcm


def run_suite(specs: list[CallSpec], asr: MockASR, dials: dict, use_llm: bool,
              inject_error: bool = False, stance: str = "careful") -> list[tuple[CallSpec, Result]]:
    rows = []
    for i, spec in enumerate(specs):
        pcm = audio_for(spec, dials)
        if inject_error:
            asr.transcript_of = corrupt(spec.text, seed=i)
        result = asr.run(pcm, spec)
        asr.transcript_of = ""
        if use_llm:
            from .agent import respond

            value, confirmed, reply = respond(result.transcript, spec.entity_type, stance)
            result.agent_entity, result.confirmed, result.agent_text = value, confirmed, reply
        rows.append((spec, result))
    return rows


def write(rows: list[tuple[CallSpec, Result]], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(to_jsonl(s, r) for s, r in rows))
    return score.aggregate(rows)


def table(title: str, agg: dict) -> str:
    keep = {k: v for k, v in agg.items() if v is not None}
    w = max(len(k) for k in keep)
    body = "\n".join(f"  {k:<{w}}  {v}" for k, v in keep.items())
    return f"\n{title}\n{'-' * len(title)}\n{body}\n"


def load_env(path: Path = ROOT / ".env") -> None:
    """Keys live in .env next to the repo. No dependency needed for four lines."""
    import os

    for line in path.read_text().splitlines() if path.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(prog="asli")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("run", "sweep", "demo"):
        s = sub.add_parser(name)
        s.add_argument("--suite", default="inepa", choices=["inepa", "pir", "sfr"])
        s.add_argument("--llm", action="store_true", help="run the reference agent (needs Azure keys)")
        s.add_argument("--telephony", action="store_true", help="8kHz G.711 mu-law round trip")
        s.add_argument("--snr", type=float, default=None)
        s.add_argument("--noise", default="pink", choices=["pink", "babble"])
        s.add_argument("--pause-ms", type=int, default=700)
        s.add_argument("--frames", type=int, default=18, help="negative_frames_count")
        s.add_argument("--stance", default="careful", choices=["careful", "eager"],
                       help="reference-agent stance; SFR should separate the two")
        s.add_argument("--out", default=None)

    a = p.parse_args(argv)
    dials = {k: v for k, v in (("telephony", a.telephony), ("snr_db", a.snr)) if v}
    if a.snr is not None:
        dials["noise"] = a.noise
    base = load_entities()
    rate = 16000

    if a.cmd == "demo":
        out = ROOT / "demo"
        (out / "wav").mkdir(parents=True, exist_ok=True)
        specs = build_pir(base, a.pause_ms)[:3]
        for spec in specs:
            clean = synth.render(spec)
            synth.write_wav(out / "wav" / f"{spec.id}-clean.wav", clean)
            for label, d in (("telephony", {"telephony": True}),
                             ("noisy-snr10", {"snr_db": 10, "noise": "babble"}),
                             ("worst", {"snr_db": 5, "noise": "babble", "telephony": True,
                                        "packet_loss_pct": 3})):
                synth.write_wav(out / "wav" / f"{spec.id}-{label}.wav",
                                degrade.apply(clean, d, speech_for_babble=clean))
            print(f"  {spec.id}: true_end={spec.true_end_ms}ms  pauses={spec.internal_pauses}")
        print(f"\nwrote {len(list((out / 'wav').glob('*.wav')))} files to demo/wav/")
        return 0

    if a.cmd == "sweep":
        specs = build_pir(base, a.pause_ms)
        audio = [(s, audio_for(s, dials)) for s in specs]
        print(f"\nPIR vs negative_frames_count  (pause={a.pause_ms}ms, "
              f"{'8k telephony' if a.telephony else '16k clean'}, n={len(specs)})\n")
        print(f"  {'frames':>6}  {'silence_ms':>10}  {'PIR':>6}  {'in_pause':>9}  {'median_ms_early':>15}")
        curve = []
        for frames in (4, 8, 12, 16, 18, 24, 32, 48):
            asr = MockASR(negative_frames_count=frames, rate=rate)
            rows = [(s, asr.run(pcm, s)) for s, pcm in audio]
            agg = score.aggregate(rows)
            ms = round(frames * 512 * 1000 / rate)
            print(f"  {frames:>6}  {ms:>10}  {agg['pir']:>6}  {agg['pir_injected']:>9}  "
                  f"{agg['median_ms_early'] if agg['median_ms_early'] is not None else '-':>15}")
            curve.append({"frames": frames, "silence_ms": ms, **agg})
        dest = Path(a.out or ROOT / "results" / "pir_sweep.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(curve, indent=2))
        print(f"\nwrote {dest}")
        return 0

    specs = build_pir(base, a.pause_ms) if a.suite == "pir" else base
    asr = MockASR(negative_frames_count=a.frames, rate=rate)
    rows = run_suite(specs, asr, dials, a.llm, inject_error=(a.suite == "sfr"), stance=a.stance)
    agg = write(rows, Path(a.out or ROOT / "results" / f"{a.suite}.jsonl"))
    print(table(f"{a.suite} (n={agg['n']}, stance={a.stance}, dials={dials or 'clean'})", agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())

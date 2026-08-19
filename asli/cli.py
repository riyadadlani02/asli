"""asli — run a suite, write JSONL, print the table.

    asli demo                     # synthesise, degrade, and write demo/ recordings
    asli run  --suite inepa --llm # parsing accuracy on a clean transcript
    asli sweep --suite pir        # the endpointing curve: PIR vs negative_frames_count
    asli run  --suite sfr --llm   # silent-failure rate under injected ASR error
    asli text --stance eager      # the same SFR in a text pipeline, no audio at all
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import degrade, fit as fitmod, score, synth
from .drive import MockASR, SarvamWS
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


FIT_PATH = ROOT / "results" / "pause_fit.json"


def build_pir(base: list[CallSpec], pause_ms: int, fitted: bool = False) -> list[CallSpec]:
    """Insert one filler + pause immediately before the entity in each utterance.

    Placement is deliberate: immediately *before* the entity is where a real speaker
    hesitates, and it is the placement that turned out to matter.

    With `fitted`, pause lengths are drawn from a distribution fitted to real speech
    (`asli fit`) instead of a constant, which is what lets PIR be read as a rate rather
    than as "at 700ms".
    """
    draws = None
    if fitted:
        if not FIT_PATH.exists():
            raise SystemExit(f"no fit at {FIT_PATH} — run `asli fit --corpus DIR` first")
        f = json.loads(FIT_PATH.read_text())
        draws = fitmod.sample_pauses(len(base), f["mu"], f["sigma"], seed=0)

    out = []
    for i, spec in enumerate(base):
        s = copy.deepcopy(spec)
        filler = FILLERS[i % len(FILLERS)]
        gap = draws[i] if draws else pause_ms
        head = s.segments[:-1]
        s.segments = [*head, Segment(filler, pause_after_ms=gap, kind="filler"), s.segments[-1]]
        s.id = f"{spec.id}-pir-{gap}"
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


def call(adapter, pcm: np.ndarray, spec: CallSpec) -> Result:
    """One adapter turn. SarvamWS is async (a live socket), MockASR is not."""
    out = adapter.run(pcm, spec)
    return asyncio.run(out) if inspect.isawaitable(out) else out


def make_adapter(name: str, *, gate: int, lang: str, rate: int, mode: str = "verbatim"):
    if name == "mock":
        return MockASR(negative_frames_count=gate, rate=rate)
    # verbatim keeps the spoken form ("triple one"), so INEPA measures the agent's
    # parsing. translit/codemix apply the vendor's own numeral normaliser first,
    # which is a different measurement — see the README.
    return SarvamWS(language_code=lang, rate=rate, silence_duration_ms=gate, mode=mode)


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
        result = call(asr, pcm, spec)
        if hasattr(asr, "transcript_of"):
            asr.transcript_of = ""
        if result.error:
            print(f"  ! {spec.id}: {result.error}", file=sys.stderr)
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

    for name in ("run", "sweep", "demo", "check", "fit", "text"):
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
        s.add_argument("--agent", default="mock", choices=["mock", "sarvam"],
                       help="system under test; 'sarvam' needs SARVAM_API_KEY")
        s.add_argument("--lang", default="hi-IN")
        s.add_argument("--mode", default="verbatim",
                       choices=["verbatim", "transcribe", "translit", "codemix"],
                       help="Sarvam transcription mode; verbatim preserves spoken numerals")
        s.add_argument("--out", default=None)
        s.add_argument("--corpus", default=None, help="directory of real-speech wavs, for `fit`")
        s.add_argument("--fitted", action="store_true",
                       help="draw pause lengths from the fitted distribution")
        s.add_argument("--dry-run", action="store_true",
                       help="`text`: print the built corpus, make no API calls")

    a = p.parse_args(argv)
    dials = {k: v for k, v in (("telephony", a.telephony), ("snr_db", a.snr)) if v}
    if a.snr is not None:
        dials["noise"] = a.noise
    base = load_entities()
    rate = 16000

    if a.cmd == "fit":
        if not a.corpus:
            print("usage: asli fit --corpus DIR\n\n"
                  "DIR should hold unscripted telephone speech — IndicVoices or Voice of\n"
                  "India. IndicVoices is gated: accept the terms at\n"
                  "  https://huggingface.co/datasets/ai4bharat/IndicVoices\n"
                  "then download the hindi/ split and point --corpus at the wavs.",
                  file=sys.stderr)
            return 2
        # anything ffmpeg reads — corpora ship as mp3 or flac as often as wav
        wavs = sorted(f for f in Path(a.corpus).rglob("*")
                      if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"))
        if not wavs:
            print(f"no audio files under {a.corpus}", file=sys.stderr)
            return 2
        f = fitmod.fit_corpus(wavs, out=FIT_PATH)
        print(table(f"pause distribution ({f['n_files']} files, {f['n_pauses']} pauses)", f))
        print(f"  wrote {FIT_PATH}\n  now: asli sweep --suite pir --agent sarvam --fitted")
        return 0

    if a.cmd == "text":
        import os

        from .text import run as textrun

        items = textrun.build(textrun.load())
        if a.dry_run:
            for it in items:
                print(f"\n--- {it.id}  ({it.corruption}, detectable={it.detectable})")
                for r in it.records:
                    print(f"      {r}")
                print(f"    Q: {it.question}\n    truth: {it.canonical}")
            print(f"\n{len(items)} items built. No API calls made.")
            return 0
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            print("AZURE_OPENAI_API_KEY is not set — `asli text` is agent-only, there is\n"
                  "no recogniser to run without it. Add keys to .env, or inspect the\n"
                  "corpus offline with:  asli text --dry-run", file=sys.stderr)
            return 2
        rows = textrun.run(items, a.stance)
        dest = Path(a.out or ROOT / "results" / f"sfr_text_{a.stance}.jsonl")
        agg = textrun.write(rows, dest)
        by_class = agg.pop("by_class")
        print(table(f"sfr text (stance={a.stance}, n={agg['n']})", agg))
        print("  acted blind, by corruption class")
        for k, v in by_class.items():
            print(f"    {k:<14}  {v['acted_blind']:<8}  n={v['n']}")
        print(f"\n  wrote {dest}")
        return 0

    if a.cmd == "check":
        import os

        if "SARVAM_API_KEY" not in os.environ:
            print("SARVAM_API_KEY is not set. Add it to .env:\n"
                  "  echo 'SARVAM_API_KEY=...' >> .env", file=sys.stderr)
            return 2
        spec = build_pir(base, a.pause_ms, a.fitted)[0]
        pcm = audio_for(spec, dials)
        print(f"  sending {len(pcm) / rate:.1f}s of audio, paced in real time...")
        asr = make_adapter("sarvam", gate=a.frames if a.frames != 18 else 500,
                           lang=a.lang, rate=rate, mode=a.mode)
        res = call(asr, pcm, spec)
        if res.error:
            print(f"\n  FAILED: {res.error}", file=sys.stderr)
            return 1
        print(f"\n  transcript: {res.transcript!r}")
        print(f"  events:     {[(e.kind, e.t_ms) for e in res.events]}")
        print(f"  true_end:   {spec.true_end_ms}ms   injected pause: {spec.internal_pauses}")
        print(f"\n  PIR verdict: {score.pir(spec, res)}")
        return 0

    if a.cmd == "demo":
        out = ROOT / "demo"
        (out / "wav").mkdir(parents=True, exist_ok=True)
        specs = build_pir(base, a.pause_ms, a.fitted)[:3]
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
        specs = build_pir(base, a.pause_ms, a.fitted)
        audio = [(s, audio_for(s, dials)) for s in specs]
        # the mock's gate is a frame count; the real endpoint takes milliseconds
        gates = ((4, 8, 12, 16, 18, 24, 32, 48) if a.agent == "mock"
                 else (100, 200, 300, 400, 500, 700, 900, 1200))
        axis = "negative_frames_count" if a.agent == "mock" else "silence_duration_ms"
        print(f"\nPIR vs {axis}  (agent={a.agent}, pause={a.pause_ms}ms, "
              f"{'8k telephony' if a.telephony else '16k clean'}, n={len(specs)})\n")
        print(f"  {'gate':>6}  {'silence_ms':>10}  {'PIR':>6}  {'in_pause':>9}  {'median_ms_early':>15}")
        curve = []
        for frames in gates:
            asr = make_adapter(a.agent, gate=frames, lang=a.lang, rate=rate, mode=a.mode)
            rows = [(s, call(asr, pcm, s)) for s, pcm in audio]
            agg = score.aggregate(rows)
            ms = round(frames * 512 * 1000 / rate) if a.agent == "mock" else frames
            print(f"  {frames:>6}  {ms:>10}  {agg['pir']:>6}  {agg['pir_injected']:>9}  "
                  f"{agg['median_ms_early'] if agg['median_ms_early'] is not None else '-':>15}")
            curve.append({"frames": frames, "silence_ms": ms, **agg})
        dest = Path(a.out or ROOT / "results" / "pir_sweep.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(curve, indent=2))
        print(f"\nwrote {dest}")
        return 0

    specs = build_pir(base, a.pause_ms, a.fitted) if a.suite == "pir" else base
    asr = make_adapter(a.agent, gate=a.frames, lang=a.lang, rate=rate, mode=a.mode)
    rows = run_suite(specs, asr, dials, a.llm, inject_error=(a.suite == "sfr"), stance=a.stance)
    agg = write(rows, Path(a.out or ROOT / "results" / f"{a.suite}.jsonl"))
    print(table(f"{a.suite} (agent={a.agent}, n={agg['n']}, stance={a.stance}, "
                f"dials={dials or 'clean'})", agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())

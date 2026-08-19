"""The vendor matrix: one utterance, one hesitation, every lane at the same gate.

Written as a script because the table was ad hoc for a while and the page ended up
rendering two of three rows. Regenerate with:

    uv run python experiments/vendor_matrix.py

`end_source` is the column that stops this being a false comparison. Sarvam, Deepgram
and OpenAI report their own turn-end event, so their timestamp is the decision itself.
Gemini Live emits no such event: its timestamp is inferred from the first chunk of the
model's reply and therefore includes response latency. Same knob, same audio, weaker
evidence — and the row says so rather than sitting silently beside the others.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from asli import score
from asli.cli import audio_for, build_pir, load_entities, load_env, make_adapter

load_env()
GATE = 500
LANES = [("sarvam", "silence 500ms", "vad"),
         ("deepgram", "silence 500ms", "vad"),
         ("openai", "server_vad 500ms", "vad"),
         ("gemini", "activity detection 500ms", "inferred: first model reply")]


def main() -> int:
    spec = build_pir(load_entities(), 700)[0]
    pcm = audio_for(spec, {})
    truth = score.normalise(spec.entity_type, spec.canonical)
    print(f"{spec.id}: true_end={spec.true_end_ms}ms  pause={spec.internal_pauses}\n")

    rows = []
    for name, detection, end_source in LANES:
        res = asyncio.run(make_adapter(name, gate=GATE, lang="hi-IN", rate=16000).run(pcm, spec))
        ends = [e.t_ms for e in res.events if e.kind == "speech_end"]
        finals = [e.text for e in res.events if e.kind == "transcript" and e.text]
        first = finals[0] if finals else res.transcript
        got = score.normalise(spec.entity_type, first)
        rows.append({"adapter": name, "detection": detection, "end_source": end_source,
                     "turns": len(finals), "ends": ends, "first_turn": first,
                     "number_in_first_turn": bool(got) and truth in got,
                     "error": res.error})
        r = rows[-1]
        print(f"  {name:<9} end={ends[:1]} turns={r['turns']} "
              f"number={'YES' if r['number_in_first_turn'] else 'no '} "
              f"{r['error'][:50] or first[:44]!r}")

    out = Path(__file__).parent.parent / "results/vendor_matrix.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

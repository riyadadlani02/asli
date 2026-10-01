"""The sample kit: voice recordings with a known truth, for checking the bench before
testing it with a real voice.

Built from the three ElevenLabs renders already in `demo/wav` by re-splicing the
hesitation. The speech either side is untouched and the pause between is replaced by
exactly N ms of digital silence, located by sample index rather than estimated — so every
sample's pause window and end of speech are known, not measured, and the bench's VAD
measurement can be checked against them.

The voice is synthetic and the pause is digital silence. That is what makes the truth
exact, and it is also the objection to the authored lane: a real line keeps its noise
floor through a pause. The `-phone` and `-noisy` entries reuse the degraded renders, and
the recording guide in samples/README.md is the way past both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spec import SAMPLE_RATE
from .synth import read_wav, write_wav

ROOT = Path(__file__).parent.parent
OUT = ROOT / "samples"
VOICE = "ElevenLabs eleven_multilingual_v2 (synthetic), re-spliced from demo/wav"

BASE = [
    {"id": "mobile", "src": "dig-01", "label": "Mobile number", "filler": "matlab",
     "head": "Mera mobile number hai, matlab", "tail": "nine eight double seven, triple one",
     "entity_type": "digits", "expected": "9877111"},
    {"id": "account", "src": "dig-02", "label": "Account digits", "filler": "woh kya bolte hain",
     "head": "Account ke last five digits, woh kya bolte hain", "tail": "char zero double five six",
     "entity_type": "digits", "expected": "40556"},
    {"id": "reference", "src": "dig-03", "label": "Reference number", "filler": "haan toh",
     "head": "Haan toh", "tail": "the reference is eight, double zero, nine",
     "entity_type": "digits", "expected": "8009",
     "note": "the filler and pause come at the START, before the whole sentence"},
]
HESITATIONS = (150, 400, 700, 1000)   # 150 is a fluent gap: the control
LEAD_MS, TAIL_MS = 300, 800           # a file people upload has margins; the bench adds
                                      # its own open-line silence after this


def _ms(n: int) -> int:
    return int(round(n * 1000 / SAMPLE_RATE))


def _zero_run(pcm: np.ndarray, min_ms: int = 300) -> tuple[int, int]:
    """(start, end) sample index of the spliced hesitation: the longest exact-zero run."""
    z = np.concatenate([[0], (pcm == 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(z))
    runs = list(zip(edges[::2], edges[1::2]))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < SAMPLE_RATE * min_ms // 1000:
        raise ValueError("no spliced pause found — is this a clean render?")
    return int(a), int(b)


def build(out: Path = OUT) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in BASE:
        clean, rate = read_wav(ROOT / "demo" / "wav" / f"{b['src']}-pir-700-clean.wav")
        assert rate == SAMPLE_RATE
        p0, p1 = _zero_run(clean)
        head, tail = clean[:p0], clean[p1:]
        said = f"{b['head']} … {b['tail']}"
        for h in HESITATIONS:
            lead = np.zeros(SAMPLE_RATE * LEAD_MS // 1000, np.int16)
            gap = np.zeros(SAMPLE_RATE * h // 1000, np.int16)
            end = np.zeros(SAMPLE_RATE * TAIL_MS // 1000, np.int16)
            pcm = np.concatenate([lead, head, gap, tail, end])
            sid = f"{b['id']}-{h}ms"
            write_wav(out / f"{sid}.wav", pcm)
            start = len(lead) + len(head)
            rows.append({
                "id": sid, "file": f"samples/{sid}.wav",
                "title": (f"{b['label']} · fluent, {h} ms gap (control)" if h < 200 else
                          f"{b['label']} · {h} ms pause after “{b['filler']}”"),
                "said": said, "entity_type": b["entity_type"], "expected": b["expected"],
                "hesitation_ms": h, "pause_ms": [_ms(start), _ms(start + len(gap))],
                "speech_end_ms": _ms(start + len(gap) + len(tail)),
                "line": "clean", "voice": VOICE, **({"note": b["note"]} if "note" in b else {}),
            })
        # the degraded renders keep the original timeline: no margins, 700 ms pause
        for line, suffix in (("phone line (8 kHz mu-law)", "telephony"),
                             ("babble noise, 10 dB SNR", "noisy-snr10")):
            rows.append({
                "id": f"{b['id']}-700ms-{suffix.split('-')[0]}",
                "file": f"demo/wav/{b['src']}-pir-700-{suffix}.wav",
                "title": f"{b['label']} · 700 ms pause · {line}",
                "said": said, "entity_type": b["entity_type"], "expected": b["expected"],
                "hesitation_ms": 700, "pause_ms": [_ms(p0), _ms(p1)],
                "speech_end_ms": _ms(len(clean)), "line": line, "voice": VOICE,
                **({"note": b["note"]} if "note" in b else {}),
            })
    _write_manifest(out / "manifest.yaml", rows)
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    import yaml

    head = ("# The sample kit. Rebuild with `asli samples`; do not edit by hand.\n"
            "# pause_ms and speech_end_ms are exact (the pause was spliced in), in ms from the\n"
            "# start of the file. Add your own recordings in samples/mine.yaml (same fields;\n"
            "# hesitation_ms, pause_ms and speech_end_ms optional) — see samples/README.md.\n")
    path.write_text(head + yaml.safe_dump(rows, sort_keys=False, allow_unicode=True, width=100))

"""Regenerate site/index.html from results/ and demo/wav/.

The site embeds its own evidence — audio, waveforms, every measurement — so it is one
file with no build step and no external requests at view time. Run this after a new
harness run to refresh it.

Output is deliberately ASCII-only: Devanagari goes out as HTML entities outside
<script> and \\u escapes inside it, so the page renders correctly even on a static
host that serves .html without a charset.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from asli.synth import read_wav

ROOT = Path(__file__).parent
HERO_ID = "dig-01-pir-700"
VARIANTS = [("clean", "clean line, 16 kHz"),
            ("telephony", "8 kHz G.711 telephony"),
            ("noisy-snr10", "call-centre babble, 10 dB"),
            ("worst", "8 kHz + babble 5 dB + 3% loss")]
SAID = [
    {"roman": "Mera mobile number hai", "deva": "मेरा मोबाइल नंबर है", "t0": 0, "t1": 1089},
    {"roman": "matlab", "deva": "मतलब", "t0": 1089, "t1": 1716, "fill": True},
    {"roman": "nine eight double seven, triple one", "deva": "नाइन एट डबल सेवन ट्रिपल वन", "t0": 2416, "t1": 4354, "ent": True},
]


def envelope(pcm: np.ndarray, n: int) -> list[float]:
    step = max(1, len(pcm) // n)
    env = [float(np.abs(pcm[i * step:(i + 1) * step]).max()) / 32768 for i in range(n)]
    peak = max(env) or 1.0
    return [round(v / peak, 3) for v in env]


def mp3(path: Path) -> str:
    """Down to 32kbps mono — small enough to inline four clips, good enough to hear the cut."""
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1",
         "-ar", "22050", "-b:a", "32k", "-codec:a", "libmp3lame", "-f", "mp3", "-"],
        capture_output=True, check=True).stdout
    return base64.b64encode(out).decode()


def collect() -> dict:
    d: dict = {"said": SAID}
    d["sweep"] = json.loads((ROOT / "results/pir_sweep_sarvam.json").read_text())
    d["hero"] = {k: v for k, v in json.loads((ROOT / "results/hero_wave.json").read_text()).items()
                 if k != "env"}

    rows = json.loads((ROOT / "results/mode_placement.json").read_text())
    d["modes"] = {r["mode"]: {"text": r["transcript"], "got": r["extracted"], "ok": r["survived"]}
                  for r in rows
                  if r["id"] == "dig-01" and r["run"] == 0 and r["placement"].startswith("hesitation")}

    d["audio"] = []
    for key, label in VARIANTS:
        wav = ROOT / f"demo/wav/{HERO_ID}-{key}.wav"
        pcm, _ = read_wav(wav)
        d["audio"].append({"key": key, "label": label,
                           "env": envelope(pcm, 300), "mp3": mp3(wav)})
    return d


def to_ascii(html: str) -> str:
    """Entities outside <script>, \\u escapes inside — HTML entities are not decoded
    inside a script element, so the two halves need different treatment."""
    esc_html = lambda t: "".join(c if ord(c) < 128 else f"&#x{ord(c):04X};" for c in t)
    esc_js = lambda t: "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in t)
    parts = re.split(r"(<script>.*?</script>)", html, flags=re.S)
    return "".join(("<script>" + esc_js(p[8:-9]) + "</script>") if p.startswith("<script>")
                   else esc_html(p) for p in parts)


def main() -> None:
    data = collect()
    tpl = (ROOT / "site/template.html").read_text()
    html = to_ascii(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                       separators=(",", ":"))))
    assert all(ord(c) < 128 for c in html), "output must be ascii"
    (ROOT / "site/index.html").write_text(html, encoding="ascii")
    print(f"site/index.html  {len(html)/1024:.0f} KB  "
          f"({len(data['audio'])} clips, {len(data['sweep'])} sweep points)")


if __name__ == "__main__":
    main()

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

from asli import fit as fitmod
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


def _score_sample(sample: dict) -> dict:
    """Score the sample here, not in the browser.

    The page would otherwise need its own copy of the entity parsers, and a second
    implementation is a second thing to be wrong — it already produced 540556 for an
    account number and 2 for a lakh amount. Verdicts come from asli.score, the same
    functions the harness is tested on, and the page just renders them.
    """
    from asli import score

    for ex in sample["exchanges"]:
        # authored in results/sample_call.json — never guessed from magnitude, which
        # mistook a 7-digit phone number for a rupee amount
        et = ex["entity_type"]
        truth = score.normalise(et, ex["truth"])
        first = ex["turns"][0]["text"] if ex["turns"] else ""
        whole = " ".join(t["text"] for t in ex["turns"])
        ex["value_first"] = score.normalise(et, first)
        ex["value_whole"] = score.normalise(et, whole)
        # containment, not equality: non-entity words legitimately contribute digits
        # ("last five digits" yields a 5 that belongs to no account number)
        ex["ok_first"] = bool(ex["value_first"]) and truth in ex["value_first"]
        ex["ok_whole"] = bool(ex["value_whole"]) and truth in ex["value_whole"]
    return sample


def _text_lane() -> dict:
    """Aggregates for the text instantiation, recomputed from the stored rows so the
    page can never drift from what was actually run."""
    from asli.spec import Result
    from asli.text import run as textrun

    out = {}
    for stance in ("careful", "eager"):
        path = ROOT / f"results/sfr_text_{stance}.jsonl"
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                rows.append((textrun.Item(**d["item"]), Result(**d["result"])))
        out[stance] = textrun.aggregate(rows)
    return out


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%".replace(".0%", "%")


def subs() -> dict[str, str]:
    """Prose numbers, taken from the stored results rather than typed into the page.

    The page states figures in sentences as well as in charts, and a sentence drifts
    from the data as quietly as a chart does. These are the ones a reader would quote.
    """
    fit = json.loads((ROOT / "results/pause_fit.json").read_text())
    conv = json.loads((ROOT / "results/conv.json").read_text())
    rec = fitmod.recommended_gate(fit)
    advice = "".join(
        f"<tr><td>{r['gate_ms']}{' (default)' if r['gate_ms'] == 500 else ''}</td>"
        f"<td>{pct(r['pauses_tripped'])}</td><td>{pct(r['calls_affected'])}</td>"
        f"<td>{r['added_latency_ms']:+d} ms</td></tr>"
        for r in fitmod.gate_advice(fit))

    d = {
        "__T_CALLS500__": pct(fit["calls_exceed"]["500"]),
        "__T_PAUSES500__": pct(fit["exceed"]["500"]),
        "__T_ADVICE_ROWS__": advice,
        "__T_REC__": (f"Within +{rec['budget_ms']} ms of added turn-end latency, "
                      f"silence_duration_ms = {rec['gate_ms']} is the setting: callers "
                      f"carrying a long enough pause go from {pct(rec['calls_affected_at_default'])} "
                      f"to {pct(rec['calls_affected'])} — "
                      f"{pct(rec['removes_share_of_affected_calls'])} of them removed. "
                      f"That budget is a product decision, so it is an input here, "
                      f"not an assumption."),
        "__T_CONV_PIR__": f"{conv['n_cut']}/{conv['n']}",
        "__T_FIRSTTURN__": pct(conv["entity_first_turn"]),
        "__T_SESSION__": pct(conv["entity_full_session"]),
        "__T_RCR__": pct(conv["rcr"]),
        "__T_BUDGET__": f"{conv['median_silence_budget_ms']} ms",
        "__T_ABSTAIN__": (
            f"{conv['entity_session_abstained']} of the {conv['n']} sessions are scored as "
            f"<em>abstained</em> rather than failed: they are spoken dates in Devanagari "
            f"digit words, which this scorer cannot parse. Calling that a vendor failure "
            f"would be inventing a result, so the session row is a rate over what is "
            f"readable." if conv.get("entity_session_abstained") else ""),
    }
    d.update(_real_lane())
    return d


def _real_lane() -> dict[str, str]:
    """The real-caller lane and its one-variable control.

    The headline here is the *in-pause* figure, not raw PIR. A spontaneous ten-second
    voice message should be split into several turns, so "a turn ended before the
    recording did" is nearly free on this corpus; "a turn ended inside a hesitation of
    500ms or more" is the thing the synthetic lane claims.
    """
    keys = ("__T_REAL_N__", "__T_REAL_ROWS__", "__T_REAL_VERDICT__", "__T_SPREAD__",
            "__T_FLOOR__", "__T_REALHEAD__")
    paths = {"as recorded": ROOT / "results/real_pir.json",
             "hesitation replaced by digital silence": ROOT / "results/real_pir_silenced.json"}
    got = {k: json.loads(v.read_text()) for k, v in paths.items() if v.exists()}
    base = got.get("as recorded")
    if not base or not base.get("gates"):
        return {k: "-" for k in keys}

    fit = json.loads((ROOT / "results/pause_fit.json").read_text())
    note = {"as recorded": "nothing touched",
            "hesitation replaced by digital silence": "one variable: the pause floor"}
    rows = "".join(
        f"<tr><td>{k}</td><td><b>{pct(g['gates'][0]['in_pause'])}</b></td>"
        f"<td>{pct(g['gates'][0]['pir'])}</td><td>{note[k]}</td></tr>"
        for k, g in got.items() if g.get("gates"))

    n = base["n"]
    real = base["gates"][0]["in_pause"]
    ctl = got.get("hesitation replaced by digital silence", {}).get("gates")
    field = real * fit["calls_exceed"]["500"]
    verdict = (f"<b>{round(real * n)} of the {n} real callers had a turn ended inside their own "
               f"hesitation</b>, at the documented default, with no synthesis anywhere in the "
               f"path. These recordings were selected for carrying a pause that long, so that "
               f"rate is conditional on it; composed with the {pct(fit['calls_exceed']['500'])} "
               f"of recordings that do, it puts roughly <b>{pct(field)} of this corpus's callers"
               f"</b> in the same position — about 1 in "
               f"{round(1 / field) if field else '-'}.")
    if ctl:
        gain = ctl[0]["in_pause"] - real
        verdict += (f" The control says how much of the rest is the line rather than the "
                    f"endpointer: replacing the hesitation with digital silence and changing "
                    f"nothing else takes it to {pct(ctl[0]['in_pause'])} — "
                    f"{'+' if gain >= 0 else ''}{round(gain * 100)} points. At a median pause "
                    f"floor of {base.get('pause_floor_db_median')} dB the live line's own noise "
                    f"protects some callers, but it is a secondary effect, not the explanation. "
                    f"The exposed caller is the one on the quiet handset, and noise suppression "
                    f"placed ahead of the VAD moves callers toward the cut, not away from it.")
    verdict += (" The wider column is every early turn end, including the ordinary splitting of "
                "a long monologue, and it is reported only for completeness: on spontaneous "
                "voice messages that number is close to free.")
    return {"__T_REAL_N__": str(n),
            "__T_REAL_ROWS__": rows,
            "__T_REAL_VERDICT__": verdict,
            "__T_REALHEAD__": f"{round(real * n)} of {n} had the turn ended inside the "
                              f"hesitation",
            "__T_SPREAD__": f"{base.get('end_spread_median_ms', '-')} ms median, "
                            f"{base.get('end_spread_max_ms', '-')} ms worst",
            "__T_FLOOR__": f"{base.get('pause_floor_db_median', '-')} dB"}


def collect() -> dict:
    d: dict = {"said": SAID}
    d["sweep"] = json.loads((ROOT / "results/pir_sweep_sarvam.json").read_text())
    d["fit"] = json.loads((ROOT / "results/pause_fit.json").read_text())
    d["text"] = _text_lane()
    d["sample"] = _score_sample(json.loads((ROOT / "results/sample_call.json").read_text()))
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
    for token, value in subs().items():
        tpl = tpl.replace(token, value)
    assert "__T_" not in tpl, "a prose token was left unfilled"
    html = to_ascii(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                       separators=(",", ":"))))
    assert all(ord(c) < 128 for c in html), "output must be ascii"
    (ROOT / "site/index.html").write_text(html, encoding="ascii")
    # GitHub Pages serves from docs/ on the branch — Actions is not available on this
    # account, so the published copy has to be committed rather than built in CI.
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(html, encoding="ascii")
    (docs / ".nojekyll").touch()
    print(f"site/index.html + docs/index.html  {len(html)/1024:.0f} KB  "
          f"({len(data['audio'])} clips, {len(data['sweep'])} sweep points)")


if __name__ == "__main__":
    main()

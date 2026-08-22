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
FIXED_MODE_MATRIX_PATH = ROOT / "results" / "mode_matrix_fixed.json"
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


# Model names are labels, not measurements — the matrix stores the adapter key and the
# turn-detection setting, which are what the rows are compared on.
LANES = {"sarvam": "Sarvam &middot; saaras:v3-realtime",
         "deepgram": "Deepgram &middot; nova-2",
         "openai": "OpenAI &middot; gpt-4o-transcribe",
         "gemini": "Gemini &middot; 3.1-flash-live"}


def _vendor_lane() -> list[dict]:
    """Every reproducible row, so a lane cannot be quietly dropped from the page.

    The page carried two of the three for a while because the comparison lived in prose.
    Reading the file is what stops that recurring.
    """
    rows = json.loads((ROOT / "results/vendor_matrix.json").read_text())
    # No derived verdict here. `turns > 1` would have called the OpenAI lane a hold: it
    # ended the turn at 2500 ms with the number still unsaid, and only one final was
    # captured before the socket closed. The turn-end timestamp and the entity check are
    # what the matrix stored, and they are what the row states.
    return [{"label": LANES.get(r["adapter"], r["adapter"]), "detection": r["detection"],
             "end_ms": r["ends"][0] if r["ends"] else None, "first": r["first_turn"],
             "kept": r["number_in_first_turn"],
             # marks the row whose timestamp is inferred rather than reported, so the
             # column is never read as four equal measurements
             "inferred": not r.get("end_source", "vad").startswith("vad")} for r in rows]


def _semantic_lane() -> dict:
    """Experiment B2, recomputed from the stored rows.

    A failed call is not a measurement and a zero-turn row is not a split — both are
    excluded here exactly as they are in the experiment's own analysis, so the page and
    the README cannot drift apart.
    """
    rows = json.loads((ROOT / "results/verbfinal2.json").read_text())

    def cell(arm: str, mode: str) -> dict:
        sel = [r for r in rows if r["arm"] == arm and r["mode"] == mode
               and r["turns"] > 0 and not r.get("error")]
        return {"rate": round(sum(r["split"] for r in sel) / len(sel), 3) if sel else None,
                "n": len(sel)}

    arms = ["dangler", "verb-final", "filler"]
    held = [r for r in rows if r["turns"] == 0 and not r.get("error")]
    return {
        "arms": [{"arm": a, "server_vad": cell(a, "server_vad"),
                  "semantic_vad": cell(a, "semantic_vad")} for a in arms],
        "n_calls": len(rows),
        "failures": sum(1 for r in rows if r.get("error")),
        "held_out": len(held),
        "held_out_no_event": sum(1 for r in held if not r["ends"]),
        "splits": [{"id": r["id"], "arm": r["arm"], "verb": r["verb"]} for r in rows
                   if r["mode"] == "semantic_vad" and r["split"] and not r.get("error")],
    }


# What each lane has actually been measured on. Uneven on purpose: Sarvam carries the
# deep lanes because it was the first vendor, and printing the gaps is more useful than
# implying parity that does not exist.
LANE_EXTRAS = {
    "sarvam": ["the endpointing sweep, 8 gate settings",
               "20 real unscripted callers, streamed unmodified",
               "four output modes, and which keep the number",
               "the 12-session conversation lane"],
    "openai": ["semantic vs silence turn detection, 186 calls"],
    "deepgram": [],
    "gemini": [],
}
ALL_EXTRAS = [x for v in LANE_EXTRAS.values() for x in v]


def _lane_results() -> dict:
    out = {}
    for row in _vendor_lane():
        key = next(k for k, v in LANES.items() if v == row["label"])
        mine = LANE_EXTRAS[key]
        out[key] = {**row, "measured": mine,
                    "not_measured": [x for x in ALL_EXTRAS if x not in mine]}
    return out


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%".replace(".0%", "%")


def fixed_mode_matrix() -> dict[str, dict[str, list[int]]]:
    """Read the aggregate generated by the fixed-filler probe, never a typed table."""
    return json.loads(FIXED_MODE_MATRIX_PATH.read_text())


def mode_cell(matrix: dict[str, dict[str, list[int]]], mode: str, placement: str) -> str:
    survived, total = matrix[mode][placement]
    return f"{survived}/{total}"


def _openai_real_pair() -> dict[str, object]:
    """Load the paired real-caller conditions and reject mismatched evidence."""
    def condition(name: str, expected_detection: str) -> tuple[dict, dict]:
        meta = json.loads((ROOT / f"results/real_pir_openai_{name}.json").read_text())
        rows = [json.loads(line) for line in
                (ROOT / f"results/real_pir_openai_{name}.jsonl").read_text().splitlines()
                if line.strip()]
        if (meta.get("turn_detection") != expected_detection or len(rows) != meta["n"]
                or any(row["result"].get("error") for row in rows)):
            raise ValueError(f"invalid OpenAI {expected_detection} real-caller evidence")
        return meta, meta["gates"][0]

    server_meta, server = condition("server_vad", "server_vad")
    semantic_meta, semantic = condition("semantic_vad", "semantic_vad")
    if server_meta["ids"] != semantic_meta["ids"]:
        raise ValueError("OpenAI real-caller conditions do not use the same caller IDs")
    return {"n": server_meta["n"], "server": {
                "in_pause": server["in_pause"], "early": server["pir"]},
            "semantic": {"in_pause": semantic["in_pause"], "early": semantic["pir"]}}


def subs() -> dict[str, str]:
    """Prose numbers, taken from the stored results rather than typed into the page.

    The page states figures in sentences as well as in charts, and a sentence drifts
    from the data as quietly as a chart does. These are the ones a reader would quote.
    """
    fit = json.loads((ROOT / "results/pause_fit.json").read_text())
    conv = json.loads((ROOT / "results/conv.json").read_text())
    eng = json.loads((ROOT / "results/pause_fit_english.json").read_text())
    sem = _semantic_lane()
    db = _diarbench()
    an = db["annotation"]
    pol = _policy_lane()
    lat = json.loads((ROOT / "results/policy_latency.json").read_text())
    modes = fixed_mode_matrix()
    real_pair = _openai_real_pair()
    uni = _union()
    sem_sv = sum(a["server_vad"]["n"] for a in sem["arms"])
    sem_worst = max(a["semantic_vad"]["rate"] for a in sem["arms"])
    sem_worst = f"{sem_worst * 100:.0f}%"
    def pair_cell(condition: str, metric: str) -> str:
        rate = real_pair[condition][metric]
        return f"{round(rate * real_pair['n'])}/{real_pair['n']} ({pct(rate)})"
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
        "__T_ENG__": (
            f"A first attempt says <b>{pct(eng['exceed']['500'])} of English gaps exceed "
            f"500&nbsp;ms</b> against {pct(fit['exceed']['500'])} of Hindi ones &mdash; which "
            f"would mean this is not Indic at all. <b>The comparison does not hold.</b> The "
            f"English figure is {eng['source']}: one headset per speaker in a meeting, so each "
            f"channel mixes that speaker's own hesitations with every silence while other "
            f"people are talking. {pct(eng['exceed']['1200'])} of the English gaps exceed "
            f"1.2&nbsp;s against {pct(fit['exceed']['1200'])} of the Hindi ones &mdash; that is "
            f"the other participants, not a speaking style. Settling it needs single-speaker "
            f"spontaneous English telephony, or diarisation applied to AMI first. Until then "
            f"the rate above is measured on Hindi and claimed only for Hindi."),
        "__T_UNION_N__": str(uni["semantic"]),
        "__T_UNION_K__": str(uni["recovered"]),
        "__T_UNION_R__": str(uni["union"]),
        "__T_UNION_TOT__": str(uni["n"]),
        "__T_UNION_P__": f"not significant (Fisher p = {uni['p']})",
        "__T_BUDGET_NOTE__": (
            f"<b>Latency is the constraint, not the variable.</b> At a budget of "
            f"{BUDGET_MS}&nbsp;ms per turn the gate cannot move at all &mdash; its smallest "
            f"useful step is +200&nbsp;ms &mdash; so the rule is not the weaker option here, "
            f"it is the only one that fits. Raise the budget past 200&nbsp;ms and the honest "
            f"answer flips: a 900&nbsp;ms gate leaves 1.3% of callers cut against 42.6% "
            f"today, which is more than this rule achieves. <b>Which column matters is a "
            f"product decision, and it is an input here rather than an assumption.</b> The "
            f"rule's own effectiveness on real callers is "
            f"<a href=\"https://github.com/riyadadlani02/asli/blob/main/docs/prereg-real-callers.md\">"
            f"pre-registered and not yet run</a>."),
        "__T_LAT__": (
            f"A false hold costs one more endpointing cycle: the turn had already ended "
            f"after {lat['gate_ms']}&nbsp;ms of silence, and holding it open means waiting "
            f"that out again. So the rule's expected cost is its false-positive rate times "
            f"{lat['gate_ms']}&nbsp;ms, and <b>it would have to wrongly hold "
            f"{lat['breakeven_fp']:.0%} of finished turns before it cost more than simply "
            f"moving the gate to 700&nbsp;ms</b>. Measured: {lat['false_holds']} false holds "
            f"in {lat['n_complete']} utterances whose true ending we built, so at 95% "
            f"confidence no worse than {lat['fp_upper95']:.1%} &mdash; "
            f"{lat['policy_expected_ms']:.0f}&nbsp;ms a turn against 200."),
        "__T_POL_NOTE_SHORT__": (
            f"The rule reads the turn's text, so it inherits what the recogniser did to it: "
            f"the filler row is {pol['arms'][1]['rescued']}/31 because the recogniser "
            f"returned <i>matlab</i> that often. One source of truth for the word lists now "
            f"lives in <code>asli.score</code> &mdash; they had been copied into three files "
            f"and drifted, and the fused genitives <code>उसका</code>/<code>उसकी</code> were "
            f"missing, which cost the first row 3/31 instead of "
            f"{pol['arms'][0]['rescued']}/31."),
        "__T_LAT_CAV_SHORT__": (
            f"Every utterance in that corpus ends on a digit or an amount &mdash; the easy "
            f"case &mdash; so the true rate on open speech will be higher. The real-speech "
            f"corpus cannot settle it: its recordings are cut mid-phrase, so the rule firing "
            f"on them is the cut, not an error."),
        "__T_LAT_CAV__": (
            f"Read that bound with its corpus in mind: {lat['corpus_note']}. A digit word is "
            f"never in the list, so this measures the rule on the easy case and the true "
            f"rate on open-ended speech will be higher. <b>The real-speech corpus cannot "
            f"settle it and was not used.</b> Its entries are fixed-length segments that "
            f"end mid-phrase &mdash; <span class=\"deva\">\u0905\u0917\u0930</span>, "
            f"<span class=\"deva\">\u0915\u0940</span>, "
            f"<span class=\"deva\">\u0932\u0947\u0915\u093f\u0928</span> &mdash; so the "
            f"rule firing on them is the recording being cut, not the rule being wrong. "
            f"Scoring against them would have produced a 19.9% false-positive rate that is "
            f"nothing of the kind."),
        "__T_POL_NOTE__": (
            f"The rule fires on the turn's own text, so it inherits whatever the "
            f"recogniser did to that text. That is the whole of the filler row: the head "
            f"ends on <i>matlab</i> in all 31, and the recogniser returned it in "
            f"{pol['arms'][1]['rescued']} &mdash; where the word was dropped the rule has "
            f"nothing to fire on. <b>An earlier version of this table was much worse and "
            f"the reason was our own list.</b> It carried the bare postpositions "
            f"<code>का/के/की</code> but not the fused forms <code>उसका</code>, "
            f"<code>उसकी</code>, <code>उनका</code>, which are the same word plus a "
            f"pronoun and just as unable to end a sentence. 27 of 31 turns in the first "
            f"row ended on one and none matched, so the rule scored 3/31 where it now "
            f"scores {pol['arms'][0]['rescued']}/31. The word lists had also been copied "
            f"into three files that had drifted apart; there is one now, in "
            f"<code>asli.score</code>, and re-scoring costs nothing because the "
            f"transcripts are already stored."),
        "__T_AN_HEAD__": (
            f"It agrees with the human annotator {an['agreement']:.1%} of the time, against "
            f"{an['chance_agreement']:.1%} from chance alone &mdash; a Cohen&rsquo;s kappa of "
            f"<b>{an['cohens_kappa']:.2f}</b>, which is &ldquo;slight&rdquo; on the scale "
            f"annotation teams use and a long way below the 0.80 that would let it stand in "
            f"for a person. Writing &ldquo;finished&rdquo; on every row without listening "
            f"scores {an['majority_label_agreement']:.1%}."),
        "__T_AN_NOTE__": (
            f"Auto-accepting only the more reliable answer covers "
            f"{an['by_answer']['finished']['share']:.0%} of the work and would put "
            f"{an['by_answer']['finished']['errors']} wrong labels into "
            f"{an['by_answer']['finished']['n']} rows &mdash; a "
            f"{1 - an['by_answer']['finished']['precision']:.0%} error rate in the reference "
            f"data everything else is measured against. Pre-labelling saves time only when "
            f"the trusted slice is right about "
            f"{an['precision_needed_to_save_work']:.0%} of the time, so the human skims "
            f"rather than verifies. Getting there needs a confidence score from the "
            f"provider, not better engineering here."),
        "__T_DB_HEAD__": (
            f"A caller pauses {db['acoustic_auc']['pause_ms']['median_continue']}&nbsp;ms "
            f"when they are still talking and "
            f"{db['acoustic_auc']['pause_ms']['median_yield']}&nbsp;ms when they have "
            f"finished. There is no threshold that separates those, at any value. "
            f"<b>Every silence timer in production is built on this variable, and on "
            f"{db['export']['decisions']:,} human-labelled pauses it carries no signal "
            f"at all.</b>"),
        "__T_DB_ACC__": f"{db['semantic_vs_human']['accuracy'] * 100:.1f}%",
        "__T_DB_ALWAYS__": f"{db['semantic_vs_human']['always_reply_accuracy'] * 100:.1f}%",
        "__T_DB_CURVE__": (
            "There is no curve to tune along, and that is a finding rather than an "
            "omission: the API returns a verdict and no confidence, so those three rows "
            "are every setting there is. Reaching a 20% hold budget needs a probability "
            "from the provider, not better engineering on our side. Raw numbers: "
            "<code>results/diarbench.json</code>."),
        "__T_SEM_HEAD__": (
            f"On the same audio, at the same nominal gate: the silence timer ended the "
            f"turn early in <b>every one of {sem_sv} calls</b>. Semantic detection did it "
            f"in <b>{sem_worst}</b> of its worst arm, and <b>never</b> in the control."),
        "__T_SEM_N__": str(sem["n_calls"]),
        "__T_MODE_N__": str(modes["transcribe"]["hesitation"][1]),
        "__T_MODE_TRANSCRIBE_CONTROL__": mode_cell(modes, "transcribe", "control"),
        "__T_MODE_TRANSCRIBE_HESITATION__": mode_cell(modes, "transcribe", "hesitation"),
        "__T_MODE_VERBATIM_CONTROL__": mode_cell(modes, "verbatim", "control"),
        "__T_MODE_VERBATIM_HESITATION__": mode_cell(modes, "verbatim", "hesitation"),
        "__T_MODE_TRANSLIT_CONTROL__": mode_cell(modes, "translit", "control"),
        "__T_MODE_TRANSLIT_HESITATION__": mode_cell(modes, "translit", "hesitation"),
        "__T_MODE_CODEMIX_CONTROL__": mode_cell(modes, "codemix", "control"),
        "__T_MODE_CODEMIX_HESITATION__": mode_cell(modes, "codemix", "hesitation"),
        "__T_REAL_PAIR_N__": str(real_pair["n"]),
        "__T_REAL_PAIR_SERVER_PAUSE__": pair_cell("server", "in_pause"),
        "__T_REAL_PAIR_SEMANTIC_PAUSE__": pair_cell("semantic", "in_pause"),
        "__T_REAL_PAIR_SERVER_EARLY__": pair_cell("server", "early"),
        "__T_REAL_PAIR_SEMANTIC_EARLY__": pair_cell("semantic", "early"),
        "__T_SEM_NOTE__": (
            f"{sem['held_out']} calls are excluded from the semantic column: they returned "
            f"no turn, no transcript and no turn-end event, with no error &mdash; on audio "
            f"that produced a turn on every silence-timer row. That is what holding a turn "
            f"forever looks like from outside, so excluding them makes each semantic figure "
            f"an <b>upper bound</b>. One more thing cuts against this experiment and is kept: "
            f"both verb-final failures are imperatives &mdash; <i>likh leejiye</i>, "
            f"<i>note kar leejiye</i> &mdash; where ending the turn is defensible rather "
            f"than premature. Drop those two and that arm is 0/24, level with the control. "
            f"That is subsetting after seeing the data, so it is a question for the next "
            f"run and not a result of this one."),
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


def _policy_lane() -> dict:
    """The intervention's own numbers, recomputed from the stored rows.

    Measured on the same 186 calls as the semantic section, so the two are directly
    comparable rather than each quoting its own private run.
    """
    rows = json.loads((ROOT / "results/verbfinal2.json").read_text())
    ok = [r for r in rows if r["turns"] > 0 and not r.get("error")]
    arms = []
    for arm in ("dangler", "filler", "verb-final"):
        cut = [r for r in ok if r["mode"] == "server_vad" and r["arm"] == arm and r["split"]]
        held = sum(r["lexical_would_hold"] for r in cut)
        arms.append({"arm": arm, "cuts": len(cut), "rescued": held,
                     "residual": round(1 - held / len(cut), 3) if cut else None})
    sem = [r for r in ok if r["mode"] == "semantic_vad" and r["split"]]
    return {"arms": arms, "sem_residual": len(sem),
            "sem_rescued": sum(r["lexical_would_hold"] for r in sem)}


def _ledger() -> list[dict]:
    """Everything measured that has no section of its own.

    One honest list rather than a scene each: a result that did not earn a section still
    has to be findable, and several of these are negative.
    """
    def j(f):
        return json.loads((ROOT / f"results/{f}").read_text())

    def at_default(f):
        return next((r for r in j(f) if r["silence_ms"] == 512), j(f)[0])

    modes, rep, hes, vf1 = fixed_mode_matrix(), j("inepa_repeats.json"), \
        j("hesitation_summary.json"), j("verbfinal.json")
    return [
        {"what": "Output modes &mdash; entity kept after a hesitation",
         "n": "24 each",
         "value": " &middot; ".join(f"{k} {v['hesitation'][0]}/{v['hesitation'][1]}" for k, v in modes.items()),
         "reading": "The fixed-filler rerun does not reproduce the earlier all-or-nothing "
                    "mode claim. Only <code>transcribe</code> changes between the control "
                    "and hesitation conditions, so the old romanisation explanation is not "
                    "established by this run."},
        {"what": "The same suite, five identical runs",
         "n": f"{len(rep['runs'])} runs",
         "value": f"{pct(rep['mean'])} every run",
         "reading": "The same three entities miss every time. The failures are "
                    "deterministic, so one run is not a lucky draw &mdash; and anything "
                    "that moves this number moved something real."},
        {"what": "Does the hesitation itself damage recognition?",
         "n": f"{sum(hes.values())} pairs",
         "value": f"{hes['BROKEN BY HESITATION']} broken &middot; "
                  f"{hes['fixed by hesitation']} fixed &middot; "
                  f"{hes['both wrong']} wrong either way",
         "reading": "Broken and fixed cancel out. The hesitation is not damaging "
                    "recognition, it is ending the turn. Two different failures, and this "
                    "is what separates them."},
        {"what": "PIR on a degraded line, at the default gate",
         "n": "12 per point",
         "value": f"clean {at_default('pir_sweep_sarvam.json')['pir']} &middot; "
                  f"8&nbsp;kHz {at_default('pir_sweep_telephony.json')['pir']} &middot; "
                  f"babble {at_default('pir_sweep_degraded.json')['pir']}",
         "reading": "The telephony codec changes nothing. Heavy babble drives it to zero "
                    "&mdash; not an improvement: the noise fills the pause so the turn "
                    "never ends at all. The mirror failure."},
        {"what": "Experiment B, first attempt",
         "n": f"{len(vf1)} calls",
         "value": "inconclusive, kept",
         "reading": "The control failed and the design could not separate its own arms. "
                    "Written up with the reason in "
                    "<code>experiments/verbfinal-README.md</code> rather than deleted, "
                    "because the next person will otherwise build it the same way."},
    ]


def _union() -> dict:
    """Semantic detection and the rule fail on different cases. Does running both help?

    Reported with its p-value because it does not survive one: two events recovered.
    """
    import math

    rows = json.loads((ROOT / "results/verbfinal2.json").read_text())
    ok = [r for r in rows if r["mode"] == "semantic_vad" and r["turns"] > 0 and not r.get("error")]
    sem = sum(r["split"] for r in ok)
    uni = sum(r["split"] and not r["lexical_would_hold"] for r in ok)
    a, b, c, d = uni, len(ok) - uni, sem, len(ok) - sem

    def pr(x):
        b_, c_, d_ = a + b - x, a + c - x, d - (a - x)
        if min(b_, c_, d_) < 0:
            return 0.0
        return math.comb(a + b, x) * math.comb(c + d, c_) / math.comb(a + b + c + d, a + c)

    p0 = pr(a)
    p = sum(pr(x) for x in range(min(a + b, a + c) + 1) if pr(x) <= p0 + 1e-12)
    return {"semantic": sem, "union": uni, "n": len(ok), "recovered": sem - uni,
            "p": round(p, 3)}


BUDGET_MS = 50  # a product decision, so it is an input here and not an assumption


def _budget() -> list[dict]:
    """Latency as a constraint rather than a free variable.

    Under a tight budget the gate cannot move at all — the smallest useful step is
    +200 ms — so the comparison is not "which is more effective" but "which is even on
    the table". Raise the budget above 200 ms and the honest answer changes: a 900 ms
    gate prevents far more cuts than this rule does.
    """
    fit = json.loads((ROOT / "results/pause_fit.json").read_text())
    lat = json.loads((ROOT / "results/policy_latency.json").read_text())
    adv = {r["gate_ms"]: r for r in fitmod.gate_advice(fit)}
    rule_ms = lat["policy_expected_ms"]
    rows = [{"label": "leave the gate at 500 ms", "cut": pct(adv[500]["calls_affected"]),
             "cut_pct": adv[500]["calls_affected"] * 100, "ms": "0 ms", "fits": True},
            {"label": "raise the gate to 700 ms", "cut": pct(adv[700]["calls_affected"]),
             "cut_pct": adv[700]["calls_affected"] * 100,
             "ms": f"{adv[700]['added_latency_ms']} ms",
             "fits": adv[700]["added_latency_ms"] <= BUDGET_MS},
            {"label": "raise the gate to 900 ms", "cut": pct(adv[900]["calls_affected"]),
             "cut_pct": adv[900]["calls_affected"] * 100,
             "ms": f"{adv[900]['added_latency_ms']} ms",
             "fits": adv[900]["added_latency_ms"] <= BUDGET_MS},
            {"label": "our rule, gate stays at 500 ms",
             "cut": "not yet measured on real callers", "cut_pct": 0,
             "ms": f"&le; {rule_ms:.0f} ms", "fits": rule_ms <= BUDGET_MS}]
    return rows


def _diarbench() -> dict:
    """Measured on Sarvam's own human-annotated benchmark. See results/diarbench.json."""
    return json.loads((ROOT / "results/diarbench.json").read_text())


def collect() -> dict:
    d: dict = {"said": SAID}
    d["sweep"] = json.loads((ROOT / "results/pir_sweep_sarvam.json").read_text())
    d["fit"] = json.loads((ROOT / "results/pause_fit.json").read_text())
    d["text"] = _text_lane()
    d["vendors"] = _vendor_lane()
    d["diarbench"] = _diarbench()
    d["lanes"] = _lane_results()
    d["semantic"] = _semantic_lane()
    d["policy"] = _policy_lane()
    d["latency"] = json.loads((ROOT / "results/policy_latency.json").read_text())
    d["budget"] = _budget()
    d["ledger"] = _ledger()
    d["sample"] = _score_sample(json.loads((ROOT / "results/sample_call.json").read_text()))
    d["hero"] = {k: v for k, v in json.loads((ROOT / "results/hero_wave.json").read_text()).items()
                 if k != "env"}

    rows = json.loads((ROOT / "results/mode_placement_fixed.json").read_text())
    d["modes"] = {r["mode"]: {"text": r["transcript"], "got": r["extracted"], "ok": r["survived"]}
                  for r in rows
                  if r["id"] == "dig-01" and r["run"] == 0 and r["placement"] == "hesitation"}

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

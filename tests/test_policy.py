"""The intervention, pinned. No keys, no network:  python tests/test_policy.py

Every test here is a failure mode the bare word list had.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from asli import policy, score  # noqa: E402


def turn(t_ms, text, **kw):
    return {"t_ms": t_ms, "text": text, **kw}


def test_a_hold_expires_instead_of_waiting_forever():
    """The speaker really did stop on a dangling word. The turn must still end."""
    led = policy.Ledger()
    out = policy.apply([turn(0, "मेरा नंबर है उसका")], hold_ms=600, ledger=led)
    assert len(out) == 1 and led.held == 1 and led.expired == 1
    assert led.merged == 0, "nothing followed, so nothing can have merged"

    # and when the rest arrives too late, it is a separate turn, not a merge
    led2 = policy.Ledger()
    out2 = policy.apply([turn(0, "मेरा नंबर है उसका"), turn(2000, "9877111")],
                        hold_ms=600, ledger=led2)
    assert len(out2) == 2 and led2.expired == 1 and led2.merged == 0


def test_the_rest_arriving_in_time_is_merged():
    led = policy.Ledger()
    out = policy.apply([turn(0, "मेरा नंबर है उसका"), turn(400, "9877111")],
                       hold_ms=600, ledger=led)
    assert len(out) == 1 and out[0]["text"].endswith("9877111")
    assert led.merged == 1 and led.expired == 0


def test_never_merge_across_agent_speech():
    """What follows the agent talking may answer the agent. Gluing invents a sentence."""
    led = policy.Ledger()
    out = policy.apply([turn(0, "मेरा नंबर है उसका"),
                        turn(300, "हाँ", agent_spoke_before=True)], hold_ms=600, ledger=led)
    assert len(out) == 2, "must not merge across the agent's turn"
    assert led.refused_agent_spoke == 1 and led.merged == 0


def test_both_columns_are_counted():
    """A rule that only reports prevented interruptions can only produce good news."""
    led = policy.Ledger()
    policy.apply([turn(0, "वो है मतलब"), turn(300, "9877111"),      # merged
                  turn(9000, "उसका")], hold_ms=600, ledger=led)     # expired
    assert led.held == 2 and led.merged == 1 and led.expired == 1
    assert led.false_hold_rate == 0.5
    assert led.added_latency_ms(600, turns=10) == 60.0


def test_finality_markers_never_hold():
    for text in ("ठीक है धन्यवाद", "शुक्रिया"):
        assert policy.decide(text).hold is False, text
    assert policy.decide("ठीक है धन्यवाद").reason == "finality-marker"


def test_bas_holds_because_the_corpus_says_so():
    """It reads as "that's it" and was a finality marker on that reading. The corpus
    ends an utterance on it 1 time in 6, and it is already a dangling filler."""
    assert policy.decide("बस").hold is True
    assert not (policy.FINALITY & policy.DANGLING), "finished and unfinished are exclusive"


def test_derived_and_asserted_finality_are_separable():
    """The claim is that this list was counted. शुक्रिया was not — it does not occur in
    the corpus once — so it is kept where that stays visible."""
    assert policy.FINALITY_DERIVED and policy.FINALITY_ASSERTED
    assert policy.FINALITY == policy.FINALITY_DERIVED | policy.FINALITY_ASSERTED
    markers = json.loads((ROOT / "results/markers.json").read_text())
    kept = {m["word"] for m in markers["finality_markers"]
            if m["p_final"] >= policy.FINALITY_MIN_P}
    assert kept == set(policy.FINALITY_DERIVED), kept


def test_a_merged_turn_keeps_its_seam():
    """Merging feeds a longer string to an extractor that concatenates every digit it
    sees. The join is for turn-taking; the entity must be read off the parts."""
    led = policy.Ledger()
    out = policy.apply([turn(0, "मेरा नंबर है उसका"), turn(300, "नौ आठ सात सात")],
                       hold_ms=600, ledger=led)
    assert led.merged == 1 and out[0]["parts"] == ["मेरा नंबर है उसका", "नौ आठ सात सात"]

    stray = policy.apply([turn(0, "मेरा एक नंबर है उसका"), turn(300, "नौ आठ सात सात")],
                         hold_ms=600)[0]
    assert score.spoken_digits(stray["text"]) == "19877", "the join picks up the एक"
    assert score.spoken_digits(stray["parts"][-1]) == "9877", "the part does not"


def test_an_empty_final_falls_back_to_the_partial():
    """Measured in this repo: speech_final can arrive empty while the interim had it all."""
    assert policy.best_text("", "मेरा नंबर है उसका").endswith("उसका")
    assert policy.decide("", "मेरा नंबर है उसका").hold is True
    # the final wins when it is the fuller of the two
    assert policy.best_text("मेरा नंबर है उसका 9877111", "मेरा नंबर").endswith("9877111")


def test_english_fillers_fire_too():
    assert policy.decide("my number is, I mean").hold is True
    assert policy.decide("the amount is, actually").hold is True
    assert policy.decide("my number is 9877111").hold is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")

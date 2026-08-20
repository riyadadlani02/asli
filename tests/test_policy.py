"""The intervention, pinned. No keys, no network:  python tests/test_policy.py

Every test here is a failure mode the bare word list had.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asli import policy  # noqa: E402


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
    for text in ("ठीक है धन्यवाद", "बस", "शुक्रिया"):
        assert policy.decide(text).hold is False, text
    assert policy.decide("ठीक है धन्यवाद").reason == "finality-marker"


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

"""Calibration for the text instantiation. No API keys, no network.

    python tests/test_sfr_text.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asli import score  # noqa: E402
from asli.spec import CallSpec, Result, Segment  # noqa: E402
from asli.text import corrupt, run as textrun  # noqa: E402

RAW = textrun.load()


def item(corruption, detectable, entity_type="digits", canonical="9877111"):
    return textrun.Item(f"t-{corruption}", entity_type, canonical, "q?", ["r"],
                        corruption, detectable)


def result(value, confirmed):
    return Result(spec_id="t", adapter="text", agent_entity=value, confirmed=confirmed)


def test_every_authored_truth_round_trips():
    """20 hand-written records. One typo in a canonical silently poisons every rate."""
    for r in RAW:
        et, truth = r["entity_type"], r["canonical"]
        assert score.normalise(et, truth) == truth, r["id"]
        # the uncorrupted surface must actually encode that truth, or the clean arm
        # measures the corpus rather than the agent
        assert score.normalise(et, r["surface"]) == truth, (r["id"], r["surface"])


def test_each_class_does_what_it_claims():
    for r in RAW:
        base = corrupt.clean(r)
        for name, (fn, _) in corrupt.CLASSES.items():
            recs = fn(r)
            if recs is None:
                continue
            assert recs != base, (r["id"], name)
            joined = "\n".join(recs)
            if name == "omission":
                assert r["surface"] not in joined, (r["id"], "value survived omission")
            elif name == "contradiction":
                # the truth stays, a rival value is added: the records disagree
                assert r["surface"] in joined and len(recs) == len(base) + 1, r["id"]
            else:
                assert r["surface"] not in joined, (r["id"], name, "value not replaced")


def test_the_undetectable_class_stays_plausible():
    """The control only controls if its injected value could pass for real.

    Different from the truth, and still parseable — an unparseable string would be
    detectable, which is the one thing this class must not be.
    """
    for r in RAW:
        other = corrupt._swapped(r["surface"])
        assert other and other != r["surface"], r["id"]
        got = score.normalise(r["entity_type"], other)
        assert got, (r["id"], other, "injected value does not parse")
        assert got != r["canonical"], (r["id"], other, "injected value equals the truth")


def test_build_labels_the_control_and_only_the_control():
    """Links the taxonomy to the scoring. Without this, mislabelling one class as
    detectable pools an uncatchable error into the headline number and every test
    still passes.
    """
    labels = {it.corruption: it.detectable for it in textrun.build(RAW)}
    assert labels["clean"] is None
    assert labels["digit_swap"] is False, "the control must stay out of the input lane"
    assert all(v is True for k, v in labels.items() if k not in ("clean", "digit_swap"))
    # every class fires on at least one record — a builder that silently returns None
    # everywhere would shrink the corpus without failing anything else
    assert len(labels) == len(corrupt.CLASSES) + 1, sorted(labels)


def test_sfr_is_pinned_and_the_control_sits_outside_the_input_lane():
    detectable = [(item("truncation", True), result("5877111", False))] * 5
    agg = textrun.aggregate(detectable)
    assert agg["sfr_input"] == 1.0 and agg["sfr_input_n"] == 5
    assert agg["sfr_outcome"] == 1.0

    always = [(item("truncation", True), result("5877111", True))] * 5
    assert textrun.aggregate(always)["sfr_input"] == 0.0

    # the undetectable class is scored, but in its own lane — pooling it would let a
    # class nobody can catch drive the headline number
    undet = [(item("digit_swap", False), result("5877111", False))] * 5
    agg = textrun.aggregate(undet)
    assert agg["sfr_input"] is None and agg["sfr_input_n"] == 0
    assert agg["sfr_undetectable"] == 1.0 and agg["sfr_undetectable_n"] == 5

    # clean items are in no input denominator at all: nothing went wrong upstream
    clean = [(item("clean", None), result("9877111", False))] * 5
    agg = textrun.aggregate(clean)
    assert agg["sfr_input"] is None and agg["sfr_outcome"] is None


def test_clean_arm_measures_parsing_not_caution():
    """The entanglement measure must not move when only the confirm behaviour moves."""
    rows = [(item("clean", None, "amount", "250000"), result("250000", c))
            for c in (True, False, True, False)]
    assert textrun.aggregate(rows)["clean_accuracy"] == 1.0

    wrong = [(item("clean", None, "amount", "250000"), result("350000", True))] * 4
    assert textrun.aggregate(wrong)["clean_accuracy"] == 0.0


def test_one_function_scores_both_domains():
    """The construct claim, as an assertion: voice and text share the scorer.

    If these ever diverge, SFR is two metrics with one name and the write-up is wrong.
    """
    spec = CallSpec(id="s", entity_type="digits", canonical="9877111",
                    segments=[Segment("nine eight double seven triple one")])
    voice = Result(spec_id="s", adapter="mock",
                   transcript="mera number hai five six double seven triple one",
                   agent_entity="5677111", confirmed=False)
    assert score.mangled_entity(spec, voice.transcript) is True
    assert score.sfr_pair(spec, voice) == score.sfr("digits", "9877111", True,
                                                    "5677111", False)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")

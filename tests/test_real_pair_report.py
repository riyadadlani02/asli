from build_site import _openai_real_pair


def test_openai_real_pair_uses_the_same_twenty_callers_and_preserves_conditions():
    pair = _openai_real_pair()

    assert pair["n"] == 20
    assert pair["server"]["in_pause"] == 0.2
    assert pair["semantic"]["in_pause"] == 0.0
    assert pair["server"]["early"] == 0.75
    assert pair["semantic"]["early"] == 0.25

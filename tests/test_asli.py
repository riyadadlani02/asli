"""Calibration. If the mock cannot pin the scorers, no measured number means anything.

Runs with no API keys and no network:  python tests/test_asli.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from asli import degrade, score  # noqa: E402
from asli.drive import FRAME_SAMPLES, MockASR  # noqa: E402
from asli.spec import CallSpec, Result, Segment  # noqa: E402

RATE = 16000


def tone(ms, hz=1000, amp=8000):
    t = np.arange(int(RATE * ms / 1000)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def silence(ms):
    return np.zeros(int(RATE * ms / 1000), dtype=np.int16)


def hesitant_utterance(pause_ms=700):
    """500ms speech, a pause, 500ms speech. True end is known: we built it."""
    pcm = np.concatenate([tone(500), silence(pause_ms), tone(500)])
    spec = CallSpec(
        id="t", entity_type="digits", canonical="9877111",
        segments=[Segment("head", pause_after_ms=pause_ms), Segment("tail")],
        seg_bounds_ms=[(0, 500), (500 + pause_ms, 1000 + pause_ms)],
        true_end_ms=1000 + pause_ms,
    )
    return pcm, spec


def test_pir_fires_only_when_the_gate_is_shorter_than_the_pause():
    pcm, spec = hesitant_utterance(pause_ms=700)
    frame_ms = FRAME_SAMPLES * 1000 / RATE  # 32ms

    # 8 frames = 256ms of silence tolerated < 700ms pause -> cut off mid-thought
    short = score.pir(spec, MockASR(negative_frames_count=8).run(pcm, spec))
    assert short.premature, "a 256ms gate must trip on a 700ms hesitation"
    assert 500 <= short.t_ms <= 500 + 700, f"endpoint landed outside the pause: {short.t_ms}"
    assert short.ms_early == spec.true_end_ms - short.t_ms

    # 32 frames = 1024ms > 700ms pause -> the caller gets to finish
    long = score.pir(spec, MockASR(negative_frames_count=32).run(pcm, spec))
    assert not long.premature, "a 1024ms gate must survive a 700ms hesitation"

    # the boundary sits where the arithmetic says it does
    gate_frames = int(700 / frame_ms)
    assert score.pir(spec, MockASR(negative_frames_count=gate_frames - 4).run(pcm, spec)).premature
    assert not score.pir(spec, MockASR(negative_frames_count=gate_frames + 6).run(pcm, spec)).premature


def test_sfr_is_pinned_by_agents_with_known_behaviour():
    spec = CallSpec(id="s", entity_type="digits", canonical="9877111",
                    segments=[Segment("nine eight double seven triple one")])
    intact = "mera number hai nine eight double seven triple one"
    mangled = "mera number hai five six double seven triple one"

    def row(confirmed, heard, final):
        return spec, Result(spec_id="s", adapter="mock", transcript=heard,
                            agent_entity=final, confirmed=confirmed)

    # entity damaged, agent never confirms -> every one is a silent failure
    never = [row(False, mangled, "5677111") for _ in range(5)]
    assert score.aggregate(never)["sfr_asr"] == 1.0
    assert score.aggregate(never)["sfr_bb"] == 1.0

    # same damage, agent always confirms -> none of them are silent
    always = [row(True, mangled, "5677111") for _ in range(5)]
    assert score.aggregate(always)["sfr_asr"] == 0.0

    # entity survived -> the call is not in the denominator at all
    clean = [row(False, intact, "9877111") for _ in range(5)]
    assert score.aggregate(clean)["sfr_asr"] is None
    assert score.aggregate(clean)["sfr_asr_n"] == 0

    # the two variants genuinely differ: ASR was wrong, the agent recovered anyway
    agg = score.aggregate([row(False, mangled, "9877111")])
    assert agg["sfr_asr"] == 1.0 and agg["sfr_bb"] is None


def test_sfr_abstains_when_the_scripts_are_not_comparable():
    """A Devanagari transcript vs a romanised script says nothing about the entity."""
    spec = CallSpec(id="d", entity_type="digits", canonical="9877111",
                    segments=[Segment("nine eight double seven triple one")])
    deva = "मेरा मोबाइल नंबर है मतलब नाइन एट डबल सेवन ट्रिपल वन"

    assert score.mangled_entity(spec, deva) is None, "must abstain, not claim damage"
    row = (spec, Result(spec_id="d", adapter="sarvam", transcript=deva,
                        agent_entity="9877111", confirmed=False))
    agg = score.aggregate([row])
    assert agg["sfr_asr"] is None and agg["sfr_asr_n"] == 0

    # a romanised transcript is still compared normally
    assert score.mangled_entity(spec, "hai nine eight double seven triple one") is False
    assert score.mangled_entity(spec, "hai five six double seven triple one") is True

    # and a Devanagari script vs a Devanagari transcript stays comparable
    deva_spec = CallSpec(id="d2", entity_type="digits", canonical="9877111",
                         segments=[Segment("नाइन एट डबल सेवन ट्रिपल वन")])
    assert score.mangled_entity(deva_spec, deva) is False


def test_spoken_digits_binds_multipliers_to_the_following_digit():
    assert score.spoken_digits("nine eight double seven, triple one") == "9877111"
    assert score.spoken_digits("char zero double five six") == "40556"  # 4-0-55-6, not 4-0-5-5-6
    assert score.spoken_digits("eight, double zero, nine") == "8009"
    assert score.normalise("amount", "Rs 2,50,000") == "250000"


def test_telephony_keeps_the_tone_and_drops_the_top_octave():
    speech_band, above_nyquist = tone(400, hz=1000), tone(400, hz=6000)
    out_lo = degrade.telephony(speech_band)
    out_hi = degrade.telephony(above_nyquist)

    def peak_hz(x):
        spec = np.abs(np.fft.rfft(x.astype(float)))
        return np.fft.rfftfreq(len(x), 1 / RATE)[np.argmax(spec)]

    assert abs(peak_hz(out_lo) - 1000) < 20, "1kHz must survive a telephony leg"
    # 6kHz is above the 4kHz Nyquist of an 8k line — it cannot come back
    assert np.abs(out_hi).mean() < np.abs(out_lo).mean() * 0.5


def test_noise_lands_at_the_snr_we_asked_for():
    clean = tone(1000)
    for target in (20, 10, 5):
        noisy = degrade.add_noise(clean, degrade.pink_noise(len(clean)), target)
        got = degrade.measure_snr(clean, noisy)
        assert abs(got - target) < 1.0, f"asked {target}dB, measured {got:.2f}dB"


def test_packet_loss_removes_roughly_the_share_requested():
    # a tone crosses zero, so count whole dropped frames against a source that never does
    clean = (degrade.pink_noise(RATE * 4) * 8000).astype(np.int16)
    lossy = degrade.packet_loss(clean, rate_pct=10, seed=1)
    n = int(RATE * 20 / 1000)
    frames = [lossy[i : i + n] for i in range(0, len(lossy) - n, n)]
    dropped = np.mean([bool(np.all(f == 0)) for f in frames])
    assert 0.05 < dropped < 0.16, f"expected ~10% of frames dropped, got {dropped:.1%}"


def test_render_timing_is_exact_by_construction(monkeypatch=None):
    """The splice, not an aligner, is what defines true_end_ms."""
    from asli import synth

    # constant amplitude: nothing for trim() to remove, so the arithmetic is visible
    synth.tts = lambda text, voice=None: np.full(int(RATE * 0.3), 5000, dtype=np.int16)
    spec = CallSpec(id="r", entity_type="digits", canonical="1",
                    segments=[Segment("a", pause_after_ms=500), Segment("b")])
    pcm = synth.render(spec)
    assert spec.seg_bounds_ms == [(0, 300), (800, 1100)]
    assert spec.true_end_ms == 1100
    assert len(pcm) == int(RATE * 1.1), "duration must equal segments + inserted silence"
    assert spec.internal_pauses == [(300, 800)]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")

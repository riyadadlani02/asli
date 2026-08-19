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


def test_entity_survival_is_script_independent():
    """Devanagari, roman, digit words and normalised digits are the same entity."""
    spec = CallSpec(id="d", entity_type="digits", canonical="9877111",
                    segments=[Segment("nine eight double seven triple one")])

    for intact in ("मेरा मोबाइल नंबर है नाइन एट डबल सेवन ट्रिपल वन",
                   "Mera mobile number hai 9877111",
                   "मेरा नंबर ९८७७१११",
                   "hai nine eight double seven triple one"):
        assert score.mangled_entity(spec, intact) is False, intact

    # the real failure seen live: one digit lost after a hesitation
    assert score.mangled_entity(spec, "Mera mobile number hai matlab 987711") is True

    # non-entity words may contribute digits ("last five digits") without being damage
    spec2 = CallSpec(id="d2", entity_type="digits", canonical="40556",
                     segments=[Segment("char zero double five six")])
    assert score.mangled_entity(spec2, "Account ka last five digits. Chaar zero double five six.") is False

    # nothing numeric recoverable -> abstain rather than invent damage
    assert score.mangled_entity(spec, "sorry, could you repeat that") is None
    row = (spec, Result(spec_id="d", adapter="sarvam", transcript="sorry, could you repeat that",
                        agent_entity="9877111", confirmed=False))
    assert score.aggregate([row])["sfr_asr_n"] == 0


def test_indian_amounts_parse_as_lakh_and_crore():
    """Lakh and crore are their own scales, and 'saade' attaches before the scale."""
    for text, want in [
        ("do lakh pachas hazaar rupaye", "250000"),
        ("saade teen lakh", "350000"),
        ("saade sat hazaar", "7500"),
        ("ek crore pachas lakh", "15000000"),
        ("one point two five lakh rupees", "125000"),
        ("दो लाख पचास हज़ार", "250000"),
        ("Rs 2,50,000", "250000"),      # a figure, not speech
        ("₹1,25,000", "125000"),
        ("250000", "250000"),
    ]:
        assert score.normalise("amount", text) == want, f"{text} -> {score.normalise('amount', text)}"


def test_pause_detection_survives_a_real_noise_floor():
    """A fixed threshold works on synthesised audio and fails on real recordings."""
    from asli.fit import adaptive_threshold, frame_rms, pause_lengths_ms

    built = [250, 700, 1100, 480]
    parts = []
    for gap in built:
        parts += [tone(300), silence(gap)]
    clean = np.concatenate(parts + [tone(300)])

    rng = np.random.default_rng(0)
    quiet = (clean + rng.normal(0, 300, len(clean))).astype(np.int16)
    loud = (clean + rng.normal(0, 2000, len(clean))).astype(np.int16)

    # the same gaps come back at three very different noise floors
    for label, sig in (("clean", clean), ("quiet", quiet), ("loud", loud)):
        got = pause_lengths_ms(sig, RATE)
        assert len(got) == len(built), f"{label}: {got}"
        assert all(abs(a - b) <= 40 for a, b in zip(built, got)), f"{label}: {got}"

    # pure noise has no speech/silence contrast — abstain rather than invent pauses
    noise = rng.normal(0, 500, RATE * 3).astype(np.int16)
    assert adaptive_threshold(frame_rms(noise, 320)) is None
    assert pause_lengths_ms(noise, RATE) == []

    # leading and trailing silence is recording margin, not hesitation
    padded = np.concatenate([silence(900), tone(300), silence(500), tone(300), silence(900)])
    assert pause_lengths_ms(padded, RATE) == [500]  # exact at 20ms frames


def test_fit_counts_only_the_clips_usable_as_a_real_caller():
    """`files_with_long_pause` is cited as a headline count, so pin it."""
    import tempfile

    from asli.fit import fit_corpus
    from asli.synth import write_wav

    def clip(gaps):
        parts = []
        for g in gaps:
            parts += [tone(200), silence(g)]
        return np.concatenate(parts + [tone(200)])

    with tempfile.TemporaryDirectory() as d:
        short, long_ = Path(d) / "short.wav", Path(d) / "long.wav"
        write_wav(short, clip([300] * 12), RATE)         # nothing near the gate
        write_wav(long_, clip([300] * 11 + [700]), RATE)  # one hesitation-length pause
        f = fit_corpus([short, long_], long_pause_ms=500)

    assert f["files_used"] == 2, f
    assert f["files_with_long_pause"] == 1, f            # the 300ms clip is not a caller
    assert f["long_pause_ms"] == 500


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


def test_recovery_separates_the_first_turn_view_from_the_session():
    """The cost of acting at end-of-turn: the agent holds less than the session does."""
    pcm = np.concatenate([tone(500), silence(700), tone(500)])
    spec = CallSpec(
        id="rec", entity_type="digits", canonical="9877111",
        segments=[Segment("mera number hai", pause_after_ms=700),
                  Segment("nine eight double seven triple one")],
        seg_bounds_ms=[(0, 500), (1200, 1700)], true_end_ms=1700,
    )
    res = MockASR(negative_frames_count=8).run(pcm, spec)
    r = score.recovery(spec, res, reply_latency_ms=800)

    assert r.cut, "a 256ms gate must trip on the 700ms hesitation"
    assert r.entity_first_turn is False, "the digits had not been said yet at the cut"
    assert r.entity_full_session is True, "they are in the session transcript"
    assert r.silence_budget_ms == spec.true_end_ms - score.pir(spec, res).t_ms
    assert r.collides is True, "800ms of agent lag < the budget: it talks over the number"

    # a slower agent stops colliding, and that is the only charitable direction
    assert score.recovery(spec, res, reply_latency_ms=5000).collides is False

    # not cut off -> the two views agree and there is nothing to collide with
    ok = score.recovery(spec, MockASR(negative_frames_count=32).run(pcm, spec))
    assert not ok.cut and ok.entity_first_turn and ok.collides is None


def test_real_audio_spec_locates_the_pause_and_bounds_its_own_error():
    """No splice: the pause and the true end are measured, and the error is reported."""
    from asli import real

    rate = 8000  # corpus telephony rate
    t = lambda ms: (8000 * np.sin(2 * np.pi * 400 * np.arange(int(rate * ms / 1000)) / rate)
                    ).astype(np.int16)
    z = lambda ms: np.zeros(int(rate * ms / 1000), dtype=np.int16)
    pcm = np.concatenate([z(200), t(800), z(700), t(600), z(300)])

    spec, spread = real.spec_from_audio(pcm, rate, "synthetic")
    assert spec.internal_pauses == [(1000, 1700)], "the 700ms gap, found not built"
    assert abs(spec.true_end_ms - 2300) <= 20, "true end within one analysis frame"
    assert spread <= 40, "the end must not move much when the threshold does"

    # and the scorer works on it unchanged, which is the point of the CallSpec shape
    assert score.pir(spec, MockASR(negative_frames_count=8, rate=rate).run(pcm, spec)).premature

    # a recording with no mid-utterance pause says nothing about PIR and is dropped
    assert real.spec_from_audio(np.concatenate([t(800), z(100), t(800)]), rate, "flat") is None


def test_gate_advice_prices_the_latency_and_recommends_within_budget():
    from asli import fit

    f = {"exceed": {"300": .553, "400": .30, "500": .179, "700": .059, "900": .004, "1200": .0},
         "calls_exceed": {"300": .90, "400": .70, "500": .43, "700": .18, "900": .02, "1200": .0}}
    rows = fit.gate_advice(f)
    assert [r["added_latency_ms"] for r in rows] == [-200, -100, 0, 200, 400, 700]
    calls = [r["calls_affected"] for r in rows]
    assert calls == sorted(calls, reverse=True), "a longer gate cannot affect more calls"

    rec = fit.recommended_gate(f, budget_ms=400)
    assert rec["gate_ms"] == 900 and rec["calls_affected"] == .02
    assert fit.recommended_gate(f, budget_ms=200)["gate_ms"] == 700, "budget must bind"
    assert fit.recommended_gate(f, budget_ms=0)["gate_ms"] == 500, "no budget, no change"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")

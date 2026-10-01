"""The test bench, offline. A fake Sarvam socket stands in for the vendor so the live path —
real-time streaming, turn pairing, partials, the verdict — is exercised without a key.

The fake is a test fixture and nothing else: it never ships in the bench's provider list,
because a stand-in that looks like a vendor is exactly what the bench exists to rule out.
"""

import asyncio
import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import yaml

from asli import bench, kit
from asli.bench_server import BenchServer
from asli.drive import EventLog, SarvamWS
from asli.spec import Event, Result, to_jsonl

RATE = 16000
ROOT = Path(__file__).parent.parent


def tone(ms, amp=8000):
    t = np.arange(int(RATE * ms / 1000)) / RATE
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.int16)


def silence(ms):
    return np.zeros(int(RATE * ms / 1000), np.int16)


def hesitant(pause_ms=800):
    """600 ms speech, a pause, 600 ms speech — a caller hesitating before the number."""
    return np.concatenate([silence(200), tone(600), silence(pause_ms), tone(600)])


# --- a fake Sarvam, protocol-accurate enough to drive SarvamWS --------------------------

TURNS = ["मेरा मोबाइल नंबर है मतलब", "नाइन एट डबल सेवन ट्रिपल वन"]


async def _fake_sarvam(ws):
    q = parse_qs(urlparse(ws.request.path).query)
    gate, rate = int(q["silence_duration_ms"][0]), int(q["sample_rate"][0])
    speaking, quiet, turn = False, 0, 0
    async for raw in ws:
        m = json.loads(raw)
        if m["event"] == "audio_input":
            pcm = np.frombuffer(base64.b64decode(m["audio"]), "<i2").astype(np.float64)
            ms = len(pcm) * 1000 // rate
            loud = np.sqrt(np.mean((pcm / 32768) ** 2)) > 0.01
            if loud:
                if not speaking:
                    speaking = True
                    await ws.send(json.dumps({"event": "vad.speech_start"}))
                quiet = 0
                await ws.send(json.dumps({"event": "transcript.partial",
                                          "text": TURNS[min(turn, len(TURNS) - 1)]}))
            elif speaking:
                quiet += ms
                if quiet >= gate:  # the silence timer, on the audio clock
                    speaking, quiet = False, 0
                    await ws.send(json.dumps({"event": "vad.speech_end"}))
                    await ws.send(json.dumps({"event": "transcript.final",
                                              "text": TURNS[min(turn, len(TURNS) - 1)]}))
                    turn += 1
        elif m["event"] == "end":
            await ws.send(json.dumps({"event": "session.end"}))
            return


@pytest.fixture
def fake_sarvam(monkeypatch):
    """Point SarvamWS at a local fake, and stream 10x faster than real time.

    Speeding up is safe because both sides keep time on the audio clock (samples sent),
    not the wall clock — which is the property the real lane depends on too.
    """
    import websockets

    ready, stop = threading.Event(), threading.Event()
    port = {}

    def serve():
        async def main():
            async with websockets.serve(_fake_sarvam, "127.0.0.1", 0) as srv:
                port["n"] = srv.sockets[0].getsockname()[1]
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.02)
        asyncio.run(main())

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    ready.wait(5)
    monkeypatch.setattr(SarvamWS, "URL", f"ws://127.0.0.1:{port['n']}/ws")
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s, *a, **k: real_sleep(s / 10, *a, **k))
    yield
    stop.set()
    th.join(2)


# --- analysis ---------------------------------------------------------------------------

def test_analysis_finds_the_pause_and_the_end_of_speech():
    an = bench.analyse(hesitant(800))
    assert an.usable
    assert abs(an.speech_start_ms - 200) <= 20 and abs(an.speech_end_ms - 2200) <= 20
    assert len(an.pauses) == 1
    a, z = an.pauses[0]
    assert abs(a - 800) <= 20 and abs(z - 1600) <= 20
    assert an.longest_pause_ms >= 780


def test_analysis_abstains_when_speech_cannot_be_separated():
    an = bench.analyse(tone(2000))  # all "speech", no floor to compare against
    assert not an.usable
    assert bench.spec_for("x", an).true_end_ms == 0


def test_resampling_keeps_duration():
    pcm = (np.sin(np.arange(48000) / 10) * 5000).astype(np.int16)  # 1 s at 48 kHz
    assert len(bench.to_16k(pcm, 48000)) == 16000
    assert len(bench.to_16k(pcm[:8000], 8000)) == 16000


def test_wav_round_trip_and_stereo_downmix():
    pcm = hesitant()
    back, rate = bench.read_wav_bytes(bench.wav_bytes(pcm))
    assert rate == RATE and np.array_equal(back, pcm)
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.stack([pcm, pcm], axis=1).astype("<i2").tobytes())
    mono, _ = bench.read_wav_bytes(buf.getvalue())
    assert len(mono) == len(pcm)


# --- ground truth -------------------------------------------------------------------------

def test_expected_values_are_read_the_way_people_type_them():
    assert bench.normalise_expected("digits", "98771 11") == "9877111"
    assert bench.normalise_expected("digits", "nine eight double seven triple one") == "9877111"
    assert bench.normalise_expected("amount", "2,50,000") == "250000"
    assert bench.normalise_expected("amount", "do lakh pachas hazaar") == "250000"
    assert bench.normalise_expected("date", "15/09/2026") == "2026-09-15"
    with pytest.raises(ValueError):
        bench.normalise_expected("date", "next tuesday")
    with pytest.raises(ValueError):
        bench.normalise_expected("colour", "red")


def test_entity_check_reads_scripts_and_abstains_when_it_cannot():
    assert bench.entity_found("digits", "9877111", "नाइन एट डबल सेवन ट्रिपल वन") is True
    assert bench.entity_found("digits", "9877111", "मेरा नंबर है मतलब") is None   # no digits at all
    assert bench.entity_found("digits", "9877111", "nine eight seven seven one") is False
    assert bench.entity_found("digits", "9877111", "") is False
    assert bench.entity_found("amount", "250000", "transfer do lakh pachas hazaar") is True
    assert bench.entity_found("amount", "250000", "transfer karna hai matlab 2,50,000 rupaye") is True
    assert bench.entity_found("amount", "250000", "मतलब २,५०,००० रुपये") is True
    assert bench.entity_found("amount", "250000", "matlab 2,25,000 rupaye") is False
    assert bench.entity_found("date", "2026-09-15", "it was on 15/9/2026") is True


# --- the verdict ---------------------------------------------------------------------------

def _res(*events):
    return Result(spec_id="t", adapter="sarvam", events=[Event(*e) for e in events])


def test_a_cut_inside_the_pause_is_reported_with_its_words_and_the_policy_rescue():
    an = bench.analyse(hesitant(800))
    spec = bench.spec_for("t", an, "9877111", "digits")
    pause_start = an.pauses[0][0]
    cut = pause_start + 500
    res = _res(("speech_end", cut), ("transcript", cut, "मेरा मोबाइल नंबर है मतलब"),
               ("speech_end", 2300), ("transcript", 2300, "नाइन एट डबल सेवन ट्रिपल वन"))
    v = bench.verdict(spec, an, res, "sarvam", 500, partials=[], hold_ms=600)
    assert v["status"] == "cut"
    assert v["cuts"][0]["in_pause"] == list(an.pauses[0])
    assert v["at_cut"]["last_word"] == "मतलब" and v["at_cut"]["hold"]
    assert v["policy"]["fires"] and v["policy"]["rescued"]
    assert v["policy"]["wait_needed_ms"] == an.pauses[0][1] - cut
    assert v["entity"]["in_first_turn"] is False
    assert v["entity"]["in_session"] is True


def test_a_turn_that_ends_after_speech_is_not_a_cut():
    an = bench.analyse(hesitant(800))
    spec = bench.spec_for("t", an, "9877111", "digits")
    res = _res(("speech_end", 2700), ("transcript", 2700, "मेरा नंबर है नाइन एट डबल सेवन ट्रिपल वन"))
    v = bench.verdict(spec, an, res, "sarvam", 900, partials=[])
    assert v["status"] == "ok" and not v["cuts"]
    assert v["entity"]["in_first_turn"] is True


def test_romanised_turns_are_not_judged_by_a_devanagari_list():
    mt = bench.marker_test("my number is")
    assert mt["reason"] == "not-checked" and not mt["hold"]
    assert bench.marker_test("my number is umm")["hold"]       # English fillers are listed
    assert bench.marker_test("नंबर है")["reason"] == "complete"


def test_gemini_fragments_are_split_at_the_inferred_turn_end():
    res = Result(spec_id="t", adapter="gemini", events=[
        Event("transcript", 300, "मेरा नंबर"), Event("transcript", 600, " है मतलब"),
        Event("speech_end", 1900), Event("transcript", 2400, "नाइन एट")])
    turns = bench.turns_of(res, "gemini")
    assert turns[0] == {"end_ms": 1900, "text": "मेरा नंबर है मतलब"}
    assert turns[1]["text"] == "नाइन एट" and turns[1]["end_ms"] is None


def test_a_missing_key_sends_nothing_and_says_so(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    rep = bench.run(hesitant(), ["deepgram"], save=False)
    r = rep["results"]["deepgram"]
    assert r["verdict"]["status"] == "error"
    assert "DEEPGRAM_API_KEY" in r["error"] and "nothing was sent" in r["error"]


def test_the_mock_is_labelled_and_never_claims_words():
    rep = bench.run(hesitant(800), ["mock"], gate=500, expected="9877111", save=False)
    r = rep["results"]["mock"]
    assert r["provenance"]["kind"] == "mock"
    assert "NOT a vendor" in r["provenance"]["provider"]
    assert r["verdict"]["status"] == "cut"
    assert all(t["text"] == "" for t in r["verdict"]["turns"])
    assert "not_applicable" in r["verdict"]["entity"]
    assert "at_cut" not in r["verdict"]


def test_long_recordings_are_refused_before_anything_is_streamed():
    with pytest.raises(ValueError, match="at most"):
        bench.run(np.zeros(RATE * (bench.MAX_SECONDS + 1), np.int16), ["mock"], save=False)


def test_event_log_reports_events_and_survives_serialisation():
    seen = []
    log = EventLog(on_event=seen.append)
    log.append(Event("speech_end", 10))
    assert seen == [Event("speech_end", 10)]
    from asli.spec import CallSpec, Segment
    spec = CallSpec(id="x", segments=[Segment("")], entity_type="digits", canonical="")
    assert '"speech_end"' in to_jsonl(spec, Result(spec_id="x", adapter="a", events=log))


# --- the live path, against the fake socket --------------------------------------------------

def test_live_run_streams_events_as_they_happen_and_scores_the_call(fake_sarvam, tmp_path):
    seen = []
    rep = bench.run(hesitant(800), ["sarvam"], gate=500, expected="9877111", name="fake call",
                    emit=seen.append, runs_dir=tmp_path)
    kinds = [m["type"] for m in seen]
    assert kinds[0] == "analysis" and kinds[-1] == "done"
    assert "partial" in kinds and kinds.index("event") < kinds.index("provider_done")

    r = rep["results"]["sarvam"]
    v = r["verdict"]
    assert r["provenance"]["kind"] == "live" and r["provenance"]["settings"]["silence_duration_ms"] == 500
    assert v["status"] == "cut", v["headline"]
    assert v["turns"][0]["text"] == TURNS[0] and v["turns"][1]["text"] == TURNS[1]
    assert v["entity"]["in_first_turn"] is False       # the number came in turn 2
    assert v["entity"]["in_session"] is True
    assert v["entity"]["in_partials"] is True
    assert v["policy"]["fires"]                          # it ended on मतलब

    saved = tmp_path / rep["id"]
    assert (saved / "audio.wav").exists()
    stored = json.loads((saved / "report.json").read_text())
    assert stored["results"]["sarvam"]["verdict"]["status"] == "cut"
    assert bench.history(tmp_path)[0]["statuses"] == {"sarvam": "cut"}


def test_a_gate_longer_than_the_pause_lets_the_caller_finish(fake_sarvam):
    rep = bench.run(hesitant(800), ["sarvam"], gate=1200, expected="9877111", save=False)
    v = rep["results"]["sarvam"]["verdict"]
    assert v["status"] == "ok", v["headline"]
    assert v["n_turns"] == 1


# --- the sample kit ----------------------------------------------------------------------------

def test_every_kit_sample_carries_the_truth_its_audio_was_built_with():
    rows = bench.load_samples()
    assert len(rows) == 18
    for s in rows:
        pcm = bench.load_audio(ROOT / s["file"])
        a, z = s["pause_ms"]
        assert z - a == s["hesitation_ms"]
        if s["line"] == "clean":
            # the spliced pause is exact digital silence, start to end
            seg = pcm[int(RATE * (a + 1) / 1000):int(RATE * (z - 1) / 1000)]
            assert seg.size and not seg.any(), s["id"]
            assert np.abs(pcm[int(RATE * (z + 5) / 1000):int(RATE * (z + 60) / 1000)]).max() > 0
        assert s["speech_end_ms"] <= round(len(pcm) * 1000 / RATE)
        assert bench.normalise_expected(s["entity_type"], s["expected"]) == s["expected"]


def test_kit_rebuild_is_deterministic(tmp_path):
    rows = kit.build(tmp_path)
    for r in rows:
        if r["file"].startswith("samples/"):
            name = Path(r["file"]).name
            assert (tmp_path / name).read_bytes() == (ROOT / r["file"]).read_bytes(), name
    assert yaml.safe_load((tmp_path / "manifest.yaml").read_text()) == \
        yaml.safe_load((ROOT / "samples" / "manifest.yaml").read_text())


def test_manifest_entries_cannot_point_outside_the_repo(tmp_path):
    m = tmp_path / "manifest.yaml"
    m.write_text(yaml.safe_dump([
        {"id": "a", "file": "../../etc/passwd", "title": "t", "said": "s",
         "entity_type": "digits", "expected": "1"},
        {"id": "b", "file": "README.md", "title": "t", "said": "s",
         "entity_type": "digits", "expected": "1"},
        {"id": "c", "file": "samples/mobile-700ms.wav", "title": "t", "said": "s",
         "entity_type": "digits", "expected": "9877111"}]))
    assert [r["id"] for r in bench.load_samples(m)] == ["c"]


def test_kit_samples_are_judged_against_their_spliced_truth():
    s = next(x for x in bench.load_samples() if x["id"] == "mobile-700ms-noisy")
    rep = bench.run(bench.load_audio(ROOT / s["file"]), ["mock"], gate=500, save=False, sample=s)
    assert rep["analysis"]["basis"] == "authored"
    assert [list(p) for p in rep["analysis"]["pauses"]] == [s["pause_ms"]]
    assert rep["analysis_measured"]["basis"] == "measured"
    assert rep["ground_truth"]["value"] == "9877111"


# --- the server ---------------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    def start(passcode=None):
        srv = BenchServer(("127.0.0.1", 0), passcode=passcode, runs_dir=tmp_path)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"
    started = []
    yield start
    for s in started:
        s.shutdown()
        s.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read()


def test_server_serves_the_page_config_and_kit(server):
    base = server()
    status, page = _get(base + "/")
    assert status == 200 and b"asli test bench" in page
    cfg = json.loads(_get(base + "/api/config")[1])
    ids = {p["id"] for p in cfg["providers"]}
    assert {"sarvam", "deepgram", "openai", "mock"} <= ids
    assert len(cfg["samples"]) == 18
    status, wav = _get(base + "/samples/mobile-700ms")
    assert status == 200 and wav[:4] == b"RIFF"


def test_server_never_returns_a_key(server, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sk-very-secret")
    base = server()
    assert b"sk-very-secret" not in _get(base + "/api/config")[1]


def test_server_requires_the_passcode_when_set(server):
    base = server(passcode="open-sesame")
    assert _get(base + "/")[0] == 200            # the page itself holds nothing
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/config")
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError):
        _get(base + "/api/config", {"X-Asli-Passcode": "wrong"})
    assert _get(base + "/api/config", {"X-Asli-Passcode": "open-sesame"})[0] == 200


def test_server_refuses_paths_outside_its_runs(server):
    base = server()
    for bad in ("/api/run/..%2F..%2Fetc", "/api/run/../../README.md", "/samples/..%2FREADME.md"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + bad)
        assert e.value.code == 404


def test_server_run_streams_ndjson_and_keeps_a_receipt(server):
    base = server()
    body = bench.wav_bytes(hesitant(800))
    req = urllib.request.Request(
        base + "/api/run?providers=mock&gate=500&expected=9877111&name=server+test&source=microphone",
        data=body, method="POST", headers={"Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=30) as r:
        msgs = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    assert msgs[0]["type"] == "analysis" and msgs[-1]["type"] == "done"
    rep = msgs[-1]["report"]
    assert rep["source"] == "microphone" and rep["results"]["mock"]["verdict"]["status"] == "cut"
    hist = json.loads(_get(base + "/api/history")[1])
    assert hist[0]["id"] == rep["id"]
    assert _get(base + f"/api/run/{rep['id']}/audio.wav")[1][:4] == b"RIFF"


def test_server_rejects_bad_requests_before_streaming(server):
    base = server()
    for q in ("providers=nobody&gate=500", "providers=mock&gate=5", "providers=mock&sample=nope",
              "providers=mock&type=date&expected=someday", "providers="):
        req = urllib.request.Request(base + "/api/run?" + q, data=bench.wav_bytes(hesitant()),
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 400


def test_server_refuses_a_tunnel_without_a_passcode(server):
    base = server()
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/config", {"X-Forwarded-For": "203.0.113.9"})
    assert e.value.code == 403
    base = server(passcode="pw")
    assert _get(base + "/api/config", {"X-Forwarded-For": "203.0.113.9",
                                        "X-Asli-Passcode": "pw"})[0] == 200

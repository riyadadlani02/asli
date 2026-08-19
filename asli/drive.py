"""Adapters: stream audio at a system under test, capture timestamped events.

An adapter is anything with `run(pcm, spec) -> Result`. Two ship here:

  MockASR  — an energy VAD with the same endpointing shape as a real one
             (`negative_frames_count` consecutive silent frames ends the turn).
             It is not a stand-in for a vendor; it is the instrument we calibrate
             the scorers against, and it runs free and offline in CI.

  DeepgramWS — a second vendor, so the harness is not one provider's test rig.
             Its turn end is `speech_final`, and after a hesitation it arrives with an
             EMPTY transcript while the interim carried the words — the same
             partial-vs-final loss seen on Sarvam, on an unrelated stack. The text a
             turn is credited with therefore falls back to the last interim, which is
             what a live consumer holds at that instant.

  SarvamWS — the real lane. Streams into saaras:v3-realtime and reads
             `vad.speech_start` / `vad.speech_end` straight off the socket, which is
             why premature interruption is a timestamp comparison here and not an
             audio-onset estimation problem. Its endpointing gate is settable in
             milliseconds (`silence_duration_ms`, default 500), so that parameter is
             the sweep axis on this lane.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .spec import SAMPLE_RATE, CallSpec, Event, Result

FRAME_SAMPLES = 512  # Sarvam's frame: 32ms @16k, 64ms @8k


@dataclass
class MockASR:
    """Energy-gated VAD + endpointer. Deterministic, so PIR is analytically checkable.

    Mirrors the vendor knobs: a turn ends after `negative_frames_count` consecutive
    frames below threshold. Default 18 frames = 576ms @16k, 1152ms @8k.

    It emits a transcript at every turn end, not only at the end of the recording,
    because a split turn is the thing the conversation metric is about: the text an
    agent holds when it acts at the first end-of-turn is not the text the session
    ends up with. Turn text is attributed from `spec.seg_bounds_ms` — ground truth,
    which is the whole point of an instrument.
    """

    negative_frames_count: int = 18
    threshold: float = 0.02  # RMS, normalised to full scale
    rate: int = SAMPLE_RATE
    transcript_of: str = ""  # what it "hears"; set by the caller to inject ASR error
    confirms: bool = True  # does the downstream agent seek confirmation?
    name: str = "mock"

    def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        x = pcm.astype(np.float64) / 32768.0
        events: list[Event] = []
        silent = 0
        speaking = False
        for i in range(0, len(x) - FRAME_SAMPLES, FRAME_SAMPLES):
            t_ms = int(i * 1000 / self.rate)
            rms = float(np.sqrt(np.mean(x[i : i + FRAME_SAMPLES] ** 2)))
            if rms >= self.threshold:
                if not speaking:
                    events.append(Event("speech_start", t_ms))
                    speaking = True
                silent = 0
            elif speaking:
                silent += 1
                if silent >= self.negative_frames_count:
                    # endpoint fires when the silence *run* completes
                    events.append(Event("speech_end", t_ms + int(FRAME_SAMPLES * 1000 / self.rate)))
                    speaking = False
                    silent = 0
        if speaking:
            events.append(Event("speech_end", int(len(x) * 1000 / self.rate)))

        heard = self.transcript_of or spec.text
        if not self.transcript_of:
            events = self._with_turn_transcripts(events, spec)
        events.append(Event("transcript", events[-1].t_ms if events else 0, heard))
        return Result(spec_id=spec.id, adapter=self.name, events=events, transcript=heard)

    @staticmethod
    def _with_turn_transcripts(events: list[Event], spec: CallSpec) -> list[Event]:
        """One final per turn end, carrying the segments that had finished by then."""
        if not spec.seg_bounds_ms:
            return events
        out: list[Event] = []
        for e in events:
            out.append(e)
            if e.kind != "speech_end":
                continue
            said = " ".join(seg.text for (_, end), seg in zip(spec.seg_bounds_ms, spec.segments)
                            if end <= e.t_ms)
            if said:
                out.append(Event("transcript", e.t_ms, said))
        return out


class SarvamWS:
    """The real lane: saaras:v3-realtime over WebSocket.

    Endpointing is a wall-clock behaviour, so the audio is paced in real time — firing
    it in as fast as the socket accepts would tell us nothing about turn detection.

    Timestamps are taken as *audio sent so far* when an event arrives, not wall clock.
    Both include network and server latency, which inflates the apparent endpoint time
    and therefore makes PIR conservative: we under-report premature cuts rather than
    inventing them. Worth stating whenever the number is quoted.
    """

    name = "sarvam"
    URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    CHUNK_MS = 100

    def __init__(self, *, language_code: str = "hi-IN", model: str = "saaras:v3-realtime",
                 rate: int = SAMPLE_RATE, silence_duration_ms: int | None = None, **params):
        self.rate = rate
        self.params = {
            "language_code": language_code, "model": model, "sample_rate": str(rate),
            "encoding": "linear16", "endpointing": "vad", "return_timestamps": "true",
            **({"silence_duration_ms": str(silence_duration_ms)} if silence_duration_ms else {}),
            **{k: str(v) for k, v in params.items()},
        }

    async def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        import asyncio
        import base64

        import websockets

        qs = "&".join(f"{k}={v}" for k, v in self.params.items())
        events: list[Event] = []
        finals: list[str] = []
        sent_ms = 0  # audio position, the clock we timestamp against

        try:
            async with websockets.connect(
                f"{self.URL}?{qs}",
                additional_headers={"API-Subscription-Key": os.environ["SARVAM_API_KEY"]},
                max_size=None,
            ) as ws:

                async def pump() -> None:
                    nonlocal sent_ms
                    step = int(self.rate * self.CHUNK_MS / 1000)
                    for i in range(0, len(pcm), step):
                        chunk = pcm[i : i + step].astype("<i2").tobytes()
                        await ws.send(json.dumps({
                            "event": "audio_input",
                            "audio": base64.b64encode(chunk).decode(),
                        }))
                        sent_ms += int(len(chunk) / 2 * 1000 / self.rate)
                        await asyncio.sleep(self.CHUNK_MS / 1000)
                    await ws.send(json.dumps({"event": "end"}))

                sender = asyncio.create_task(pump())
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        ev = msg.get("event", "")
                        if ev == "vad.speech_start":
                            events.append(Event("speech_start", sent_ms))
                        elif ev == "vad.speech_end":
                            events.append(Event("speech_end", sent_ms))
                        elif ev == "transcript.final":
                            text = msg.get("text", "")
                            finals.append(text)
                            # start_s/end_s are audio-relative and immune to network
                            # jitter, so prefer them when the server sends them.
                            end_s = msg.get("end_s")
                            at = int(float(end_s) * 1000) if end_s else sent_ms
                            events.append(Event("transcript", at, text))
                        elif ev == "error":
                            return Result(spec_id=spec.id, adapter=self.name, events=events,
                                          transcript=" ".join(finals),
                                          error=f"{msg.get('code')}: {msg.get('message')}")
                        elif ev == "session.end":
                            break
                finally:
                    sender.cancel()
        except Exception as exc:  # surfaced per-call, never aborts a run
            return Result(spec_id=spec.id, adapter=self.name, events=events,
                          transcript=" ".join(finals), error=f"{type(exc).__name__}: {exc}")

        return Result(spec_id=spec.id, adapter=self.name, events=events,
                      transcript=" ".join(finals).strip())


class DeepgramWS:
    """Deepgram streaming. Same contract as SarvamWS, different wire.

    The endpointing gate is `endpointing` in milliseconds, so the PIR sweep axis means
    the same thing on both vendors even though the parameter is named differently.
    """

    name = "deepgram"
    URL = "wss://api.deepgram.com/v1/listen"
    CHUNK_MS = 100

    def __init__(self, *, language_code: str = "hi", model: str = "nova-2",
                 rate: int = SAMPLE_RATE, silence_duration_ms: int | None = None, **params):
        self.rate = rate
        gate = silence_duration_ms or 500
        self.params = {
            "model": model, "language": language_code.split("-")[0],
            "encoding": "linear16", "sample_rate": str(rate), "channels": "1",
            "interim_results": "true", "endpointing": str(gate),
            "utterance_end_ms": str(max(1000, gate * 2)),
            **{k: str(v) for k, v in params.items()},
        }

    async def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        import asyncio

        import websockets

        qs = "&".join(f"{k}={v}" for k, v in self.params.items())
        events: list[Event] = []
        finals: list[str] = []
        sent_ms = 0
        interim = ""

        try:
            async with websockets.connect(
                f"{self.URL}?{qs}",
                additional_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
                max_size=None,
            ) as ws:

                async def pump() -> None:
                    nonlocal sent_ms
                    step = int(self.rate * self.CHUNK_MS / 1000)
                    for i in range(0, len(pcm), step):
                        await ws.send(pcm[i : i + step].astype("<i2").tobytes())
                        sent_ms += self.CHUNK_MS
                        await asyncio.sleep(self.CHUNK_MS / 1000)
                    await ws.send(json.dumps({"type": "CloseStream"}))

                sender = asyncio.create_task(pump())
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "Results":
                            alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
                            text = alt.get("transcript", "")
                            if msg.get("speech_final"):
                                events.append(Event("speech_end", sent_ms))
                                # an empty final after a hesitation is the finding, not
                                # an absence of speech — credit the interim
                                said = text or interim
                                finals.append(said)
                                events.append(Event("transcript", sent_ms, said))
                                interim = ""
                            elif text:
                                interim = text
                        elif msg.get("type") == "Metadata":
                            break
                        elif msg.get("type") == "Error" or msg.get("error"):
                            return Result(spec_id=spec.id, adapter=self.name, events=events,
                                          transcript=" ".join(finals),
                                          error=str(msg.get("description") or msg.get("error")))
                finally:
                    sender.cancel()
        except Exception as exc:
            return Result(spec_id=spec.id, adapter=self.name, events=events,
                          transcript=" ".join(finals), error=f"{type(exc).__name__}: {exc}")

        return Result(spec_id=spec.id, adapter=self.name, events=events,
                      transcript=" ".join(finals).strip())

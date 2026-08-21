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


class OpenAIWS:
    """OpenAI realtime transcription with explicit turn detection.

    `server_vad` is a silence timer, so `silence_duration_ms` means the same thing here
    as on the other lanes and the sweep axis is comparable across all of them.

    Both modes use a general Realtime session. OpenAI's transcription-only session
    supports server VAD but not semantic VAD, while a general session supports both
    and can still emit input-audio transcription events.

    Audio is resampled to 24kHz because the endpoint rejects 16k, and a wrong rate
    silently distorts every timestamp.
    """

    name = "openai"
    URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    CHUNK_MS = 100
    NATIVE_RATE = 24000
    OPEN_TIMEOUT_S = 20
    WRITE_TIMEOUT_S = 15
    CLOSE_TIMEOUT_S = 5

    def __init__(self, *, language_code: str = "hi", model: str = "gpt-4o-transcribe",
                 rate: int = SAMPLE_RATE, silence_duration_ms: int | None = None,
                 turn_detection: str = "server_vad", trailing_silence_ms: int = 2500,
                 require_endpoint_timestamps: bool = False, **params):
        self.rate, self.model = rate, model
        self.lang = language_code.split("-")[0]
        self.gate = silence_duration_ms or 500
        if turn_detection not in {"server_vad", "semantic_vad"}:
            raise ValueError(f"unsupported OpenAI turn detection: {turn_detection}")
        if (isinstance(trailing_silence_ms, bool) or
                not isinstance(trailing_silence_ms, int) or trailing_silence_ms < 0):
            raise ValueError("trailing_silence_ms must be a non-negative integer")
        self.turn_detection = turn_detection
        self.trailing_silence_ms = trailing_silence_ms
        self.require_endpoint_timestamps = require_endpoint_timestamps
        self.missing_endpoint_timestamp = False

    @staticmethod
    def _endpoint_timestamp(message: dict[str, object]) -> int | None:
        timestamp = message.get("audio_end_ms")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            return None
        return timestamp

    def _speech_stopped_timestamp(
        self, message: dict[str, object], *, sent_ms: int
    ) -> int | None:
        """Keep legacy sent-audio timing unless strict observation requests provider time."""
        if self.require_endpoint_timestamps:
            return self._endpoint_timestamp(message)
        return sent_ms

    def turn_detection_payload(self) -> dict[str, object]:
        """Return the provider payload without exposing a semantic mode as a timer."""
        if self.turn_detection == "semantic_vad":
            return {"type": "semantic_vad"}
        return {"type": "server_vad", "silence_duration_ms": self.gate}

    def session_update_payload(self) -> dict[str, object]:
        """Build the documented general-Realtime setup for either turn detector."""
        turn_detection = {**self.turn_detection_payload(), "create_response": False}
        return {"type": "session.update", "session": {
            "type": "realtime",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": self.NATIVE_RATE},
                "transcription": {"model": self.model, "language": self.lang},
                "turn_detection": turn_detection,
            }},
        }}

    def _to_native(self, pcm: np.ndarray) -> np.ndarray:
        if self.rate == self.NATIVE_RATE:
            return pcm
        ratio = self.NATIVE_RATE / self.rate
        n = int(len(pcm) * ratio)
        idx = np.arange(n) / ratio
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(pcm) - 1)
        frac = idx - lo
        return (pcm[lo] * (1 - frac) + pcm[hi] * frac).astype(np.int16)

    async def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        import asyncio
        import base64
        from contextlib import suppress

        import websockets

        audio = self._to_native(pcm)
        if self.trailing_silence_ms:
            audio = np.concatenate([
                audio,
                np.zeros(self.NATIVE_RATE * self.trailing_silence_ms // 1000, np.int16),
            ])
        events: list[Event] = []
        finals: list[str] = []
        sent_ms = 0
        self.missing_endpoint_timestamp = False
        try:
            async with websockets.connect(
                self.URL, additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                max_size=None, open_timeout=self.OPEN_TIMEOUT_S, close_timeout=self.CLOSE_TIMEOUT_S,
            ) as ws:
                await asyncio.wait_for(ws.send(json.dumps(self.session_update_payload())),
                                       timeout=self.WRITE_TIMEOUT_S)

                async def pump() -> None:
                    nonlocal sent_ms
                    step = int(self.NATIVE_RATE * self.CHUNK_MS / 1000)
                    for i in range(0, len(audio), step):
                        await asyncio.wait_for(ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                audio[i : i + step].astype("<i2").tobytes()).decode()})),
                            timeout=self.WRITE_TIMEOUT_S)
                        sent_ms += self.CHUNK_MS
                        await asyncio.sleep(self.CHUNK_MS / 1000)

                sender = asyncio.create_task(pump())

                async def reader() -> None:
                    async for raw in ws:
                        msg = json.loads(raw)
                        ty = msg.get("type", "")
                        if "speech_stopped" in ty:
                            endpoint_ms = self._speech_stopped_timestamp(msg, sent_ms=sent_ms)
                            if endpoint_ms is None:
                                if self.require_endpoint_timestamps:
                                    self.missing_endpoint_timestamp = True
                                else:
                                    events.append(Event("speech_end", sent_ms))
                            else:
                                events.append(Event("speech_end", endpoint_ms))
                        elif "transcription" in ty and "completed" in ty:
                            text = msg.get("transcript", "")
                            finals.append(text)
                            events.append(Event("transcript", sent_ms, text))
                            if finals and not self.require_endpoint_timestamps:
                                return
                        elif ty == "error":
                            raise RuntimeError(str(msg.get("error", {}))[:200])

                try:
                    await asyncio.wait_for(reader(),
                                           timeout=len(audio) / self.NATIVE_RATE + 20)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if self.require_endpoint_timestamps:
                        await sender
                    else:
                        sender.cancel()
                        with suppress(asyncio.CancelledError):
                            await sender
        except Exception as exc:
            return Result(spec_id=spec.id, adapter=self.name, events=events,
                          transcript=" ".join(finals), error=f"{type(exc).__name__}: {exc}")
        return Result(spec_id=spec.id, adapter=self.name, events=events,
                      transcript=" ".join(finals).strip())


class GeminiLive:
    """Gemini Live, automatic activity detection.

    `silenceDurationMs` is the same knob the other lanes sweep, so the axis is
    comparable. The turn-end EVENT is not: Gemini Live emits no explicit VAD-stop
    message, so `speech_end` here is inferred from the first chunk of the model's
    reply — the moment an agent built on this would start talking over the caller.

    That inference includes model response latency, so it lands LATER than the true
    turn-end decision and makes a cut look less early than it was. The bias runs
    against finding prematurity, the same direction as the socket-timestamp caveat on
    the other lanes. Read the Gemini row as a lower bound, and never as an equal-footing
    comparison with the lanes that report their own VAD.

    Two API traps, both silent: the live models reject a TEXT response modality, and
    audio sent as `realtimeInput.mediaChunks` is accepted and then ignored — the socket
    stays open and simply never answers. It has to be `realtimeInput.audio`.
    """

    name = "gemini"
    URL = ("wss://generativelanguage.googleapis.com/ws/"
           "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
    CHUNK_MS = 100
    NATIVE_RATE = 16000

    def __init__(self, *, language_code: str = "hi",
                 model: str = "models/gemini-3.1-flash-live-preview",
                 rate: int = SAMPLE_RATE, silence_duration_ms: int | None = None, **params):
        self.rate, self.model = rate, model
        self.lang = language_code.split("-")[0]
        self.gate = silence_duration_ms or 500

    async def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        import asyncio
        import base64

        import websockets

        if self.rate != self.NATIVE_RATE:
            return Result(spec_id=spec.id, adapter=self.name,
                          error=f"gemini lane needs {self.NATIVE_RATE} Hz, got {self.rate}")
        audio = np.concatenate([pcm, np.zeros(int(self.NATIVE_RATE * 2.5), np.int16)])
        events: list[Event] = []
        heard: list[str] = []
        sent_ms = 0
        try:
            async with websockets.connect(
                f"{self.URL}?key={os.environ['GEMINI_API_KEY']}", max_size=None,
            ) as ws:
                await ws.send(json.dumps({"setup": {
                    "model": self.model,
                    "generationConfig": {"responseModalities": ["AUDIO"]},
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {"silenceDurationMs": self.gate}},
                    "inputAudioTranscription": {}}}))
                await asyncio.wait_for(ws.recv(), timeout=20)  # setupComplete

                async def pump() -> None:
                    nonlocal sent_ms
                    step = int(self.NATIVE_RATE * self.CHUNK_MS / 1000)
                    for i in range(0, len(audio), step):
                        await ws.send(json.dumps({"realtimeInput": {"audio": {
                            "mimeType": f"audio/pcm;rate={self.NATIVE_RATE}",
                            "data": base64.b64encode(
                                audio[i : i + step].astype("<i2").tobytes()).decode()}}}))
                        sent_ms += self.CHUNK_MS
                        await asyncio.sleep(self.CHUNK_MS / 1000)
                    await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))

                sender = asyncio.create_task(pump())

                async def reader() -> None:
                    replied = False
                    async for raw in ws:
                        content = json.loads(raw).get("serverContent")
                        if not isinstance(content, dict):
                            continue
                        if text := content.get("inputTranscription", {}).get("text", ""):
                            heard.append(text)
                            events.append(Event("transcript", sent_ms, text))
                        if "modelTurn" in content and not replied:
                            replied = True  # inferred turn end — see the class docstring
                            events.append(Event("speech_end", sent_ms))
                        if content.get("turnComplete"):
                            return

                try:
                    await asyncio.wait_for(reader(),
                                           timeout=len(audio) / self.NATIVE_RATE + 20)
                except asyncio.TimeoutError:
                    pass
                finally:
                    sender.cancel()
        except Exception as exc:
            return Result(spec_id=spec.id, adapter=self.name, events=events,
                          transcript=" ".join(heard), error=f"{type(exc).__name__}: {exc}")
        return Result(spec_id=spec.id, adapter=self.name, events=events,
                      transcript=" ".join(heard).strip())

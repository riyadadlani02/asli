"""Adapters: stream audio at a system under test, capture timestamped events.

An adapter is anything with `run(pcm, spec) -> Result`. Two ship here:

  MockASR  — an energy VAD with the same endpointing shape as a real one
             (`negative_frames_count` consecutive silent frames ends the turn).
             It is not a stand-in for a vendor; it is the instrument we calibrate
             the scorers against, and it runs free and offline in CI.

  SarvamWS — the real lane. Streams into saaras:v3 with `vad_signals=true` and
             reads `speech_start`/`speech_end` straight off the socket, which is
             why premature interruption is a timestamp comparison here and not an
             audio-onset estimation problem.
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
        events.append(Event("transcript", events[-1].t_ms if events else 0, heard))
        return Result(spec_id=spec.id, adapter=self.name, events=events, transcript=heard)


class SarvamWS:
    """Real lane. Requires SARVAM_API_KEY.

    Endpointing knobs are passed straight through so PIR can be swept over them —
    that sweep is the point of the harness, not a side feature.
    """

    name = "sarvam"
    URL = "wss://api.sarvam.ai/speech-to-text/ws"

    def __init__(self, *, language_code: str = "hi-IN", model: str = "saaras:v3",
                 rate: int = SAMPLE_RATE, **vad):
        self.language_code, self.model, self.rate, self.vad = language_code, model, rate, vad

    async def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        import time

        import websockets

        params = {
            "language-code": self.language_code, "model": self.model,
            "sample-rate": str(self.rate), "vad_signals": "true",
            **{k.replace("_", "-"): str(v) for k, v in self.vad.items()},
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        events: list[Event] = []
        transcript: list[str] = []
        t0 = time.monotonic()

        async with websockets.connect(
            f"{self.URL}?{qs}", additional_headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]}
        ) as ws:
            async def send() -> None:
                # real time pacing: endpointing is a wall-clock behaviour, so we
                # must not fire the audio in as fast as the socket will take it.
                import asyncio

                step = FRAME_SAMPLES * 2
                for i in range(0, len(pcm), step):
                    await ws.send(json.dumps({
                        "audio": {"data": pcm[i : i + step].astype("<i2").tobytes().hex(),
                                  "encoding": "audio/wav", "sample_rate": self.rate}}))
                    await asyncio.sleep(step / self.rate)

            import asyncio

            pump = asyncio.create_task(send())
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    t_ms = int((time.monotonic() - t0) * 1000)
                    kind = msg.get("type")
                    if kind == "events":
                        sig = str(msg.get("data", {}).get("signal", "")).lower()
                        if "start" in sig:
                            events.append(Event("speech_start", t_ms))
                        elif "end" in sig:
                            events.append(Event("speech_end", t_ms))
                    elif kind == "data":
                        text = msg.get("data", {}).get("transcript", "")
                        if text:
                            transcript.append(text)
                            events.append(Event("transcript", t_ms, text))
                    if pump.done() and transcript:
                        break
            finally:
                pump.cancel()

        return Result(spec_id=spec.id, adapter=self.name, events=events,
                      transcript=" ".join(transcript))

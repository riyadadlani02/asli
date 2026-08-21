import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np

from asli.cli import make_adapter, real_gates, real_result_stem
from asli.drive import OpenAIWS
from asli.spec import CallSpec
from asli.turnbench.auto_label import OpenAISemanticObserver


def test_openai_server_vad_keeps_the_configured_gate():
    adapter = OpenAIWS(silence_duration_ms=700)

    assert adapter.turn_detection_payload() == {
        "type": "server_vad",
        "silence_duration_ms": 700,
    }


def test_openai_semantic_vad_has_no_silence_timer():
    adapter = OpenAIWS(turn_detection="semantic_vad", silence_duration_ms=700)

    assert adapter.turn_detection_payload() == {"type": "semantic_vad"}


def test_openai_semantic_vad_uses_a_general_realtime_session_not_transcription_only():
    adapter = OpenAIWS(turn_detection="semantic_vad", silence_duration_ms=700)

    payload = adapter.session_update_payload()

    assert adapter.URL == "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    assert payload["session"]["type"] == "realtime"
    assert payload["session"]["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "create_response": False,
    }


def test_make_adapter_forwards_openai_turn_detection_choice():
    adapter = make_adapter(
        "openai",
        gate=700,
        lang="hi-IN",
        rate=16000,
        turn_detection="semantic_vad",
    )

    assert adapter.turn_detection_payload() == {"type": "semantic_vad"}


def test_semantic_real_caller_results_cannot_share_the_silence_timer_filename():
    assert real_result_stem("openai", "semantic_vad", silence_pause=False) == (
        "real_pir_openai_semantic_vad"
    )
    assert real_result_stem("openai", "server_vad", silence_pause=True) == (
        "real_pir_openai_server_vad_silenced"
    )


def test_semantic_real_caller_run_has_no_silence_gate_or_sweep_axis():
    assert real_gates("openai", "semantic_vad", gate=500, gates="500,900") == [None]


def test_openai_default_trailing_silence_remains_legacy_value():
    adapter = OpenAIWS()

    assert adapter.trailing_silence_ms == 2500
    assert adapter.require_endpoint_timestamps is False


def test_strict_semantic_observer_uses_no_tail_and_provider_timestamps():
    observer = OpenAISemanticObserver(model="gpt-4o-transcribe")

    assert observer.adapter.turn_detection == "semantic_vad"
    assert observer.adapter.trailing_silence_ms == 0
    assert observer.adapter.require_endpoint_timestamps is True


def test_openai_strict_timestamp_parser_rejects_missing_bool_and_float_values():
    adapter = OpenAIWS(require_endpoint_timestamps=True)

    assert adapter._endpoint_timestamp({"audio_end_ms": 1234}) == 1234
    assert adapter._endpoint_timestamp({"audio_end_ms": None}) is None
    assert adapter._endpoint_timestamp({"audio_end_ms": True}) is None
    assert adapter._endpoint_timestamp({"audio_end_ms": 12.5}) is None


def test_openai_default_keeps_legacy_sent_audio_timestamp():
    adapter = OpenAIWS()

    assert adapter._speech_stopped_timestamp({"audio_end_ms": 1234}, sent_ms=900) == 900
    assert OpenAIWS(require_endpoint_timestamps=True)._speech_stopped_timestamp(
        {"audio_end_ms": 1234}, sent_ms=900
    ) == 1234


def test_openai_strict_mode_finishes_bounded_audio_after_transcription_completion(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

        def __aiter__(self):
            return self.messages()

        async def messages(self):
            yield json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "x"})
            await asyncio.sleep(0.01)
            yield json.dumps({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 2})

    class FakeConnection:
        def __init__(self, socket):
            self.socket = socket

        async def __aenter__(self):
            return self.socket

        async def __aexit__(self, *args):
            return None

    socket = FakeSocket()
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda *args, **kwargs: FakeConnection(socket)))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    adapter = OpenAIWS(rate=24000, trailing_silence_ms=0, require_endpoint_timestamps=True)
    adapter.CHUNK_MS = 1

    result = asyncio.run(adapter.run(np.zeros(48, dtype=np.int16), CallSpec("d", [], "x", "")))

    audio_messages = [message for message in socket.sent if message["type"] == "input_audio_buffer.append"]
    assert len(audio_messages) == 2
    assert [event.t_ms for event in result.events if event.kind == "speech_end"] == [2]

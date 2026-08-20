from asli.cli import make_adapter, real_gates, real_result_stem
from asli.drive import OpenAIWS


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

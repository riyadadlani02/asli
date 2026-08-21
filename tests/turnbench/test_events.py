import pytest

from asli.turnbench.events import first_event_ms, trace_availability
from asli.turnbench.schema import ProviderTrace, TraceEvent


def _trace(*events, status="ok"):
    return ProviderTrace(
        run_id="run1",
        decision_id="d1",
        provider="fixture",
        status=status,
        error=None,
        events=tuple(events),
    )


@pytest.fixture
def trace_with_two_audio_events():
    return _trace(
        TraceEvent("speech_started", 1000),
        TraceEvent("agent_first_audio", 1300),
        TraceEvent("agent_first_audio", 1500),
    )


@pytest.fixture
def failed_trace():
    return _trace(status="failed")


@pytest.fixture
def timeout_trace():
    return _trace(status="timeout")


@pytest.fixture
def ok_trace_without_audio():
    return _trace(TraceEvent("turn_committed", 1400))


@pytest.fixture
def trace_only_audio():
    return _trace(TraceEvent("agent_first_audio", 1300))


def test_first_event_uses_first_audible_agent_event(trace_with_two_audio_events):
    assert first_event_ms(trace_with_two_audio_events, "agent_first_audio") == 1300


def test_failed_trace_is_unavailable(failed_trace):
    assert trace_availability(failed_trace) == "provider_failed"


def test_ok_trace_missing_agent_audio_is_unavailable(ok_trace_without_audio):
    assert trace_availability(ok_trace_without_audio) == "missing_agent_first_audio"


def test_turn_committed_is_not_required_for_audible_score(trace_only_audio):
    assert first_event_ms(trace_only_audio, "turn_committed") is None
    assert trace_availability(trace_only_audio) == "available"


def test_timeout_is_distinct_from_provider_failure(timeout_trace):
    assert trace_availability(timeout_trace) == "provider_timeout"

import numpy as np
import pytest

from asli.spec import Result
from asli.turnbench.auto_label import (
    EndpointObservation,
    OpenAISemanticObserver,
    predict_candidate,
)
from asli.turnbench.auto_schema import DiarBenchCandidate


@pytest.fixture
def candidate():
    return DiarBenchCandidate(
        decision_id="d1", recording_id="r1", source_recording_id="source-1",
        audio_path="fixture.wav", language="hi", condition="diarbench",
        target_speaker_id="speaker-1", context_start_ms=500,
        previous_speech_end_ms=1000, observation_end_ms=2000,
    )


def test_predict_candidate_slices_only_through_observation_end(candidate):
    seen = {}

    def read_audio(path):
        assert path == "fixture.wav"
        return np.arange(10_000, dtype=np.int16), 1000

    def observe(pcm, rate, language):
        seen["samples"] = len(pcm)
        assert rate == 1000
        assert language == "hi"
        return EndpointObservation(endpoint_ms=None, missing_timestamp=False, error=None)

    prediction = predict_candidate(
        candidate, read_audio=read_audio, observe=observe, run_id="r1",
        agent="openai", model="m", config={"context_ms": 500},
    )

    assert seen["samples"] == candidate.observation_end_ms - candidate.context_start_ms
    assert prediction.status == "available"
    assert prediction.outcome == "continue"
    assert prediction.endpoint_ms is None


def test_predict_candidate_converts_in_window_provider_endpoint_to_absolute_yield(candidate):
    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(10_000, dtype=np.int16), 1000),
        observe=lambda *_: EndpointObservation(endpoint_ms=700, missing_timestamp=False, error=None),
        run_id="r1", agent="openai", model="m", config={},
    )

    assert prediction.status == "available"
    assert prediction.outcome == "yield"
    assert prediction.endpoint_ms == 1200


def test_predict_candidate_ignores_context_endpoint_and_selects_first_decision_endpoint(candidate):
    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(10_000, dtype=np.int16), 1000),
        observe=lambda *_: EndpointObservation(
            endpoint_ms=300,
            missing_timestamp=False,
            error=None,
            endpoint_timestamps=(300, 700, 900),
        ),
        run_id="r1", agent="openai", model="m", config={},
    )

    assert prediction.status == "available"
    assert prediction.outcome == "yield"
    assert prediction.endpoint_ms == 1200


@pytest.mark.parametrize(
    ("timestamps", "reason"),
    [
        ((500,), "endpoint_outside_window"),
        ((-1,), "invalid_endpoint_timestamp"),
        ((1500,), "endpoint_outside_window"),
    ],
)
def test_predict_candidate_rejects_strict_endpoint_timestamps_outside_open_window(
    candidate, timestamps, reason
):
    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(10_000, dtype=np.int16), 1000),
        observe=lambda *_: EndpointObservation(
            endpoint_ms=timestamps[0],
            missing_timestamp=False,
            error=None,
            endpoint_timestamps=timestamps,
        ),
        run_id="r1", agent="openai", model="m", config={},
    )

    assert prediction.status == "unavailable"
    assert prediction.outcome is None
    assert prediction.unavailable_reason == reason


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (EndpointObservation(endpoint_ms=500, missing_timestamp=False, error=None), "endpoint_outside_window"),
        (EndpointObservation(endpoint_ms=None, missing_timestamp=True, error=None), "missing_endpoint_timestamp"),
        (EndpointObservation(endpoint_ms=None, missing_timestamp=False, error="timeout"), "observer_error"),
    ],
)
def test_predict_candidate_marks_unusable_observations_unavailable(candidate, observation, reason):
    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(10_000, dtype=np.int16), 1000),
        observe=lambda *_: observation,
        run_id="r1", agent="openai", model="m", config={},
    )

    assert prediction.status == "unavailable"
    assert prediction.outcome is None
    assert prediction.endpoint_ms is None
    assert prediction.unavailable_reason == reason


def test_predict_candidate_rejects_audio_that_cannot_reach_observation_boundary(candidate):
    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(1500, dtype=np.int16), 1000),
        observe=lambda *_: pytest.fail("observer must not receive incomplete audio"),
        run_id="r1", agent="openai", model="m", config={},
    )

    assert prediction.status == "unavailable"
    assert prediction.unavailable_reason == "audio_boundary_error"


def test_predict_candidate_normalizes_config_with_report_rules(candidate):
    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(10_000, dtype=np.int16), 1000),
        observe=lambda *_: EndpointObservation(None, False, None),
        run_id="r1", agent="openai", model="m",
        config={"nested": (1, {"value": 2.5})},
    )

    assert prediction.config == {"nested": [1, {"value": 2.5}]}


def test_predict_candidate_can_omit_the_provider_language_hint(candidate):
    seen = {}

    prediction = predict_candidate(
        candidate,
        read_audio=lambda _: (np.zeros(10_000, dtype=np.int16), 1000),
        observe=lambda _pcm, _rate, language: (
            seen.setdefault("language", language) or EndpointObservation(None, False, None)
        ),
        run_id="r1", agent="openai", model="m", config={}, provider_language=None,
    )

    assert seen["language"] is None
    assert prediction.outcome == "continue"


def test_semantic_observer_sets_adapter_language_for_each_candidate_window():
    class FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.lang = "hi"
            self.rate = 0
            self.missing_endpoint_timestamp = False
            self.session_payload = None

        def session_update_payload(self):
            return {"language": self.lang}

        async def run(self, pcm, spec):
            self.session_payload = self.session_update_payload()
            return Result(spec_id=spec.id, adapter="fake")

    observer = OpenAISemanticObserver(adapter_factory=FakeAdapter)

    observation = observer(np.zeros(100, dtype=np.int16), 1000, "ta-IN")

    assert observation == EndpointObservation(None, False, None)
    assert observer.adapter.session_payload == {"language": "ta"}
    assert observer.adapter.rate == 1000

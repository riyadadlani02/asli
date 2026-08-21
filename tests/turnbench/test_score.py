from dataclasses import replace
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from asli.turnbench import (
    Annotation,
    DecisionLabel,
    ProviderTrace,
    Recording,
    SchemaError,
    SourceTurn,
    TraceEvent,
    read_jsonl,
)
from asli.turnbench.score import (
    DecisionScore,
    aggregate,
    bootstrap_by_recording,
    score_decision,
)
from asli.turnbench.report import score_inputs


FIXTURES = Path(__file__).parents[2] / "turnbench" / "fixtures"


@pytest.fixture
def recordings():
    return {
        row.recording_id: row
        for row in read_jsonl(FIXTURES / "recordings.jsonl", Recording.from_dict)
    }


@pytest.fixture
def labels():
    return {
        row.decision_id: row
        for row in read_jsonl(FIXTURES / "labels.jsonl", DecisionLabel.from_dict)
    }


@pytest.fixture
def traces():
    return {
        row.decision_id: row
        for row in read_jsonl(FIXTURES / "events.jsonl", ProviderTrace.from_dict)
    }


@pytest.fixture
def scores(labels, traces):
    return [score_decision(label, traces[label.decision_id]) for label in labels.values()]


def test_continue_with_early_agent_audio_is_interruption(labels, traces):
    result = score_decision(labels["hi-continue"], traces["hi-continue"])
    assert result.status == "scored"
    assert result.interrupted is True
    assert result.response_delay_ms is None


def test_yield_contributes_delay_not_pir(labels, traces):
    result = score_decision(labels["ta-yield"], traces["ta-yield"])
    assert result.interrupted is None
    assert result.response_delay_ms == 300


def test_overlap_and_unclear_do_not_enter_headline_denominators(scores):
    summary = aggregate(scores)
    assert summary["pir_n"] == 1
    assert summary["delay_n"] == 1
    assert summary["excluded_n"] == 2


def test_timeout_is_reported_not_counted_as_a_safe_continue(scores):
    summary = aggregate(scores)
    assert summary["unavailable_n"] == 1
    assert summary["provider_timeout_n"] == 1
    assert summary["trace_n"] == 5
    assert summary["available_n"] == 4
    assert summary["availability_rate"] == pytest.approx(4 / 5)
    assert summary["provider_timeout_rate"] == pytest.approx(1 / 5)
    assert summary["provider_failed_rate"] == 0
    assert summary["missing_agent_first_audio_rate"] == 0


@pytest.mark.parametrize(
    ("decision_id", "status", "expected_reason"),
    [
        ("hi-overlap", "timeout", "provider_timeout"),
        ("ta-unclear", "failed", "provider_failed"),
        ("hi-overlap", "ok", "missing_agent_first_audio"),
    ],
)
def test_excluded_labels_preserve_provider_availability(
    labels, traces, decision_id, status, expected_reason
):
    trace = replace(traces[decision_id], status=status, events=())
    result = score_decision(labels[decision_id], trace)
    summary = aggregate([result])
    assert result.status == "excluded"
    assert result.exclusion_reason == labels[decision_id].exclusion_reason
    assert result.unavailable_reason == expected_reason
    assert summary["excluded_n"] == 1
    assert summary["unavailable_n"] == 1
    assert summary[f"{expected_reason}_n"] == 1
    assert summary["trace_n"] == 1
    assert summary["available_n"] == 0
    assert summary["availability_rate"] == 0
    assert summary[f"{expected_reason}_rate"] == 1


def test_report_separates_language_and_source_groups(labels, recordings, traces):
    report = score_inputs(
        recordings.values(),
        labels.values(),
        traces.values(),
        provider="fixture",
        config={},
    )
    assert set(report["by_language"]) == {"hi", "ta"}
    assert set(report["by_source_recording"]) == {"hi-source-1", "ta-source-2"}
    assert report["run_id"] == "fixture-run"
    assert report["overall"]["aggregation"] == "macro_by_language"
    assert report["micro_overall"]["aggregation"] == "micro_by_decision"
    for grouped in (
        *report["by_language"].values(),
        *report["by_source_recording"].values(),
        *report["by_condition"].values(),
    ):
        assert grouped["aggregation"] == "micro_by_decision"


def test_report_macro_language_headline_differs_from_imbalanced_micro(
    labels, recordings, traces
):
    synthetic_recordings = dict(recordings)
    synthetic_recordings["ta-macro-fixture"] = replace(
        recordings["hi-fixture-1"],
        recording_id="ta-macro-fixture",
        audio_path="fixtures/ta-macro-fixture.wav",
        language="ta",
        source_recording_id="ta-source-2",
    )
    synthetic_labels = {}
    synthetic_traces = []
    for index in range(10):
        decision_id = f"hi-safe-{index}"
        synthetic_labels[decision_id] = replace(
            labels["hi-continue"], decision_id=decision_id
        )
        synthetic_traces.append(
            replace(
                traces["hi-continue"],
                decision_id=decision_id,
                events=(TraceEvent("agent_first_audio", 1700),),
            )
        )
    synthetic_labels["ta-interrupted"] = replace(
        labels["hi-continue"],
        decision_id="ta-interrupted",
        recording_id="ta-macro-fixture",
        source_recording_id="ta-source-2",
    )
    synthetic_traces.append(
        replace(
            traces["hi-continue"],
            decision_id="ta-interrupted",
            events=(TraceEvent("agent_first_audio", 1300),),
        )
    )

    report = score_inputs(
        synthetic_recordings.values(),
        synthetic_labels.values(),
        synthetic_traces,
        provider="fixture",
        config={},
    )

    assert report["by_language"]["hi"]["pir"] == 0
    assert report["by_language"]["ta"]["pir"] == 1
    assert report["micro_overall"]["pir"] == pytest.approx(1 / 11)
    assert report["overall"]["pir"] == pytest.approx(0.5)
    assert report["overall"]["pir_language_n"] == 2
    assert report["micro_overall"]["pir"] != report["overall"]["pir"]
    traces_by_decision = {trace.decision_id: trace for trace in synthetic_traces}
    synthetic_scores = [
        score_decision(label, traces_by_decision[decision_id])
        for decision_id, label in synthetic_labels.items()
    ]
    expected_micro_interval = bootstrap_by_recording(
        synthetic_scores, synthetic_labels, draws=1000, seed=0
    )
    assert expected_micro_interval is not None
    assert report["pir_bootstrap_95"] == {
        "aggregation": "micro_by_decision",
        "metric": "pir",
        **expected_micro_interval,
    }


def test_macro_denominators_ignore_languages_without_eligible_metric(
    labels, recordings, traces
):
    eligible_ids = {"hi-continue", "ta-yield"}
    recording_ids = {
        labels[decision_id].recording_id for decision_id in eligible_ids
    }
    report = score_inputs(
        [
            recording
            for recording in recordings.values()
            if recording.recording_id in recording_ids
        ],
        [label for label in labels.values() if label.decision_id in eligible_ids],
        [trace for trace in traces.values() if trace.decision_id in eligible_ids],
        provider="fixture",
        config={},
    )

    assert report["by_language"]["hi"]["pir"] == 1
    assert report["by_language"]["hi"]["delay_p50_ms"] is None
    assert report["by_language"]["ta"]["pir"] is None
    assert report["by_language"]["ta"]["delay_p50_ms"] == 300
    assert report["overall"]["pir"] == 1
    assert report["overall"]["pir_language_n"] == 1
    assert report["overall"]["delay_p50_ms"] == 300
    assert report["overall"]["delay_p50_ms_language_n"] == 1
    assert report["overall"]["delay_p95_ms_language_n"] == 1


def test_empty_report_has_null_run_and_undefined_rates():
    report = score_inputs([], [], [], provider="fixture", config={})

    assert report["run_id"] is None
    assert report["label_provenance"] == {
        "fixture_label_n": 0,
        "adjudicated_label_n": 0,
    }
    assert report["by_language"] == {}
    assert report["by_condition"] == {}
    assert report["by_source_recording"] == {}
    assert report["micro_overall"]["trace_n"] == 0
    assert report["micro_overall"]["availability_rate"] is None
    assert report["overall"]["availability_rate"] is None
    assert report["overall"]["availability_rate_language_n"] == 0
    assert report["pir_bootstrap_95"] == {
        "aggregation": "micro_by_decision",
        "metric": "pir",
        "low": None,
        "high": None,
    }


def test_empty_report_normalizes_json_safe_config():
    report = score_inputs(
        [],
        [],
        [],
        provider="fixture",
        config={
            "sequence": (1, True, None, {"nested": ("value", 2.5)}),
            "list": [False, {"number": 3}],
        },
    )

    assert report["config"] == {
        "sequence": [1, True, None, {"nested": ["value", 2.5]}],
        "list": [False, {"number": 3}],
    }


@pytest.mark.parametrize(
    ("provider", "config", "message"),
    [
        ("", {}, "provider must be a non-empty string"),
        ("   ", {}, "provider must be a non-empty string"),
        ("fixture", {"value": float("nan")}, "finite"),
        ("fixture", {"value": float("inf")}, "finite"),
        ("fixture", {"value": object()}, "standard JSON value"),
        ("fixture", {1: "value"}, "string keys"),
        ("fixture", [], "config must be a mapping"),
    ],
)
def test_score_inputs_rejects_invalid_report_provenance(
    provider, config, message
):
    with pytest.raises(ValueError, match=message):
        score_inputs([], [], [], provider=provider, config=config)


def test_invalid_config_is_rejected_before_report_construction(monkeypatch):
    import asli.turnbench.report as report_module

    called = False

    def unexpected_report(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(report_module, "_build_report", unexpected_report)

    with pytest.raises(ValueError, match="standard JSON value"):
        report_module.score_inputs(
            [], [], [], provider="fixture", config={"bad": object()}
        )
    assert called is False


def test_one_source_group_has_no_bootstrap_interval(scores, labels):
    single_scores = [score for score in scores if score.decision_id == "hi-continue"]
    assert bootstrap_by_recording(single_scores, labels, draws=20, seed=7) is None


def test_bootstrap_bounds_are_independent_of_label_input_order(
    recordings, labels, traces
):
    base_recording = recordings["hi-fixture-1"]
    base_label = labels["hi-continue"]
    base_trace = traces["hi-continue"]
    specifications = (
        ("a-safe", "recording-a", "source-a", 1700),
        ("b-interrupted", "recording-b", "source-b", 1300),
        ("c-safe-1", "recording-c", "source-c", 1700),
        ("c-safe-2", "recording-c", "source-c", 1700),
    )
    synthetic_recordings = {
        recording_id: replace(
            base_recording,
            recording_id=recording_id,
            source_recording_id=source_id,
        )
        for _, recording_id, source_id, _ in specifications
    }
    synthetic_labels = {
        decision_id: replace(
            base_label,
            decision_id=decision_id,
            recording_id=recording_id,
            source_recording_id=source_id,
        )
        for decision_id, recording_id, source_id, _ in specifications
    }
    synthetic_traces = {
        decision_id: replace(
            base_trace,
            decision_id=decision_id,
            events=(TraceEvent("agent_first_audio", first_audio_ms),),
        )
        for decision_id, _, _, first_audio_ms in specifications
    }

    first_order = ["a-safe", "b-interrupted", "c-safe-1", "c-safe-2"]
    second_order = ["a-safe", "c-safe-1", "c-safe-2", "b-interrupted"]
    first_report = score_inputs(
        synthetic_recordings.values(),
        [synthetic_labels[item] for item in first_order],
        synthetic_traces.values(),
        provider="fixture",
        config={},
    )
    second_report = score_inputs(
        synthetic_recordings.values(),
        [synthetic_labels[item] for item in second_order],
        synthetic_traces.values(),
        provider="fixture",
        config={},
    )

    assert first_report["pir_bootstrap_95"] == second_report["pir_bootstrap_95"]


@pytest.mark.parametrize(
    "score",
    [
        DecisionScore("d", "continue", "scored", False, None, None, None),
    ],
)
def test_valid_decision_score_is_immutable(score):
    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        score.interrupted = True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("d", "anything", "scored", False, None, None, None), "outcome"),
        (("d", "continue", "scored", None, None, None, None), "interrupted as bool"),
        (("d", "continue", "scored", 1, None, None, None), "interrupted as bool"),
        (("d", "continue", "scored", False, 0, None, None), "response_delay_ms"),
        (("d", "yield", "scored", None, None, None, None), "integer response_delay_ms"),
        (("d", "yield", "scored", None, True, None, None), "integer response_delay_ms"),
        (("d", "overlap", "scored", False, None, None, None), "continue or yield"),
        (("d", "continue", "unavailable", None, None, None, None), "unavailable_reason"),
        (("d", "continue", "unavailable", False, None, "provider_timeout", None), "contribute"),
        (("d", "overlap", "unavailable", None, None, "provider_timeout", None), "continue or yield"),
        (("d", "overlap", "excluded", None, None, None, None), "exclusion_reason"),
        (("d", "yield", "excluded", None, None, None, "ambiguous"), "overlap or unclear"),
        (("d", "unclear", "excluded", None, 10, None, "ambiguous"), "contribute"),
        (("d", "unclear", "excluded", None, None, "anything", "ambiguous"), "unavailable_reason"),
    ],
)
def test_decision_score_constructor_rejects_incoherent_state(args, message):
    with pytest.raises(SchemaError, match=message):
        DecisionScore(*args)


def test_aggregate_revalidates_mutated_score():
    score = DecisionScore("d", "continue", "scored", False, None, None, None)
    object.__setattr__(score, "interrupted", None)

    with pytest.raises(SchemaError, match="interrupted as bool"):
        aggregate([score])


def test_bootstrap_revalidates_mutated_score(labels):
    score = DecisionScore(
        "hi-continue", "continue", "scored", False, None, None, None
    )
    object.__setattr__(score, "interrupted", None)

    with pytest.raises(SchemaError, match="interrupted as bool"):
        bootstrap_by_recording([score], labels, draws=20, seed=7)


def test_bootstrap_rejects_missing_and_mismatched_label_map_entries(labels):
    score = DecisionScore(
        "hi-continue", "continue", "scored", False, None, None, None
    )
    with pytest.raises(ValueError, match="missing label"):
        bootstrap_by_recording([score], {}, draws=20, seed=7)

    wrong_label = replace(labels["hi-continue"], decision_id="other")
    with pytest.raises(ValueError, match="mapping key does not match"):
        bootstrap_by_recording(
            [score], {"hi-continue": wrong_label}, draws=20, seed=7
        )

    wrong_outcome = replace(
        labels["hi-continue"],
        outcome="yield",
        continuation_resume_ms=None,
        true_end_ms=labels["hi-continue"].previous_speech_end_ms,
    )
    with pytest.raises(ValueError, match="label outcome does not match"):
        bootstrap_by_recording(
            [score], {"hi-continue": wrong_outcome}, draws=20, seed=7
        )


def test_bootstrap_revalidates_label(labels):
    score = DecisionScore(
        "hi-continue", "continue", "scored", False, None, None, None
    )
    label = labels["hi-continue"]
    object.__setattr__(label, "outcome", "anything")

    with pytest.raises(SchemaError, match="outcome"):
        bootstrap_by_recording(
            [score], {"hi-continue": label}, draws=20, seed=7
        )


def test_score_inputs_validates_linked_ids(recordings, labels, traces):
    report = score_inputs(
        recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
    )
    assert report["overall"]["decision_n"] == 5
    assert report["micro_overall"]["decision_n"] == 5
    assert report["run_id"] == "fixture-run"
    assert report["label_provenance"] == {
        "fixture_label_n": 5,
        "adjudicated_label_n": 0,
    }
    assert report["pir_bootstrap_95"] == {
        "aggregation": "micro_by_decision",
        "metric": "pir",
        "low": None,
        "high": None,
    }
    assert set(report["by_condition"]) == {"fixture"}

    micro = report["micro_overall"]
    assert micro["source_recording_n"] == 2
    assert micro["fixture_label_n"] == 5
    assert micro["adjudicated_label_n"] == 0
    assert micro["trace_n"] == 5
    assert micro["available_n"] == 4
    assert micro["unavailable_n"] == 1
    assert micro["availability_rate"] == pytest.approx(4 / 5)
    assert micro["provider_failed_n"] == 0
    assert micro["provider_failed_rate"] == 0
    assert micro["provider_timeout_n"] == 1
    assert micro["provider_timeout_rate"] == pytest.approx(1 / 5)
    assert micro["missing_agent_first_audio_n"] == 0
    assert micro["missing_agent_first_audio_rate"] == 0

    hi = report["by_language"]["hi"]
    assert hi["trace_n"] == 3
    assert hi["available_n"] == 2
    assert hi["availability_rate"] == pytest.approx(2 / 3)
    assert hi["provider_timeout_n"] == 1
    assert hi["provider_timeout_rate"] == pytest.approx(1 / 3)
    assert report["by_condition"]["fixture"] == micro
    assert report["by_source_recording"]["hi-source-1"]["source_recording_n"] == 1
    assert report["by_source_recording"]["ta-source-2"]["source_recording_n"] == 1

    grouped_summaries = (
        micro,
        *report["by_language"].values(),
        *report["by_condition"].values(),
        *report["by_source_recording"].values(),
    )
    unavailable_reasons = (
        "provider_failed",
        "provider_timeout",
        "missing_agent_first_audio",
    )
    for summary in grouped_summaries:
        assert summary["available_n"] + summary["unavailable_n"] == summary["trace_n"]
        expected_availability = (
            summary["available_n"] / summary["trace_n"]
            if summary["trace_n"]
            else None
        )
        assert summary["availability_rate"] == pytest.approx(expected_availability)
        for reason in unavailable_reasons:
            expected_rate = (
                summary[f"{reason}_n"] / summary["trace_n"]
                if summary["trace_n"]
                else None
            )
            assert summary[f"{reason}_rate"] == pytest.approx(expected_rate)

    overall = report["overall"]
    assert overall["source_recording_n"] == 2
    assert overall["availability_rate"] == pytest.approx((2 / 3 + 1) / 2)
    assert overall["availability_rate_language_n"] == 2
    assert overall["provider_timeout_rate"] == pytest.approx((1 / 3 + 0) / 2)
    assert overall["provider_timeout_rate_language_n"] == 2
    for metric in (
        "availability_rate",
        "provider_failed_rate",
        "provider_timeout_rate",
        "missing_agent_first_audio_rate",
    ):
        assert overall[f"{metric}_language_n"] == 2


def test_report_coverage_keeps_timeout_on_excluded_decision(
    recordings, labels, traces
):
    traces["hi-overlap"] = replace(
        traces["hi-overlap"], status="timeout", events=()
    )

    report = score_inputs(
        recordings.values(),
        labels.values(),
        traces.values(),
        provider="fixture",
        config={},
    )

    assert report["micro_overall"]["excluded_n"] == 2
    assert report["micro_overall"]["provider_timeout_n"] == 2
    assert report["micro_overall"]["provider_timeout_rate"] == pytest.approx(2 / 5)
    assert report["by_language"]["hi"]["provider_timeout_n"] == 2
    assert report["by_language"]["hi"]["provider_timeout_rate"] == pytest.approx(2 / 3)


def test_conditions_and_adjudicated_provenance_come_from_linked_records(
    recordings, labels, traces
):
    real_recordings = {
        recording_id: replace(
            recording,
            condition="telephone" if recording.language == "hi" else "studio",
        )
        for recording_id, recording in recordings.items()
    }
    adjudicated_labels = {
        decision_id: replace(
            label,
            final_label=label.outcome,
            annotations=(
                Annotation("native-a", label.outcome),
                Annotation("native-b", label.outcome),
            ),
        )
        for decision_id, label in labels.items()
    }

    report = score_inputs(
        real_recordings.values(),
        adjudicated_labels.values(),
        traces.values(),
        provider="fixture",
        config={"adapter": "offline-test"},
    )

    assert set(report["by_condition"]) == {"studio", "telephone"}
    assert report["by_condition"]["telephone"]["decision_n"] == 3
    assert report["by_condition"]["studio"]["decision_n"] == 2
    assert report["label_provenance"] == {
        "fixture_label_n": 0,
        "adjudicated_label_n": 5,
    }
    assert report["micro_overall"]["adjudicated_label_n"] == 5


def test_score_inputs_rejects_missing_trace(recordings, labels, traces):
    traces.pop("hi-continue")
    with pytest.raises(ValueError, match="missing trace"):
        score_inputs(recordings.values(), labels.values(), traces.values(), provider="fixture", config={})


def test_score_inputs_rejects_duplicate_and_unknown_ids(recordings, labels, traces):
    with pytest.raises(ValueError, match="duplicate decision_id"):
        score_inputs(
            recordings.values(),
            [*labels.values(), labels["hi-continue"]],
            traces.values(),
            provider="fixture",
            config={},
        )
    unknown_trace = ProviderTrace(
        run_id="fixture-run",
        decision_id="not-a-label",
        provider="fixture",
        status="ok",
        error=None,
        events=(),
    )
    with pytest.raises(ValueError, match="unknown decision_id"):
        score_inputs(
            recordings.values(),
            labels.values(),
            [*traces.values(), unknown_trace],
            provider="fixture",
            config={},
        )


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        ("provider", "trace provider does not match report provider"),
        ("run_id", "mixed run_id values"),
    ],
)
def test_score_inputs_rejects_mixed_provider_provenance(
    recordings, labels, traces, trace, message
):
    replacement = replace(
        traces["ta-yield"], **{trace: "other" if trace == "provider" else "other-run"}
    )
    traces["ta-yield"] = replacement
    with pytest.raises(ValueError, match=message):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )


def test_score_inputs_rejects_target_speaker_mismatch(recordings, labels, traces):
    labels["hi-continue"] = replace(labels["hi-continue"], target_speaker_id="agent")
    with pytest.raises(ValueError, match="target_speaker_id does not match recording"):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )


def test_score_decision_rejects_mismatched_trace_id(labels, traces):
    with pytest.raises(ValueError, match="decision_id must match"):
        score_decision(labels["hi-continue"], replace(traces["hi-continue"], decision_id="other"))


def test_fixture_labels_match_recording_boundaries(labels, recordings):
    for label in labels.values():
        recording = recordings[label.recording_id]
        assert label.target_speaker_id == recording.target_speaker_id
        assert any(
            turn.speaker_id == label.target_speaker_id and turn.end_ms == label.previous_speech_end_ms
            for turn in recording.turns
        )
        matching_next = [
            turn for turn in recording.turns if turn.start_ms == label.next_event_start_ms
        ]
        assert matching_next
        if label.outcome == "continue":
            assert label.continuation_resume_ms == label.next_event_start_ms
            assert any(turn.speaker_id == label.target_speaker_id for turn in matching_next)


def test_score_inputs_rejects_fabricated_continuation_time(recordings, labels, traces):
    corrupted = labels["hi-continue"]
    object.__setattr__(corrupted, "continuation_resume_ms", 500)
    with pytest.raises(ValueError, match="continuation_resume_ms"):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )


def test_score_inputs_rejects_continuation_owned_by_wrong_speaker(
    recordings, labels, traces
):
    recording = recordings["hi-fixture-1"]
    resumed = replace(recording.turns[1], speaker_id="agent")
    recordings[recording.recording_id] = replace(
        recording, turns=(recording.turns[0], resumed)
    )
    with pytest.raises(ValueError, match="continue next event must belong to target speaker"):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )


def test_score_inputs_rejects_target_and_non_target_co_starting_continuation(
    recordings, labels, traces
):
    recording = recordings["hi-fixture-1"]
    recordings[recording.recording_id] = replace(
        recording,
        turns=(
            recording.turns[0],
            SourceTurn("agent", 1600, 1700),
            recording.turns[1],
        ),
    )

    with pytest.raises(ValueError, match="includes a non-target speaker"):
        score_inputs(
            [recordings[recording.recording_id]],
            [labels["hi-continue"]],
            [traces["hi-continue"]],
            provider="fixture",
            config={},
        )


def test_score_inputs_rejects_unlinked_decision_boundaries(recordings, labels, traces):
    labels["hi-continue"] = replace(
        labels["hi-continue"], next_event_start_ms=1500, continuation_resume_ms=1500
    )
    with pytest.raises(ValueError, match="next_event_start_ms does not match"):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )


def test_score_inputs_rejects_active_agent_across_continuation_boundary(
    recordings, labels, traces
):
    recording = recordings["hi-fixture-1"]
    recordings[recording.recording_id] = replace(
        recording,
        turns=(
            recording.turns[0],
            replace(recording.turns[1], speaker_id="agent", start_ms=900, end_ms=1400),
            recording.turns[1],
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not end at a silent boundary: hi-continue",
    ):
        score_inputs(
            [recordings[recording.recording_id]],
            [labels["hi-continue"]],
            [traces["hi-continue"]],
            provider="fixture",
            config={},
        )


def test_score_inputs_rejects_continuation_that_skips_earlier_agent_event(
    recordings, labels, traces
):
    recording = recordings["hi-fixture-1"]
    recordings[recording.recording_id] = replace(
        recording,
        turns=(
            recording.turns[0],
            SourceTurn("agent", 1200, 1400),
            recording.turns[1],
        ),
    )

    with pytest.raises(
        ValueError,
        match="not the earliest distinct source-turn start: hi-continue",
    ):
        score_inputs(
            [recordings[recording.recording_id]],
            [labels["hi-continue"]],
            [traces["hi-continue"]],
            provider="fixture",
            config={},
        )


def test_score_inputs_rejects_yield_that_skips_earlier_event(
    recordings, labels, traces
):
    recording = recordings["ta-fixture-1"]
    recordings[recording.recording_id] = replace(
        recording,
        turns=(
            recording.turns[0],
            SourceTurn("caller", 2200, 2400),
            recording.turns[1],
        ),
    )

    with pytest.raises(
        ValueError,
        match="not the earliest distinct source-turn start: ta-yield",
    ):
        score_inputs(
            [recordings[recording.recording_id]],
            [labels["ta-yield"]],
            [traces["ta-yield"]],
            provider="fixture",
            config={},
        )


def test_score_inputs_rejects_duplicate_preceding_target_boundaries(
    recordings, labels, traces
):
    recording = recordings["hi-fixture-1"]
    recordings[recording.recording_id] = replace(
        recording,
        turns=(
            recording.turns[0],
            SourceTurn("caller", 200, 1000),
            recording.turns[1],
        ),
    )

    with pytest.raises(
        ValueError,
        match="ambiguous preceding target-speaker turn: hi-continue",
    ):
        score_inputs(
            [recordings[recording.recording_id]],
            [labels["hi-continue"]],
            [traces["hi-continue"]],
            provider="fixture",
            config={},
        )


def test_score_inputs_accepts_silent_immediate_target_continuation(
    recordings, labels, traces
):
    report = score_inputs(
        [recordings["hi-fixture-1"]],
        [labels["hi-continue"]],
        [traces["hi-continue"]],
        provider="fixture",
        config={},
    )

    assert report["micro_overall"]["pir_n"] == 1
    assert report["micro_overall"]["excluded_n"] == 0


def test_score_inputs_rejects_fixture_label_on_real_recording(recordings, labels, traces):
    recording = recordings["hi-fixture-1"]
    recordings[recording.recording_id] = replace(recording, condition="telephone")
    with pytest.raises(ValueError, match="fixture label requires a fixture recording"):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )


def test_score_inputs_rejects_adjudicated_label_on_fixture_recording(
    recordings, labels, traces
):
    labels["hi-continue"] = replace(
        labels["hi-continue"],
        final_label="continue",
        annotations=(
            Annotation("native-a", "continue"),
            Annotation("native-b", "yield"),
        ),
    )
    with pytest.raises(ValueError, match="adjudicated label cannot reference a fixture"):
        score_inputs(
            [recordings["hi-fixture-1"]],
            [labels["hi-continue"]],
            [traces["hi-continue"]],
            provider="fixture",
            config={},
        )


def test_score_inputs_rejects_mixed_fixture_and_adjudicated_invocation(
    recordings, labels, traces
):
    recording = recordings["hi-fixture-1"]
    recordings[recording.recording_id] = replace(recording, condition="telephone")
    labels["hi-continue"] = replace(
        labels["hi-continue"],
        final_label="continue",
        annotations=(
            Annotation("native-a", "continue"),
            Annotation("native-b", "continue"),
        ),
    )
    with pytest.raises(ValueError, match="mixed fixture and adjudicated"):
        score_inputs(
            recordings.values(), labels.values(), traces.values(), provider="fixture", config={}
        )

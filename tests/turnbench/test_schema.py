from dataclasses import replace

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
    write_jsonl,
)


def _use_consistent_continue_time(row):
    row["continuation_resume_ms"] = row["next_event_start_ms"]
    return row


def test_continue_requires_resume_time():
    row = {
        "schema": "turnbench.label.v1", "decision_id": "d1",
        "recording_id": "r1", "source_recording_id": "src1",
        "target_speaker_id": "caller", "previous_speech_end_ms": 1000,
        "next_event_start_ms": 1600, "outcome": "continue",
        "true_end_ms": None, "exclusion_reason": None,
        "final_label": "fixture", "annotations": [],
    }
    with pytest.raises(SchemaError, match="continuation_resume_ms"):
        DecisionLabel.from_dict(row)


def test_trace_rejects_descending_timestamps():
    row = {
        "schema": "turnbench.event.v1", "run_id": "run1", "decision_id": "d1",
        "provider": "fixture", "status": "ok", "error": None,
        "events": [
            {"kind": "turn_committed", "t_ms": 1200},
            {"kind": "agent_first_audio", "t_ms": 1100},
        ],
    }
    with pytest.raises(SchemaError, match="monotonic"):
        ProviderTrace.from_dict(row)


def test_recording_round_trip(tmp_path, recording):
    path = tmp_path / "nested" / "recordings.jsonl"
    write_jsonl(path, [recording.to_dict()])
    assert read_jsonl(path, Recording.from_dict) == [recording]


def test_jsonl_round_trips_multilingual_text_as_utf8(tmp_path):
    path = tmp_path / "hindi.jsonl"
    row = {"phrase": "मुझे बाद में एक नंबर देना है"}
    write_jsonl(path, [row])
    assert read_jsonl(path, lambda value: value) == [row]


def test_source_turn_is_unversioned_nested_record():
    import asli.turnbench as turnbench
    import asli.turnbench.schema as schema_module

    turn = SourceTurn("caller", 0, 1)
    assert not hasattr(turnbench, "SOURCE_TURN_SCHEMA")
    assert not hasattr(schema_module, "SOURCE_TURN_SCHEMA")
    assert not hasattr(SourceTurn, "schema")
    assert "schema" not in turn.to_dict()


@pytest.mark.parametrize("outcome,missing", [
    ("yield", "true_end_ms"),
    ("overlap", "exclusion_reason"),
    ("unclear", "exclusion_reason"),
])
def test_outcome_requires_its_specific_field(label_row, outcome, missing):
    _use_consistent_continue_time(label_row)
    label_row["outcome"] = outcome
    label_row.pop(missing, None)
    with pytest.raises(SchemaError, match=missing):
        DecisionLabel.from_dict(label_row)


def test_source_turns_may_overlap_but_must_be_ordered(recording):
    row = recording.to_dict()
    row["turns"][1]["start_ms"] = 900
    assert Recording.from_dict(row).turns[1].start_ms == 900


def test_non_fixture_label_requires_two_distinct_annotations(label_row):
    _use_consistent_continue_time(label_row)
    label_row.update(final_label="continue", annotations=[{"annotator_id": "a", "label": "continue"}])
    with pytest.raises(SchemaError, match="exactly two"):
        DecisionLabel.from_dict(label_row)


@pytest.mark.parametrize("outcome,final_label", [("overlap", "continue"), ("unclear", "yield")])
def test_excluded_outcome_cannot_have_headline_final_label(label_row, outcome, final_label):
    _use_consistent_continue_time(label_row)
    label_row.update(
        outcome=outcome,
        final_label=final_label,
        exclusion_reason="ambiguous",
        annotations=[
            {"annotator_id": "a", "label": "continue"},
            {"annotator_id": "b", "label": "yield"},
        ],
    )
    with pytest.raises(SchemaError, match="final_label must equal outcome"):
        DecisionLabel.from_dict(label_row)


@pytest.mark.parametrize("status", ["failed", "timeout"])
def test_failed_trace_can_have_no_error_or_events(status):
    trace = ProviderTrace.from_dict({
        "schema": "turnbench.event.v1", "run_id": "run1", "decision_id": "d1",
        "provider": "fixture", "status": status, "error": None, "events": [],
    })
    assert trace.status == status
    assert trace.error is None
    assert trace.events == ()


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: SourceTurn("", 0, 1), "speaker_id"),
        (lambda: SourceTurn("caller", -1, 1), "start_ms"),
        (lambda: SourceTurn("caller", 2, 1), "end_ms"),
        (lambda: TraceEvent("made_up", 1), "unknown event kind"),
        (lambda: TraceEvent("speech_started", -1), "t_ms"),
        (lambda: Annotation("a", "fixture"), "annotation label"),
        (lambda: Annotation("a", "banana"), "annotation label"),
        (
            lambda: ProviderTrace("run", "d", "p", "anything", None, ()),
            "status",
        ),
        (
            lambda: ProviderTrace(
                "run",
                "d",
                "p",
                "ok",
                None,
                (TraceEvent("speech_started", 2), TraceEvent("agent_first_audio", 1)),
            ),
            "monotonic",
        ),
    ],
)
def test_public_record_constructors_reject_invalid_values(constructor, message):
    with pytest.raises(SchemaError, match=message):
        constructor()


def test_recording_constructor_rejects_unsorted_source_turns(recording):
    with pytest.raises(SchemaError, match="sorted"):
        replace(recording, turns=tuple(reversed(recording.turns)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"outcome": "anything"}, "outcome"),
        ({"continuation_resume_ms": 500}, "next_event_start_ms"),
        ({"continuation_resume_ms": 1000, "next_event_start_ms": 1000}, "after"),
        ({"true_end_ms": 900}, "true_end_ms to be null"),
        ({"exclusion_reason": "not used"}, "exclusion_reason to be null"),
    ],
)
def test_decision_constructor_rejects_invalid_continue_fields(label_row, changes, message):
    _use_consistent_continue_time(label_row)
    label_row.update(changes)
    label_row.pop("schema")
    label_row["annotations"] = ()
    with pytest.raises(SchemaError, match=message):
        DecisionLabel(**label_row)


def test_decision_constructor_enforces_yield_time_semantics(label_row):
    _use_consistent_continue_time(label_row)
    label_row.update(
        outcome="yield",
        continuation_resume_ms=None,
        true_end_ms=1001,
    )
    label_row.pop("schema")
    label_row["annotations"] = ()
    with pytest.raises(SchemaError, match="at or before"):
        DecisionLabel(**label_row)


@pytest.mark.parametrize("outcome", ["overlap", "unclear"])
def test_excluded_decision_constructor_rejects_headline_times(label_row, outcome):
    _use_consistent_continue_time(label_row)
    label_row.update(
        outcome=outcome,
        exclusion_reason="ambiguous",
        continuation_resume_ms=None,
        true_end_ms=1000,
    )
    label_row.pop("schema")
    label_row["annotations"] = ()
    with pytest.raises(SchemaError, match="true_end_ms to be null"):
        DecisionLabel(**label_row)


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema":"turnbench.event.v1","run_id":"r","decision_id":"d",'
        '"provider":"p","status":"failed","status":"ok","error":null,"events":[]}',
        '{"schema":"turnbench.label.v1","decision_id":"d","recording_id":"r",'
        '"source_recording_id":"s","target_speaker_id":"caller",'
        '"previous_speech_end_ms":0,"next_event_start_ms":1,"outcome":"yield",'
        '"outcome":"continue","true_end_ms":null,"exclusion_reason":null,'
        '"final_label":"fixture","annotations":[],"continuation_resume_ms":1}',
        '{"schema":"turnbench.event.v1","run_id":"r","decision_id":"d",'
        '"provider":"p","status":"ok","error":null,'
        '"events":[{"kind":"speech_started","kind":"agent_first_audio","t_ms":1}]}',
    ],
)
def test_read_jsonl_rejects_duplicate_keys_at_every_object_level(tmp_path, contents):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + contents + "\n")
    with pytest.raises(SchemaError, match=r"bad\.jsonl:2: duplicate key"):
        read_jsonl(path, ProviderTrace.from_dict)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_read_jsonl_rejects_non_standard_constants(tmp_path, constant):
    path = tmp_path / "bad.jsonl"
    path.write_text(f'{{"value":{constant}}}\n')
    with pytest.raises(SchemaError, match=r"bad\.jsonl:1: non-standard JSON constant"):
        read_jsonl(path, lambda row: row)


def test_write_jsonl_rejects_non_finite_numbers_with_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    with pytest.raises(SchemaError, match=r"bad\.jsonl:2:"):
        write_jsonl(path, [{"value": 1.0}, {"value": float("nan")}])
    assert not path.exists()


@pytest.mark.parametrize("row", [1, ["not", "an", "object"]])
def test_write_jsonl_rejects_non_object_rows_with_line(tmp_path, row):
    path = tmp_path / "bad.jsonl"
    with pytest.raises(
        SchemaError, match=r"bad\.jsonl:1: record must be an object"
    ):
        write_jsonl(path, [row])
    assert not path.exists()

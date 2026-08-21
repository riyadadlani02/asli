from dataclasses import replace

import pytest

from asli.turnbench.auto_report import (
    DiarBenchExportProvenance,
    compare_auto_predictions,
)
from asli.turnbench.auto_schema import (
    REFERENCE_SOURCE,
    AutoPrediction,
    DiarBenchCandidate,
    DiarBenchReference,
)


def candidate(
    decision_id: str,
    *,
    language: str = "Hindi",
    condition: str = "clean",
    source: str = "source-1",
    previous_end: int = 1000,
    observation_end: int = 1600,
) -> DiarBenchCandidate:
    return DiarBenchCandidate(
        decision_id=decision_id,
        recording_id=f"recording-{decision_id}",
        source_recording_id=source,
        audio_path=f"audio/{decision_id}.wav",
        language=language,
        condition=condition,
        target_speaker_id="speaker-a",
        context_start_ms=500,
        previous_speech_end_ms=previous_end,
        observation_end_ms=observation_end,
    )


def reference(
    row: DiarBenchCandidate,
    outcome: str,
) -> DiarBenchReference:
    return DiarBenchReference(
        row,
        outcome,
        REFERENCE_SOURCE,
        None if outcome in {"continue", "yield"} else f"{outcome}-reason",
    )


def prediction(
    decision_id: str,
    outcome: str | None,
    *,
    endpoint_ms: int | None = None,
    run_id: str = "run-1",
    agent: str = "openai",
    model: str = "gpt-4o-transcribe",
    config: dict[str, object] | None = None,
) -> AutoPrediction:
    available = outcome is not None
    return AutoPrediction(
        decision_id=decision_id,
        run_id=run_id,
        agent=agent,
        model=model,
        config=config or {"context_ms": 500, "mode": "semantic_vad"},
        status="available" if available else "unavailable",
        outcome=outcome,
        endpoint_ms=endpoint_ms,
        unavailable_reason=None if available else "provider_error",
    )


def export_provenance(
    *,
    languages: tuple[str, ...] = ("Hindi", "Tamil", "Kannada"),
    min_pause_ms: int = 300,
    max_pause_ms: int = 2000,
    context_ms: int = 500,
) -> DiarBenchExportProvenance:
    return DiarBenchExportProvenance(
        dataset="sarvamai/indic-diarbench",
        dataset_revision="test",
        requested_languages=languages,
        min_pause_ms=min_pause_ms,
        max_pause_ms=max_pause_ms,
        context_ms=context_ms,
    )


def compare(candidates, references, predictions, *, provenance=None):
    return compare_auto_predictions(
        candidates,
        references,
        predictions,
        export_provenance=provenance or export_provenance(),
    )


def test_export_provenance_round_trips_its_versioned_manifest_contract():
    provenance = export_provenance(languages=("Tamil", "Hindi"))

    assert DiarBenchExportProvenance.from_dict(provenance.to_dict()) == provenance
    assert provenance.to_dict() == {
        "schema": "turnbench.diarbench.export.v1",
        "dataset": "sarvamai/indic-diarbench",
        "dataset_revision": "test",
        "requested_languages": ["Hindi", "Tamil"],
        "min_pause_ms": 300,
        "max_pause_ms": 2000,
        "context_ms": 500,
    }


def test_known_binary_matrix_and_sorted_groupings():
    rows = [
        candidate("d1", language="Tamil", condition="noisy", source="source-b"),
        candidate("d2", language="Hindi", condition="clean", source="source-a"),
        candidate("d3", language="Tamil", condition="noisy", source="source-b"),
        candidate("d4", language="Hindi", condition="clean", source="source-a"),
    ]
    references = [
        reference(rows[0], "continue"),
        reference(rows[1], "continue"),
        reference(rows[2], "yield"),
        reference(rows[3], "yield"),
    ]
    predictions = [
        prediction("d1", "continue"),
        prediction("d2", "yield", endpoint_ms=1200),
        prediction("d3", "continue"),
        prediction("d4", "yield", endpoint_ms=1200),
    ]

    report = compare(
        rows,
        references,
        predictions,
        provenance=export_provenance(languages=("Hindi", "Tamil")),
    )

    assert report["schema"] == "turnbench.auto_accuracy.v1"
    assert report["dataset"] == "sarvamai/indic-diarbench"
    assert report["dataset_revision"] == "test"
    assert report["reference_source"] == REFERENCE_SOURCE
    assert report["run_id"] == "run-1"
    assert report["agent"] == "openai"
    assert report["model"] == "gpt-4o-transcribe"
    assert report["config"] == {"context_ms": 500, "mode": "semantic_vad"}
    assert report["requested_languages"] == ["Hindi", "Tamil"]
    assert report["min_pause_ms"] == 300
    assert report["max_pause_ms"] == 2000
    assert report["context_ms"] == 500
    assert report["micro_overall"]["accuracy"] == 0.5
    assert report["micro_overall"]["continue_precision"] == 0.5
    assert report["micro_overall"]["continue_recall"] == 0.5
    assert report["micro_overall"]["continue_f1"] == 0.5
    assert report["micro_overall"]["confusion"] == {
        "continue": {"continue": 1, "yield": 1},
        "yield": {"continue": 1, "yield": 1},
    }
    assert list(report["by_language"]) == ["Hindi", "Tamil"]
    assert list(report["by_condition"]) == ["clean", "noisy"]
    assert list(report["by_source_recording"]) == ["source-a", "source-b"]


def test_unavailable_lowers_coverage_without_entering_accuracy():
    rows = [candidate("d1"), candidate("d2")]
    report = compare(
        rows,
        [reference(rows[0], "continue"), reference(rows[1], "yield")],
        [prediction("d1", "continue"), prediction("d2", None)],
    )

    summary = report["micro_overall"]
    assert summary["eligible_reference_n"] == 2
    assert summary["available_n"] == 1
    assert summary["unavailable_n"] == 1
    assert summary["coverage_rate"] == 0.5
    assert summary["unavailable_rate"] == 0.5
    assert summary["correct_n"] == 1
    assert summary["accuracy"] == 1.0
    assert summary["confusion"] == {
        "continue": {"continue": 1, "yield": 0},
        "yield": {"continue": 0, "yield": 0},
    }


def test_excluded_references_are_counted_and_never_scored():
    binary = candidate("binary")
    overlap = candidate("overlap", language="Tamil")
    unclear = candidate("unclear", language="Kannada")

    report = compare(
        [binary],
        [
            reference(binary, "yield"),
            reference(overlap, "overlap"),
            reference(unclear, "unclear"),
        ],
        [prediction("binary", "yield", endpoint_ms=1100)],
    )

    assert report["reference_n"] == 3
    assert report["reference_excluded_n"] == 2
    assert report["overlap_n"] == 1
    assert report["unclear_n"] == 1
    assert report["micro_overall"]["eligible_reference_n"] == 1
    assert report["micro_overall"]["accuracy"] == 1.0


def test_macro_accuracy_is_an_unweighted_mean_of_defined_language_values():
    rows = [
        candidate("h1", language="Hindi"),
        candidate("t1", language="Tamil"),
        candidate("t2", language="Tamil"),
        candidate("t3", language="Tamil"),
    ]
    report = compare(
        rows,
        [reference(row, "yield") for row in rows],
        [
            prediction("h1", "yield", endpoint_ms=1100),
            prediction("t1", "yield", endpoint_ms=1100),
            prediction("t2", "continue"),
            prediction("t3", "continue"),
        ],
    )

    assert report["micro_overall"]["accuracy"] == 0.5
    assert report["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert report["overall"]["accuracy_language_n"] == 2
    assert report["overall"]["eligible_reference_n"] == 4


def test_empty_continue_class_has_null_metrics_not_invented_zeroes():
    row = candidate("d1")
    report = compare(
        [row],
        [reference(row, "yield")],
        [prediction("d1", "yield", endpoint_ms=1100)],
    )

    summary = report["micro_overall"]
    assert summary["continue_precision"] is None
    assert summary["continue_recall"] is None
    assert summary["continue_f1"] is None
    assert report["overall"]["continue_precision"] is None
    assert report["overall"]["continue_precision_language_n"] == 0


def test_endpoint_error_uses_correct_timestamped_yields_and_nearest_rank():
    rows = [candidate(f"d{index}") for index in range(1, 6)]
    references = [reference(row, "yield") for row in rows]
    predictions = [
        prediction("d1", "yield", endpoint_ms=1010),
        prediction("d2", "yield", endpoint_ms=1020),
        prediction("d3", "yield", endpoint_ms=1030),
        prediction("d4", "continue"),
        prediction("d5", None),
    ]

    report = compare(rows, references, predictions)

    assert report["endpoint_timing"] == {
        "correct_timestamped_yield_n": 3,
        "endpoint_observation_error_p50_ms": 20,
        "endpoint_observation_error_p95_ms": 30,
    }


@pytest.mark.parametrize("endpoint_ms", [999, 1000, 1600, 1700])
def test_available_yield_endpoint_must_be_strictly_inside_candidate_window(
    endpoint_ms,
):
    row = candidate("d1")

    with pytest.raises(ValueError, match="yield endpoint outside candidate window"):
        compare(
            [row],
            [reference(row, "yield")],
            [prediction("d1", "yield", endpoint_ms=endpoint_ms)],
        )


def test_export_provenance_is_emitted_even_when_a_requested_language_has_no_rows():
    row = candidate("d1", language="Hindi")

    report = compare(
        [row],
        [reference(row, "continue")],
        [prediction("d1", "continue")],
        provenance=export_provenance(languages=("Tamil", "Hindi")),
    )

    assert report["requested_languages"] == ["Hindi", "Tamil"]
    assert report["dataset"] == "sarvamai/indic-diarbench"
    assert report["dataset_revision"] == "test"
    assert report["min_pause_ms"] == 300
    assert report["max_pause_ms"] == 2000
    assert report["context_ms"] == 500


@pytest.mark.parametrize(
    ("row", "provenance", "config", "message"),
    [
        (
            candidate("language", language="Malayalam"),
            export_provenance(languages=("Hindi",)),
            None,
            "candidate language not requested",
        ),
        (
            candidate("short-pause"),
            export_provenance(min_pause_ms=700),
            None,
            "candidate pause outside export bounds",
        ),
        (
            candidate("long-pause"),
            export_provenance(max_pause_ms=500),
            None,
            "candidate pause outside export bounds",
        ),
        (
            candidate("context"),
            export_provenance(context_ms=400),
            None,
            "candidate context does not match export provenance",
        ),
        (
            candidate("run-context"),
            export_provenance(),
            {"context_ms": 400, "mode": "semantic_vad"},
            "prediction config context_ms does not match export provenance",
        ),
    ],
)
def test_export_provenance_rejects_incompatible_exported_inputs(
    row, provenance, config, message
):
    with pytest.raises(ValueError, match=message):
        compare(
            [row],
            [reference(row, "continue")],
            [prediction(row.decision_id, "continue", config=config)],
            provenance=provenance,
        )


def test_prediction_configs_compare_json_types_strictly_in_nested_values():
    first = candidate("d1")
    second = candidate("d2")
    common = {"context_ms": 500, "nested": {"flag": True, "items": [False]}}
    type_changed = {"context_ms": 500, "nested": {"flag": 1, "items": [0]}}

    with pytest.raises(ValueError, match="mixed config"):
        compare(
            [first, second],
            [reference(first, "continue"), reference(second, "continue")],
            [
                prediction("d1", "continue", config=common),
                prediction("d2", "continue", config=type_changed),
            ],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_candidate", "duplicate candidate decision_id"),
        ("duplicate_reference", "duplicate reference decision_id"),
        ("duplicate_prediction", "duplicate prediction decision_id"),
        ("missing_candidate", "binary reference missing candidate"),
        ("missing_reference", "candidate missing binary reference"),
        ("missing_prediction", "candidate missing prediction"),
        ("extra_prediction", "prediction missing candidate"),
        ("mixed_run", "mixed run_id"),
        ("mixed_agent", "mixed agent"),
        ("mixed_model", "mixed model"),
        ("mixed_config", "mixed config"),
        ("timing_mismatch", "candidate/reference metadata mismatch"),
        ("nonbinary_candidate", "candidate linked to excluded reference"),
    ],
)
def test_closed_join_rejects_partial_duplicate_or_mixed_inputs(mutation, message):
    first = candidate("d1")
    second = candidate("d2")
    candidates = [first, second]
    references = [reference(first, "continue"), reference(second, "yield")]
    predictions = [
        prediction("d1", "continue"),
        prediction("d2", "yield", endpoint_ms=1100),
    ]

    if mutation == "duplicate_candidate":
        candidates.append(first)
    elif mutation == "duplicate_reference":
        references.append(references[0])
    elif mutation == "duplicate_prediction":
        predictions.append(predictions[0])
    elif mutation == "missing_candidate":
        candidates.pop()
        predictions.pop()
    elif mutation == "missing_reference":
        references.pop()
    elif mutation == "missing_prediction":
        predictions.pop()
    elif mutation == "extra_prediction":
        candidates.pop()
        references.pop()
    elif mutation == "mixed_run":
        predictions[1] = replace(predictions[1], run_id="run-2")
    elif mutation == "mixed_agent":
        predictions[1] = replace(predictions[1], agent="other-agent")
    elif mutation == "mixed_model":
        predictions[1] = replace(predictions[1], model="other-model")
    elif mutation == "mixed_config":
        predictions[1] = replace(predictions[1], config={"context_ms": 1000})
    elif mutation == "timing_mismatch":
        changed = replace(second, observation_end_ms=1700)
        references[1] = reference(changed, "yield")
    elif mutation == "nonbinary_candidate":
        references[1] = reference(second, "overlap")

    with pytest.raises(ValueError, match=message):
        compare(candidates, references, predictions)

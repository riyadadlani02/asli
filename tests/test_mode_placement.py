import json

from asli.spec import CallSpec, Segment
from experiments.run_mode_placement import (
    as_mode_matrix,
    build_placement,
    main,
    missing_credentials,
    row_key,
    summary,
)


def base_spec() -> CallSpec:
    return CallSpec(
        id="dig-01",
        entity_type="digits",
        canonical="9877111",
        segments=[Segment("Mera mobile number hai"), Segment("nine eight")],
    )


def test_hesitation_placement_inserts_the_fixed_filler_before_the_entity():
    actual = build_placement(base_spec(), "hesitation")

    assert [segment.text for segment in actual.segments][-2:] == ["matlab", "nine eight"]
    assert actual.segments[-2].pause_after_ms == 700


def test_control_placement_keeps_the_fixed_filler_leading_without_a_pause():
    actual = build_placement(base_spec(), "control")

    assert [segment.text for segment in actual.segments][:2] == ["matlab", "Mera mobile number hai"]
    assert actual.segments[0].pause_after_ms == 0


def test_row_key_uses_all_experimental_coordinates():
    assert row_key({
        "id": "dig-01", "mode": "verbatim", "placement": "hesitation", "run": 5,
    }) == ("dig-01", "verbatim", "hesitation", 5)


def test_summary_excludes_error_rows_and_groups_each_mode_and_placement():
    rows = [
        {"id": "dig-01", "mode": "verbatim", "placement": "control", "run": 0,
         "survived": True, "error": ""},
        {"id": "dig-02", "mode": "verbatim", "placement": "control", "run": 0,
         "survived": False, "error": ""},
        {"id": "dig-03", "mode": "verbatim", "placement": "control", "run": 0,
         "survived": False, "error": "timeout"},
    ]

    assert summary(rows) == {
        "verbatim": {"control": {"survived": 1, "n": 2}},
    }


def test_missing_credentials_are_reported_before_the_probe_starts():
    assert missing_credentials({"SARVAM_API_KEY": "present"}) == [
        "ELEVEN_API_KEY", "ELEVEN_VOICE_ID"
    ]


def test_mode_matrix_has_the_site_schema_and_is_derived_from_completed_rows():
    rows = [
        {"id": "dig-01", "mode": "verbatim", "placement": "control", "run": 0,
         "survived": True, "error": ""},
        {"id": "dig-02", "mode": "verbatim", "placement": "control", "run": 0,
         "survived": False, "error": ""},
        {"id": "dig-01", "mode": "verbatim", "placement": "hesitation", "run": 0,
         "survived": True, "error": ""},
    ]

    assert as_mode_matrix(rows) == {
        "verbatim": {"control": [1, 2], "hesitation": [1, 1]},
    }


def test_report_only_refreshes_the_matrix_from_checkpointed_rows(tmp_path):
    out = tmp_path / "rows.json"
    matrix = tmp_path / "matrix.json"
    out.write_text(json.dumps([
        {"id": "dig-01", "mode": "verbatim", "placement": "control", "run": 0,
         "filler": "matlab", "survived": True, "error": ""},
        {"id": "dig-01", "mode": "verbatim", "placement": "hesitation", "run": 0,
         "filler": "matlab", "survived": False, "error": ""},
    ]))

    assert main(["--out", str(out), "--matrix-out", str(matrix), "--report-only"]) == 0
    assert json.loads(matrix.read_text()) == {
        "verbatim": {"control": [1, 1], "hesitation": [0, 1]},
    }

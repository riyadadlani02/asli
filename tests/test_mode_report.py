from build_site import mode_cell


def test_mode_cell_formats_the_generated_fixed_filler_counts():
    matrix = {
        "transcribe": {
            "control": [18, 24],
            "hesitation": [12, 24],
        }
    }

    assert mode_cell(matrix, "transcribe", "control") == "18/24"
    assert mode_cell(matrix, "transcribe", "hesitation") == "12/24"

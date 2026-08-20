# Semantic VAD and Mode Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make semantic VAD selectable for the OpenAI lane and produce a resumable, fixed-filler Sarvam mode comparison.

**Architecture:** `OpenAIWS` owns its provider payload behind a small pure helper, so tests can assert configuration without opening a socket. A dedicated experiment runner owns stimulus placement, checkpointing, and aggregation; it uses existing `SarvamWS`, `synth`, and `score` rather than duplicating provider or number-parsing code.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, websockets, pytest, existing `asli` modules.

**Spec:** `docs/superpowers/specs/2026-08-21-semantic-vad-and-mode-probe-design.md`

## Global Constraints

- Default OpenAI behavior is `server_vad` at 500 ms.
- `semantic_vad` must never be reported as a real-caller result without paired rows.
- Mode probe filler is exactly `matlab` and the result path is `results/mode_placement_fixed.json`.
- Each probe row is checkpointed immediately; failed calls are saved with an error and retried on resume.

---

### Task 1: Make OpenAI turn detection explicit

**Files:**
- Modify: `asli/drive.py:278-374`
- Modify: `asli/cli.py:102-110,162-460`
- Test: `tests/test_openai_turn_detection.py`

**Interfaces:**
- Produces: `OpenAIWS.turn_detection_payload() -> dict[str, object]`
- Produces: `make_adapter(..., turn_detection: str = "server_vad") -> OpenAIWS`

- [ ] **Step 1: Write the failing tests**

```python
def test_openai_server_vad_keeps_the_configured_gate():
    adapter = OpenAIWS(silence_duration_ms=700)
    assert adapter.turn_detection_payload() == {
        "type": "server_vad", "silence_duration_ms": 700
    }

def test_openai_semantic_vad_has_no_silence_timer():
    adapter = OpenAIWS(turn_detection="semantic_vad", silence_duration_ms=700)
    assert adapter.turn_detection_payload() == {"type": "semantic_vad"}
```

- [ ] **Step 2: Run tests and observe failure**

Run: `uv run --with pytest python -m pytest tests/test_openai_turn_detection.py -q`

- [ ] **Step 3: Implement the minimal configuration helper**

```python
def turn_detection_payload(self) -> dict[str, object]:
    if self.turn_detection == "semantic_vad":
        return {"type": "semantic_vad"}
    return {"type": "server_vad", "silence_duration_ms": self.gate}
```

Pass the helper's result to the existing session update and forward the CLI option only
to the OpenAI adapter.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run --with pytest python -m pytest tests/test_openai_turn_detection.py -q`

Run: `uv run --with pytest python -m pytest -q`

### Task 2: Add the fixed-filler mode runner

**Files:**
- Create: `experiments/run_mode_placement.py`
- Test: `tests/test_mode_placement.py`

**Interfaces:**
- Produces: `build_placement(spec: CallSpec, placement: str, filler: str = "matlab") -> CallSpec`
- Produces: `row_key(row: dict) -> tuple[str, str, str, int]`
- Produces: `results/mode_placement_fixed.json`

- [ ] **Step 1: Write failing pure-function tests**

```python
def test_hesitation_placement_inserts_fixed_filler_before_the_entity():
    actual = build_placement(base_spec(), "hesitation")
    assert [s.text for s in actual.segments][-2:] == ["matlab", "nine eight"]
    assert actual.segments[-2].pause_after_ms == 700

def test_done_key_uses_all_experimental_coordinates():
    assert row_key({"id": "dig-01", "mode": "verbatim", "placement": "hesitation", "run": 5}) == (
        "dig-01", "verbatim", "hesitation", 5
    )
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `uv run --with pytest python -m pytest tests/test_mode_placement.py -q`

- [ ] **Step 3: Implement a checkpointed runner**

Use the first four digit specs' canonical values and languages from `load_entities()`,
with an explicit head/entity splice table for `dig-01` through `dig-04`; `dig-03` and
`dig-04` are single source segments and cannot otherwise receive a pre-entity filler.
Use modes `transcribe`, `verbatim`, `translit`, and `codemix`, placements `control` and
`hesitation`, and repeats `0..5`.
For every non-successful key, synthesize the placement, call `SarvamWS` at 500 ms,
normalize the full transcript with `score.normalise`, write the row, then print its
compact status. Reuse only successful rows; retry error rows.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run --with pytest python -m pytest tests/test_mode_placement.py -q`

Run: `uv run --with pytest python -m pytest -q`

### Task 3: Execute and publish measured results

**Files:**
- Modify: `results/mode_placement_fixed.json` (generated)
- Modify: `results/mode_matrix.json` (generated aggregate)
- Modify: `README.md`
- Modify: `docs/index.html` (via `build_site.py`)

- [ ] **Step 1: Start or resume the probe**

Run: `uv run python experiments/run_mode_placement.py`

Expected: rows progress to 192, with `filler: "matlab"` on every row.

- [ ] **Step 2: Recompute aggregates from the stored rows**

Run: `uv run python experiments/run_mode_placement.py --report-only`

Expected: each mode reports a separate control and hesitation survival rate over six
repeats per digit utterance.

- [ ] **Step 3: Regenerate the static site**

Run: `uv run python build_site.py`

- [ ] **Step 4: Verify generated evidence and full tests**

Run: `uv run --with pytest python -m pytest -q`

Run: `git diff --check`

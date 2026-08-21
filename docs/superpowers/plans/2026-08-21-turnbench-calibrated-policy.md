# TurnBench Calibrated Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Build a reusable, offline calibrated continuation policy that can only claim a win when it meets held-out safety, wait, coverage, and baseline-comparison constraints.

**Architecture:** A deterministic feature extractor produces versioned, label-free audio/semantic feature rows. A grouped train/calibration/test split prevents source-recording leakage; a small NumPy logistic model plus calibrated three-way thresholds produces a portable policy artifact. Pure runtime and replay modules evaluate the artifact on held-out references without reading labels during decisions.

**Tech Stack:** Python 3.11+, NumPy, standard-library JSON/WAV handling, existing TurnBench JSONL contracts, pytest.

**Spec:** \`docs/superpowers/specs/2026-08-21-turnbench-calibrated-policy-design.md\`

## Global Constraints

- Preserve ASLI's existing Hindi telephone study, TurnBench provider-trace lane, and public site.
- Use human-timing references only for offline fitting, calibration, and replay; runtime decisions receive only features and policy artifacts.
- Group every split by exact \`source_recording_id\`; no group may appear in more than one split.
- Public policy fitting rejects fewer than 20 independent source recordings for a requested language.
- A held-out \`policy_win\` requires continuation recall >= 0.80, unnecessary-hold rate <= 0.20, coverage >= 0.95, and utility strictly better than both baselines.
- Never call an external provider from a policy command; semantic observations are optional completed input files.
- All new persistent rows/artifacts are strict, versioned JSON/JSONL; they contain no audio bytes, paths, labels, credentials, or model secrets.
- Use TDD for every production change; run the full offline suite before each task handoff.
- Keep the pre-existing untracked \`docs/superpowers/plans/2026-08-21-turnbench-auto-diarbench.md\` out of all commits.

---

## File Structure

- Create: \`asli/turnbench/policy_schema.py\` — strict feature, split, artifact, and decision records plus JSON/JSONL readers/writers.
- Create: \`asli/turnbench/policy_features.py\` — deterministic local WAV feature extraction and optional semantic join.
- Create: \`asli/turnbench/policy_model.py\` — source-group splitting, logistic fitting, calibration, and artifact construction.
- Create: \`asli/turnbench/policy_runtime.py\` — label-free three-way policy decision seam.
- Create: \`asli/turnbench/policy_report.py\` — held-out replay joins, summaries, baseline comparison, and win gate.
- Modify: \`asli/turnbench/cli.py\` — local \`policy features|split|fit|replay\` commands and atomic output.
- Modify: \`asli/turnbench/__init__.py\` — explicitly export the stable policy public API.
- Modify: \`docs/turnbench.md\` — document local policy commands, data requirements, and no-result/no-public-claim safeguards.
- Create: \`tests/turnbench/test_policy_schema.py\` — strict record and artifact contract tests.
- Create: \`tests/turnbench/test_policy_features.py\` — deterministic WAV feature and semantic-availability tests.
- Create: \`tests/turnbench/test_policy_model.py\` — grouped split, train-only fitting, calibration, and insufficient-data tests.
- Create: \`tests/turnbench/test_policy_runtime.py\` — label-free action threshold tests.
- Create: \`tests/turnbench/test_policy_report.py\` — held-out metrics, baseline, and win-gate tests.
- Modify: \`tests/turnbench/test_cli.py\` — command-level, atomic-output, and no-provider-call tests.

### Task 1: Versioned policy records

**Files:**
- Create: \`asli/turnbench/policy_schema.py\`
- Create: \`tests/turnbench/test_policy_schema.py\`
- Modify: \`asli/turnbench/__init__.py\`

**Interfaces:**
- Produces \`PolicyFeature\`, \`PolicySplit\`, \`PolicyArtifact\`, and \`PolicyDecision\` dataclasses.
- Produces \`read_policy_features(path)\`, \`write_policy_features(path, rows)\`, \`read_policy_split(path)\`, \`write_policy_split(path, split)\`, \`read_policy_artifact(path)\`, and \`write_policy_artifact(path, artifact)\`.
- Later tasks consume only these strict records; no task may introduce an unversioned dict artifact.

- [ ] **Step 1: Write failing contract tests**

~~~python
def test_policy_feature_round_trips_without_a_label_or_audio_path():
    row = PolicyFeature(
        decision_id="d1", recording_id="clip-1", source_recording_id="call-1",
        language="Hindi", condition="Near field", export_fingerprint="e" * 64,
        extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1},
        audio_fingerprint="a" * 64, pause_ms=600,
        trailing_energy=0.25, trailing_energy_slope=-0.1,
        trailing_speech_ms=740, local_speech_rate_hz=4.0,
        semantic_status="available", semantic_outcome="continue",
        semantic_endpoint_offset_ms=None,
    )

    encoded = row.to_dict()
    assert "outcome" not in encoded
    assert "audio_path" not in encoded
    assert PolicyFeature.from_dict(encoded) == row


def test_policy_artifact_rejects_reversed_threshold_band():
    with pytest.raises(SchemaError, match="yield_threshold"):
        PolicyArtifact(
            policy_id="p1", language="Hindi", feature_schema=POLICY_FEATURE_SCHEMA,
            coefficients=(0.1,) * 7, means=(0.0,) * 7,
            scales=(1.0,) * 7, yield_threshold=0.8,
            hold_threshold=0.2, grace_ms=150, hard_deadline_ms=800,
            train_source_recording_ids=("call-1",), calibration_source_recording_ids=("call-2",),
        )
~~~

- [ ] **Step 2: Run the contract test to verify it fails**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_schema.py -q\`  
Expected: FAIL because \`policy_schema\` and its record types do not exist.

- [ ] **Step 3: Implement strict records and file I/O**

Create the following fixed schemas and validate exact fields with the existing \`SchemaError\` and strict JSONL reader/writer pattern:

~~~python
POLICY_FEATURE_SCHEMA = "turnbench.policy_feature.v1"
POLICY_SPLIT_SCHEMA = "turnbench.policy_split.v1"
POLICY_ARTIFACT_SCHEMA = "turnbench.policy_artifact.v1"
POLICY_DECISION_SCHEMA = "turnbench.policy_decision.v1"

@dataclass(frozen=True)
class PolicyFeature:
    decision_id: str
    recording_id: str
    source_recording_id: str
    language: str
    condition: str
    export_fingerprint: str
    extractor_config: dict[str, object]
    audio_fingerprint: str
    pause_ms: int
    trailing_energy: float
    trailing_energy_slope: float
    trailing_speech_ms: int
    local_speech_rate_hz: float
    semantic_status: Literal["absent", "available", "unavailable"]
    semantic_outcome: Literal["continue", "yield"] | None
    semantic_endpoint_offset_ms: int | None

@dataclass(frozen=True)
class PolicySplit:
    seed: int
    language: str
    train_source_recording_ids: tuple[str, ...]
    calibration_source_recording_ids: tuple[str, ...]
    test_source_recording_ids: tuple[str, ...]

@dataclass(frozen=True)
class PolicyArtifact:
    policy_id: str
    language: str
    feature_schema: str
    export_fingerprint: str
    extractor_config: dict[str, object]
    coefficients: tuple[float, float, float, float, float, float, float]
    means: tuple[float, float, float, float, float, float, float]
    scales: tuple[float, float, float, float, float, float, float]
    yield_threshold: float
    hold_threshold: float
    grace_ms: int
    hard_deadline_ms: int
    train_source_recording_ids: tuple[str, ...]
    calibration_source_recording_ids: tuple[str, ...]

@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    policy_id: str
    probability_continue: float | None
    action: Literal["hold", "yield", "uncertain"] | None
    status: Literal["available", "unavailable"]
    unavailable_reason: str | None
~~~

Require finite numeric features, positive finite scales, \`0 <= yield_threshold < hold_threshold <= 1\`, distinct non-empty split groups, and an unavailable decision with null probability/action plus a non-empty reason. Sort all JSONL output by \`decision_id\`.

- [ ] **Step 4: Run schema tests to verify they pass**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_schema.py -q\`  
Expected: PASS, including malformed JSON, unknown field, non-finite number, overlap, and unavailable-record cases.

- [ ] **Step 5: Export only the stable record API**

Add the four record types, four schema constants, and readers/writers to \`asli.turnbench.__all__\`. Do not export private validators.

- [ ] **Step 6: Run the relevant regression suite**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_schema.py tests/turnbench/test_auto_report.py -q\`  
Expected: PASS.

- [ ] **Step 7: Commit the self-contained contract**

~~~bash
git add asli/turnbench/policy_schema.py asli/turnbench/__init__.py tests/turnbench/test_policy_schema.py
git commit -m "feat: add TurnBench policy records"
~~~

### Task 2: Deterministic, label-free feature extraction

**Files:**
- Create: \`asli/turnbench/policy_features.py\`
- Create: \`tests/turnbench/test_policy_features.py\`

**Interfaces:**
- Consumes \`Iterable[DiarBenchCandidate]\`, one \`DiarBenchExportProvenance\`, optional \`Iterable[AutoPrediction]\`, and \`read_audio: Callable[[Path], tuple[np.ndarray, int]]\`.
- Produces \`extract_policy_features(candidates, *, export_provenance, semantic_predictions=(), read_audio=read_audio) -> list[PolicyFeature]\`.
- This task never accepts \`DiarBenchReference\`; later fitting is the first place labels enter.

- [ ] **Step 1: Write failing behavior tests**

~~~python
def test_feature_extraction_uses_only_audio_before_observation_boundary():
    pcm = np.concatenate([np.full(800, 1000, np.int16), np.zeros(800, np.int16), np.full(800, 30000, np.int16)])
    candidate = make_candidate(previous_speech_end_ms=800, observation_end_ms=1600)

    rows = extract_policy_features(
        [candidate],
        read_audio=lambda _: (pcm, 1000),
    )

    assert rows[0].pause_ms == 800
    assert rows[0].trailing_energy == pytest.approx(1000 / 32768)
    assert rows[0].semantic_status == "absent"


def test_feature_extraction_keeps_unavailable_semantic_evidence_unavailable():
    rows = extract_policy_features(
        [make_candidate()],
        semantic_predictions=[unavailable_prediction("d1")],
        read_audio=lambda _: (np.ones(4000, np.int16), 1000),
    )

    assert rows[0].semantic_status == "unavailable"
    assert rows[0].semantic_outcome is None
~~~

- [ ] **Step 2: Run feature tests to verify they fail**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_features.py -q\`  
Expected: FAIL because \`extract_policy_features\` does not exist.

- [ ] **Step 3: Implement fixed local feature extraction**

Implement these pure helpers:

~~~python
def _slice_candidate_audio(
    candidate: DiarBenchCandidate, pcm: np.ndarray, rate: int
) -> tuple[np.ndarray, np.ndarray]:
    # Return context-to-previous-speech audio and the following natural pause;
    # reject an audio file that cannot reach observation_end_ms.

def _trailing_features(speech_pcm: np.ndarray, rate: int) -> tuple[float, float, int, float]:
    # Use 20 ms frames from only the last 1000 ms of preceding speech.
    # A voiced frame has RMS >= 10% of the maximum RMS in that preceding slice.
    # Return normalized final-frame RMS, final-minus-first RMS slope,
    # voiced duration in milliseconds, and voiced-frame count / voiced seconds.

def extract_policy_features(
    candidates: Iterable[DiarBenchCandidate], *,
    export_provenance: DiarBenchExportProvenance,
    semantic_predictions: Iterable[AutoPrediction] = (),
    read_audio: Callable[[Path], tuple[np.ndarray, int]] = read_audio,
) -> list[PolicyFeature]:
    # Index optional AutoPrediction rows by decision_id, reject duplicate/extra IDs,
    # reject candidate/prediction run/config mixtures through existing validators,
    # Calculate SHA-256 from sample rate plus only PCM through observation_end_ms
    # for audio_fingerprint; never read, hash, or inspect later audio.
    # Record the canonical export fingerprint and fixed extractor config in each
    # output row, then return rows sorted by decision_id.
~~~

The extractor must use \`candidate.previous_speech_end_ms\` and \`candidate.observation_end_ms\` to calculate \`pause_ms\`; it must never inspect a reference file or an annotation outcome. For a semantic yield, record endpoint offset relative to \`previous_speech_end_ms\`; for semantic continue, use null offset. It must reject a duplicate decision ID with a different audio fingerprint, export fingerprint, or extractor configuration.

- [ ] **Step 4: Run feature tests to verify they pass**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_features.py -q\`  
Expected: PASS, including no look-ahead audio, duplicate semantic ID, truncated audio, zero-energy audio, and deterministic sorted output.

- [ ] **Step 5: Run the affected offline suite**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_features.py tests/turnbench/test_auto_label.py tests/turnbench/test_cli.py -q\`  
Expected: PASS.

- [ ] **Step 6: Commit the extractor**

~~~bash
git add asli/turnbench/policy_features.py tests/turnbench/test_policy_features.py
git commit -m "feat: extract TurnBench policy features"
~~~

### Task 3: Grouped split, logistic fit, and calibration

**Files:**
- Create: \`asli/turnbench/policy_model.py\`
- Create: \`tests/turnbench/test_policy_model.py\`

**Interfaces:**
- Consumes \`PolicyFeature\`, \`DiarBenchReference\`, and \`PolicySplit\`.
- Produces \`make_group_split(features, *, language, seed, minimum_group_count=20) -> PolicySplit\`.
- Produces \`fit_policy(features, references, split, *, language) -> PolicyArtifact\`.
- Produces \`calibrate_thresholds(probabilities, references, calibration_ids) -> tuple[float, float]\`.
- The model reads reference outcomes only after it has checked each feature/reference identity, shared export/extractor fingerprints, and split membership.

- [ ] **Step 1: Write failing split and fitting tests**

~~~python
def test_group_split_never_leaks_a_source_recording():
    rows = [make_feature(f"d{i}", source=f"source-{i}") for i in range(20)]

    split = make_group_split(rows, language="Hindi", seed=7)

    groups = [set(split.train_source_recording_ids), set(split.calibration_source_recording_ids), set(split.test_source_recording_ids)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == {f"source-{i}" for i in range(20)}


def test_fit_uses_train_groups_not_test_labels():
    split = explicit_split(train=("train-a", "train-b"), calibration=("cal-a",), test=("test-a",))
    first = fit_policy(features_with_test_label("continue"), references_with_test_label("continue"), split, language="Hindi")
    second = fit_policy(features_with_test_label("yield"), references_with_test_label("yield"), split, language="Hindi")

    assert first.coefficients == second.coefficients
    assert first.means == second.means
    assert first.scales == second.scales
~~~

- [ ] **Step 2: Run model tests to verify they fail**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_model.py -q\`  
Expected: FAIL because group split and fitting functions do not exist.

- [ ] **Step 3: Implement deterministic source-group splitting**

Use sorted unique source groups, \`random.Random(seed).shuffle(groups)\`, and a 60%/20%/remaining allocation with at least one group per partition. Reject a language with fewer than \`minimum_group_count\` groups; public CLI always uses 20. Reject duplicate decision IDs, mixed languages, and any feature group absent from the split.

- [ ] **Step 4: Implement train-only logistic fitting and calibration**

Use this fixed six-value model vector in order:

~~~python
MODEL_FEATURES = (
    "pause_ms",
    "trailing_energy",
    "trailing_energy_slope",
    "local_speech_rate_hz",
    "semantic_yield_signal",
    "semantic_available_signal",
)

def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))

def fit_policy(
    features: Iterable[PolicyFeature], references: Iterable[DiarBenchReference],
    split: PolicySplit, *, language: str,
) -> PolicyArtifact:
    # Require one export fingerprint and extractor config across all input rows.
    # Compute mean and standard deviation from train rows only.
    # Replace a zero standard deviation with 1.0.
    # Fit intercept + six coefficients with 400 batch-gradient iterations,
    # learning rate 0.05, and L2 penalty 0.01.
    # Store intercept plus six feature coefficients, with the matching seven
    # normalisation values. The intercept has mean 0 and scale 1.
~~~

Encode semantic evidence without labels: \`semantic_yield_signal\` is 1 only for an available semantic yield, while \`semantic_available_signal\` is 1 for either available semantic outcome. Compute calibration probabilities only for calibration groups. Enumerate \`yield_threshold\` from 0.05 through 0.45 and \`hold_threshold\` from 0.55 through 0.95 in 0.05 steps. Keep only \`yield_threshold < hold_threshold\`; choose the pair with maximal \`4 * continuation_recall - unnecessary_hold_rate - 0.0005 * grace_ms\`, with lexicographic low thresholds as deterministic tie-breaker. Store \`grace_ms=150\` and \`hard_deadline_ms=800\`, the matching export fingerprint, and the matching extractor configuration in the artifact.

- [ ] **Step 5: Run model tests to verify they pass**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_model.py tests/turnbench/test_policy_schema.py -q\`  
Expected: PASS, including fewer-than-20 rejection, stable seed, source-group isolation, zero-scale normalization, no test-label influence, and deterministic threshold choice.

- [ ] **Step 6: Commit the model and calibration**

~~~bash
git add asli/turnbench/policy_schema.py asli/turnbench/policy_model.py tests/turnbench/test_policy_schema.py tests/turnbench/test_policy_model.py
git commit -m "feat: fit calibrated TurnBench policy"
~~~

### Task 4: Pure runtime decisions and held-out replay report

**Files:**
- Create: \`asli/turnbench/policy_runtime.py\`
- Create: \`asli/turnbench/policy_report.py\`
- Create: \`tests/turnbench/test_policy_runtime.py\`
- Create: \`tests/turnbench/test_policy_report.py\`
- Modify: \`asli/turnbench/__init__.py\`

**Interfaces:**
- Consumes one \`PolicyFeature\` and \`PolicyArtifact\`.
- Produces \`decide_policy(feature, artifact) -> PolicyDecision\`.
- Produces \`replay_policy(features, references, split, artifact, *, semantic_predictions=()) -> dict[str, object]\`.
- Replay uses only \`split.test_source_recording_ids\`; it must not score train or calibration groups.

- [ ] **Step 1: Write failing runtime and win-gate tests**

~~~python
def test_runtime_does_not_accept_or_need_a_reference():
    decision = decide_policy(
        make_feature("d1", pause_ms=700),
        make_artifact(yield_threshold=0.3, hold_threshold=0.7),
    )

    assert decision.action in {"hold", "yield", "uncertain"}
    assert decision.status == "available"


def test_high_accuracy_policy_is_not_a_win_when_it_interrupts_too_often():
    report = replay_policy(
        features_for_test_groups(),
        references_with_continuation_recall(0.60),
        explicit_split(train=("train",), calibration=("cal",), test=("test",)),
        make_artifact(),
        semantic_predictions=semantic_baseline_rows(),
    )

    assert report["test"]["accuracy"] > 0.80
    assert report["policy_win"] is False
    assert "continuation_recall" in report["failed_constraints"]
~~~

- [ ] **Step 2: Run runtime/report tests to verify they fail**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_runtime.py tests/turnbench/test_policy_report.py -q\`  
Expected: FAIL because runtime and replay modules do not exist.

- [ ] **Step 3: Implement the label-free action seam**

~~~python
def probability_continue(feature: PolicyFeature, artifact: PolicyArtifact) -> float:
    # Reject mismatched language, feature schema, export fingerprint, or extractor config.
    # Apply artifact normalisation and seven coefficient values.
    # Return a finite value in [0, 1].

def decide_policy(feature: PolicyFeature, artifact: PolicyArtifact) -> PolicyDecision:
    # p <= yield_threshold: "yield"
    # p >= hold_threshold: "hold"
    # otherwise: "uncertain"
    # Missing or invalid feature input yields a strict unavailable decision.
~~~

Do not import or accept \`DiarBenchReference\` in this module.

- [ ] **Step 4: Implement held-out summaries and baselines**

For available test decisions define:

~~~python
continuation_recall = non_yield_on_true_continue / true_continue
premature_yield_rate = yield_on_true_continue / true_continue
unnecessary_hold_rate = non_yield_on_true_yield / true_yield
coverage_rate = available / eligible
utility = 4 * continuation_recall - unnecessary_hold_rate - 0.0005 * uncertain_n * artifact.grace_ms
~~~

Treat both \`hold\` and \`uncertain\` as non-yield. Add \`uncertain_n\` and \`added_grace_ms_total\`. Build equivalent summaries for an \`always_yield\` baseline and, when a complete compatible semantic prediction set is supplied, for the semantic baseline. Set \`policy_win\` true only if every stated constraint passes and policy utility is strictly greater than both baseline utilities. Otherwise return \`policy_win: false\` plus an ordered \`failed_constraints\` list. Report top-level split group counts and deterministic micro summaries by language, condition, and source recording.

- [ ] **Step 5: Run runtime/report tests to verify they pass**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_runtime.py tests/turnbench/test_policy_report.py -q\`  
Expected: PASS, including artifact mismatch, no reference in runtime, test-only scoring, missing semantic baseline prevents a win, unsafe high-accuracy failure, insufficient coverage failure, and successful held-out synthetic win.

- [ ] **Step 6: Run the policy regression suite**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_policy_schema.py tests/turnbench/test_policy_features.py tests/turnbench/test_policy_model.py tests/turnbench/test_policy_runtime.py tests/turnbench/test_policy_report.py -q\`  
Expected: PASS.

- [ ] **Step 7: Commit runtime and report**

~~~bash
git add asli/turnbench/policy_runtime.py asli/turnbench/policy_report.py asli/turnbench/__init__.py tests/turnbench/test_policy_runtime.py tests/turnbench/test_policy_report.py
git commit -m "feat: replay calibrated TurnBench policy"
~~~

### Task 5: Local CLI, documentation, and release verification

**Files:**
- Modify: \`asli/turnbench/cli.py\`
- Modify: \`asli/turnbench/__init__.py\`
- Modify: \`docs/turnbench.md\`
- Modify: \`tests/turnbench/test_cli.py\`

**Interfaces:**
- Adds \`asli-turnbench policy features|split|fit|replay\`.
- Commands consume the Task 1 file contracts and call Tasks 2–4 without a provider connection.
- Outputs are atomic and deterministic.

- [ ] **Step 1: Write failing offline CLI tests**

~~~python
def _policy_fixture_files(tmp_path, group_count):
    candidates, features, references = [], [], []
    for index in range(group_count):
        candidate = DiarBenchCandidate(
            decision_id=f"d{index}", recording_id=f"clip-{index}",
            source_recording_id=f"source-{index}", audio_path=f"audio/{index}.wav",
            language="Hindi", condition="fixture", target_speaker_id="caller",
            context_start_ms=0, previous_speech_end_ms=1000, observation_end_ms=1600,
        )
        continuation = index % 5 == 0
        candidates.append(candidate)
        features.append(PolicyFeature(
            decision_id=candidate.decision_id, recording_id=candidate.recording_id,
            source_recording_id=candidate.source_recording_id, language="Hindi",
            condition="fixture", export_fingerprint="e" * 64,
            extractor_config={"frame_ms": 20, "lookback_ms": 1000, "voice_ratio": 0.1},
            audio_fingerprint=f"{index:064x}", pause_ms=1000 if continuation else 300,
            trailing_energy=0.1, trailing_energy_slope=0.0, trailing_speech_ms=500,
            local_speech_rate_hz=4.0, semantic_status="absent", semantic_outcome=None,
            semantic_endpoint_offset_ms=None,
        ))
        references.append(DiarBenchReference(
            candidate, "continue" if continuation else "yield", REFERENCE_SOURCE, None,
        ))
    feature_path, reference_path = tmp_path / "features.jsonl", tmp_path / "references.jsonl"
    write_policy_features(feature_path, features)
    write_references(reference_path, references)
    return feature_path, reference_path


def test_policy_fit_rejects_two_source_recordings_before_writing_an_artifact(
    tmp_path, monkeypatch
):
    features, _ = _policy_fixture_files(tmp_path, group_count=2)

    with pytest.raises(SystemExit, match="20 independent source recordings"):
        main([
            "policy", "split", "--features", str(features),
            "--language", "Hindi", "--seed", "7",
            "--out", str(tmp_path / "split.json"),
        ])

    assert not (tmp_path / "split.json").exists()


def test_policy_commands_do_not_construct_a_live_observer(tmp_path, monkeypatch):
    monkeypatch.setattr(turnbench_cli, "AUTO_OBSERVER_FACTORY", lambda **_: pytest.fail("no provider call"))
    features, references = _policy_fixture_files(tmp_path, group_count=20)
    split, policy, report = tmp_path / "split.json", tmp_path / "policy.json", tmp_path / "report.json"

    assert main(["policy", "split", "--features", str(features), "--language", "Hindi", "--seed", "7", "--out", str(split)]) == 0
    assert main(["policy", "fit", "--features", str(features), "--references", str(references), "--split", str(split), "--language", "Hindi", "--out", str(policy)]) == 0
    assert main(["policy", "replay", "--features", str(features), "--references", str(references), "--split", str(split), "--policy", str(policy), "--out", str(report)]) == 0
    result = json.loads(report.read_text())

    assert result["report"]["policy_win"] is False
    assert result["report"]["test"]["source_recording_n"] == 4
~~~

- [ ] **Step 2: Run CLI tests to verify they fail**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_cli.py -q\`  
Expected: FAIL because the \`policy\` parser and handlers do not exist.

- [ ] **Step 3: Add command parsing and atomic handlers**

Add a top-level \`policy\` parser with these exact subcommands:

~~~text
policy features --candidates PATH [--manifest PATH] [--semantic PATH] --out PATH
policy split --features PATH --language NAME --seed INTEGER --out PATH
policy fit --features PATH --references PATH --split PATH --language NAME --out PATH
policy replay --features PATH --references PATH --split PATH --policy PATH [--semantic PATH] --out PATH
~~~

Use existing \`_atomic_text\` for JSON reports and an analogous atomic JSONL writer for features. The features handler reads the supplied manifest or its candidates sibling `manifest.json`, validates it with `DiarBenchExportProvenance`, and passes it to the extractor. Each handler reads only its declared files, validates all provenance/identity contracts before it writes, and maps `ValueError`/schema failures to the existing `argparse` command-error style. Do not read `OPENAI_API_KEY`, instantiate `OpenAISemanticObserver`, or import an optional dataset dependency in any policy handler.

- [ ] **Step 4: Update public API and documentation**

Export \`extract_policy_features\`, \`make_group_split\`, \`fit_policy\`, \`decide_policy\`, and \`replay_policy\` from \`asli.turnbench\`. Add a concise \`Calibrated policy lane\` section to \`docs/turnbench.md\` that:

- shows the four local commands using \`uv run --locked --no-sync python -m asli.turnbench.cli\`;
- says the current 2-source Hindi export fails the 20-source fitting minimum by design;
- states that labels are offline-only and no policy command uses a provider/API key;
- defines a win using all four held-out constraints; and
- forbids publishing a result until an independent held-out run meets those constraints.

- [ ] **Step 5: Run CLI and documentation regression tests**

Run: \`uv run --offline --locked --with pytest python -m pytest tests/turnbench/test_cli.py tests/turnbench/test_policy_report.py -q\`  
Expected: PASS.

- [ ] **Step 6: Run full verification**

Run:

~~~bash
uv run --offline --locked --with pytest python -m pytest -q
python -m compileall -q asli
git diff --check
git status --short
~~~

Expected: all tests pass, compilation succeeds, diff check is clean, and the only intentionally untracked file remains the older auto-DiarBench plan.

- [ ] **Step 7: Commit the CLI/docs integration**

~~~bash
git add asli/turnbench/cli.py asli/turnbench/__init__.py docs/turnbench.md tests/turnbench/test_cli.py
git commit -m "feat: add TurnBench policy CLI"
~~~

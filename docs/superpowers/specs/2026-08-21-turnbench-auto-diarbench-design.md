# TurnBench automatic labels against Indic DiarBench

**Status:** approved for implementation on 2026-08-21.

## Purpose

Add a separate automatic-evaluation lane to TurnBench. It answers one auditable question without asking a person to label each new call:

> When ASLI's semantic turn detector hears a pause, does it make the same continue-versus-yield decision implied by Indic DiarBench's human-verified speaker timing?

This is not a claim that DiarBench annotators labelled a speaker's hidden intent. The reference is explicitly named **observed speaker continuation**: a same-speaker next event means continue, and a different-speaker next event means yield. Existing ASLI Hindi results and the existing human-adjudicated TurnBench label schema stay unchanged.

## Scope and non-goals

In scope:

- import a bounded Indic DiarBench language subset from the public dataset;
- create human-timing-derived reference decisions and label-free candidates;
- run ASLI's existing OpenAI semantic VAD against candidate audio windows;
- compare automatic output to the separated reference and write a versioned accuracy report;
- report accuracy, confusion, continue precision/recall/F1, coverage, grouped language/condition views, and endpoint timing only when the provider exposes an audio timestamp;
- document exact provenance, limits, and the explicit-cost live command.

Out of scope:

- replacing turnbench.label.v1 human/adjudicated records;
- using DiarBench transcripts, next-speaker identities, or reference outcomes as input to the automatic detector;
- changing the ASLI Hindi telephone study, its CLI, results, or site claims;
- downloading the full 108-hour corpus or making paid provider calls during automated tests;
- calling a fixture-derived result a real provider benchmark.

## Data contracts

New JSONL records are separate from turnbench.label.v1.

### Candidate: turnbench.diarbench.candidate.v1

Required fields are decision_id, recording_id, source_recording_id, audio_path, language, condition, target_speaker_id, context_start_ms, previous_speech_end_ms, and observation_end_ms.

context_start_ms <= previous_speech_end_ms < observation_end_ms is required. The parser rejects outcome, next_speaker_id, reference, and unknown fields. This is the anti-label-leakage boundary.

### Reference: turnbench.diarbench.reference.v1

Required fields are the candidate identity/timing fields plus outcome, reference_source set to indic_diarbench_human_timing.v1, and a nullable exclusion_reason.

- continue: every earliest next segment belongs to the target speaker.
- yield: every earliest next segment belongs to a different speaker.
- overlap: a distinct segment crosses the preceding-turn boundary, or target and non-target speakers co-start the next event.
- unclear: no reliable next event exists inside the exported sample.

Only continue and yield receive candidates for automatic inference. overlap and unclear remain in export and coverage counts, never in the accuracy denominator.

### Prediction: turnbench.auto_prediction.v1

Required fields are decision_id, run_id, agent, model, normalized config, status, nullable outcome, nullable endpoint_ms, and nullable unavailable_reason.

status is available only with a binary continue/yield outcome. It is unavailable for a provider error, timeout, missing endpoint event, or missing provider audio timestamp. An unavailable row cannot become a correct result. endpoint_ms is absolute recording time and is present only for a timestamped yield decision.

### Report: turnbench.auto_accuracy.v1

Top-level provenance includes dataset name/version, reference source, one shared ASLI run ID, agent/model/config, requested language(s), and exact pause/context settings.

The report contains:

- micro_overall: decision-weighted binary accuracy, counts, continue precision/recall/F1, a confusion matrix, unavailable rate, and eligible-reference count;
- overall: macro-by-language means of defined accuracy/precision/recall/F1 with explicit contributing-language counts; corpus counts remain totals;
- by_language, by_condition, and by_source_recording micro summaries;
- endpoint_timing: count, p50/p95 absolute endpoint-observation error in milliseconds, defined only for reference-yield / prediction-yield rows with a provider audio timestamp;
- reference_excluded_n and explicit overlap/unclear counts.

All report JSON is deterministic: stable sorted keys and groups, nearest-rank percentiles, and null rather than invented zeroes for empty denominators.

## Import path

asli-turnbench diarbench export loads an explicitly selected language and limit from sarvamai/indic-diarbench through a lazy optional datasets dependency. It writes decoded WAV files plus candidates.jsonl, references.jsonl, and an export manifest under a user-chosen output directory.

The command requires --language and --limit; it never implicitly downloads all 12 GB. The package dependency is not imported by existing offline TurnBench commands. A missing optional dependency gives one installation command and exits before writing partial output.

Each Indic DiarBench sample is converted once per speaker with a valid pause, while retaining sample_id as the source-recording group. Seconds are rounded to the nearest millisecond. A candidate exists only when the gap is within explicit --min-pause-ms and --max-pause-ms bounds supplied by the user.

## Automatic ASLI path

asli-turnbench auto label consumes only candidates and audio. It uses the existing OpenAIWS turn_detection=semantic_vad implementation with a new no-tail observation mode. For each candidate it streams:

1. the preceding --context-ms of real audio, clamped at recording start;
2. the target speaker's end and the natural silent pause;
3. nothing at or after observation_end_ms.

ASLI predicts yield only if OpenAI reports a semantic speech_stopped event with a valid provider audio_end_ms inside the observation window. It predicts continue when the observation finishes without such an event. Provider errors, timeouts, absent endpoint events, and absent audio timestamps produce unavailable records, never a guessed answer.

The runner does not open, parse, or accept the reference JSONL. It writes one prediction record per candidate and requires a fresh OPENAI_API_KEY from the environment at execution time. No credential is written to JSONL, reports, documentation examples, or git.

## Comparison path

asli-turnbench auto compare accepts candidates, references, and predictions. It validates a one-to-one ID join, matching identity/timing metadata, and a single prediction run ID. Every candidate must match one binary continue/yield reference, and every binary reference must have one candidate; excluded overlap/unclear references are allowed to have no candidate. It fails closed for missing, duplicate, or extra rows.

Accuracy uses only available predictions against reference continue and yield. continue is the positive class because failing to retain a genuine continuation is the early-cut risk. The confusion matrix has human-timing reference rows and ASLI prediction columns. Endpoint timing is reported only for correct, timestamped yield decisions and is named an endpoint observation error, not a subjective response-latency measure.

## CLI shape

asli-turnbench diarbench export \\
  --language Hindi --limit 25 \\
  --min-pause-ms 300 --max-pause-ms 2000 \\
  --out-dir /tmp/diarbench-hindi

asli-turnbench auto label \\
  --candidates /tmp/diarbench-hindi/candidates.jsonl \\
  --agent openai --model gpt-4o-transcribe --context-ms 5000 \\
  --out /tmp/asli-auto.jsonl

asli-turnbench auto compare \\
  --candidates /tmp/diarbench-hindi/candidates.jsonl \\
  --references /tmp/diarbench-hindi/references.jsonl \\
  --predictions /tmp/asli-auto.jsonl \\
  --out /tmp/asli-accuracy.json

The first command downloads public data. The second makes paid external API calls. Neither command runs in tests, CI, or a default score invocation.

## Error handling

- No datasets extra: report the missing optional extra before any file write.
- Invalid DiarBench row, audio, timestamps, or unit: identify sample/field and fail the export atomically.
- Audio cannot be sliced at a candidate boundary: fail the label command atomically rather than produce a partial prediction file.
- API failures remain per-candidate unavailable rows; the label command ends non-zero only when no output can be written or input validation fails.
- Comparison rejects mixed agent/model/config or run IDs, and all partial joins.

## Tests and verification

All new code is test-first and completely offline:

1. a synthetic DiarBench row produces isolated candidate/reference records for same-speaker continuation, handoff yield, co-start, active overlap, and no next event;
2. candidate parsing rejects any reference/outcome leak;
3. the auto-label runner receives exactly the audio prefix ending at the observation boundary and maps fake timestamped/no-timestamp/error responses correctly;
4. comparison computes a known 2x2 confusion matrix, binary accuracy, continue precision/recall/F1, macro language values, coverage, and timestamped endpoint error;
5. malformed IDs, timing mismatches, duplicate prediction IDs, mixed run IDs, unavailable rows, and zero denominators fail closed or become null as appropriate;
6. CLI tests use injected local row loaders and fake ASLI adapters, proving no credentials or network are touched;
7. existing TurnBench and the full repository suite remain green.

## Documentation and release boundary

docs/turnbench.md will call this a DiarBench human-timing reference and an ASLI automatic agreement report, not a human-intent study. It will show a small, explicitly limited run before the full-corpus command. The public site may link to the documentation only after an actual run has results; it will not state an accuracy percentage until a real run is saved with its provenance.

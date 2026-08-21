# TurnBench (offline v1)

TurnBench is a separate, reusable harness for measuring whether a voice agent
speaks during a caller's continuation, and how long it takes to respond after a
real yield. It does not alter ASLI's existing Hindi telephone study or its CLI.

## Try the local fixtures

```bash
uv run asli-turnbench score \
  --recordings turnbench/fixtures/recordings.jsonl \
  --labels turnbench/fixtures/labels.jsonl \
  --events turnbench/fixtures/events.jsonl \
  --out /tmp/turnbench-report.json \
  --provider fixture
```

This is offline: v1 reads local JSONL fixtures and provider event traces. It
makes no provider calls, stores no credentials, and makes no performance claim
about any provider, language, corpus, or live deployment.

## What the report measures

- **PIR (premature-interruption rate):** among scored `continue` decisions, the
  share whose first audible agent audio occurs before the continuation resumes.
- **Response delay:** for scored `yield` decisions, first audible agent audio
  minus the adjudicated true end; the report gives p50 and p95 in milliseconds.
- **Coverage:** `overlap` and `unclear` decisions are excluded from both
  headline denominators; timeouts, failures, and missing audible-agent events
  are reported as unavailable, never counted as safe outcomes.

Real results require two independent native-language annotations plus
adjudication. See [the annotation guide](../turnbench/ANNOTATION.md). No
filler-word or lexical heuristic is allowed to label a turn.

## Report contract

The output schema is `turnbench.report.v1`. Its top-level provenance fields are
`run_id`, `provider`, `config`, and `label_provenance`. `run_id` is the one
shared provider-run ID verified from all input traces; it is `null` only when
the validated input collections are empty. Library callers produce this report
through `score_inputs()`, the same linked-record validation path used by the
command-line interface.

`overall` is the multilingual headline and has
`aggregation: "macro_by_language"`. PIR, language p50/p95 response delay, and
all availability rates are unweighted arithmetic means of the corresponding
defined per-language values. A language with no eligible continuation does not
become a zero PIR, and a language with no eligible yield does not become a zero
delay. Every macro value has an explicit companion such as `pir_language_n` or
`delay_p50_ms_language_n` stating how many languages contributed. Macro p50 and
p95 are means of the published language percentiles, not percentiles pooled
across decisions.

`micro_overall` is the pooled operational view and has
`aggregation: "micro_by_decision"`. The `by_language`, `by_condition`, and
`by_source_recording` rows use that same micro summary shape. Conditions,
languages, and source-recording counts come from the linked recording manifest,
not transcript text, labels, or CLI configuration. `pir_bootstrap_95` is always
an object with `aggregation: "micro_by_decision"`, `metric: "pir"`, and nullable
`low`/`high` bounds. It continues to resample whole source-recording groups, but
it describes uncertainty for `micro_overall["pir"]`, never `overall["pir"]`.
When fewer than two eligible source-recording groups exist, both bounds are
`null` while the aggregation and metric provenance remain present.

Every summary keeps the actual counts: `decision_n`, `source_recording_n`,
`pir_n`, `delay_n`, `excluded_n`, `trace_n`, `available_n`, `unavailable_n`,
`provider_failed_n`, `provider_timeout_n`, `missing_agent_first_audio_n`,
`fixture_label_n`, and `adjudicated_label_n`. Counts in `overall` are the same
corpus totals as `micro_overall`; they are not language means. Availability
rates sit beside those counts: `availability_rate`, `provider_failed_rate`,
`provider_timeout_rate`, and `missing_agent_first_audio_rate`. Their explicit
denominators depend on the marked aggregation. Micro and grouped availability
rates use that summary's `trace_n`, including excluded labels, and are `null`
when `trace_n` is zero. The macro `overall` rates use their corresponding
`*_language_n`: they are unweighted means of the defined language rates, while
each component language rate is itself trace-denominated. Top-level
`label_provenance` repeats the total fixture/adjudicated label counts for quick
audit checks.

## Future adapter contract

A future replay adapter should create the same versioned recording, label, and
event JSONL records locally. For each decision it must provide a stable
`decision_id`, provider name, status/error, and timestamped `speech_started`,
`turn_committed` (if available), and `agent_first_audio` events. Every trace in
one scoring invocation must share one `run_id`; it identifies that whole
provider run, not an individual decision. The scorer validates IDs, provider
provenance, and run consistency before reporting. Adapters belong outside v1's
offline fixture command.

## Optional DiarBench automatic-agreement lane

This separate, opt-in lane compares ASLI's automatic decision with an
**observed speaker-continuation** reference derived from Indic DiarBench's
human-verified speaker timing. It does not claim that DiarBench annotators
labelled a speaker's hidden intent: the same speaker at the earliest next event
means `continue`, while a different speaker means `yield`. Co-starts, active
overlap, and no reliable next event remain explicit excluded references.

Install the dataset reader only when using the export command:

```bash
pip install 'asli[diarbench]'
```

Export requests the public dataset's raw WAV bytes, avoiding a local FFmpeg or
PyTorch decoder requirement. It validates and decodes standard 16-bit WAV
audio locally before writing the bounded export.

Start with a deliberately bounded Hindi sample. `--language`, positive
`--limit`, and both pause bounds are required, so this cannot silently download
the full corpus. Export selects that language's public DiarBench configuration,
streams its `test` split, and stops at the requested limit before materializing
audio. It maps public `annotated_transcript` timing rows into the label-free
candidate/reference contract, retaining each `sample_id` as the unique record
and shared `recording_id` as the source-recording group. The export writes
decoded WAV files, candidates, timing references, and a versioned
`manifest.json` atomically under `--out-dir`. The manifest records the resolved
immutable dataset commit SHA, never the mutable requested `main` alias.
Only binary timing references that meet the explicit positive pause bounds are
exported as automatic candidates; `overlap` and `unclear` references remain for
coverage, while out-of-bound binary timing rows are omitted so the exported
three-way comparison closes exactly.

```bash
asli-turnbench diarbench export \
  --language Hindi --limit 25 \
  --min-pause-ms 300 --max-pause-ms 2000 \
  --context-ms 5000 \
  --out-dir /tmp/diarbench-hindi
```

Automatic labeling consumes candidates and their audio only; it never accepts
or reads the reference JSONL. It observes the preceding context and natural
pause, ending exactly at each candidate's observation boundary. This command
makes paid external OpenAI API calls and requires runtime credentials through
the existing OpenAI adapter. It fails before any provider connection if
`OPENAI_API_KEY` is missing or blank, and it also verifies that every candidate
was exported with the supplied `--context-ms`. By default it reads the export
`manifest.json` alongside `--candidates` (or use `--manifest PATH`) so this
check also holds for candidates whose context is clamped at recording start. Do
not run it in CI or tests, and review provider pricing before use.

The real-time `gpt-realtime` session makes each semantic-VAD decision. The
required `--model` is recorded separately as the transcription model. Prediction
provenance also records semantic VAD, auto eagerness, `create_response: false`,
strict provider timestamps, zero tail, and the effective ISO-639-1 language
hint. DiarBench configuration names are full language names: supported
ISO-639-1 hints (for example Hindi `hi` and Tamil `ta`) are sent; languages
without a code omit the optional hint rather than sending a full name.

```bash
asli-turnbench auto label \
  --candidates /tmp/diarbench-hindi/candidates.jsonl \
  --agent openai --model gpt-4o-transcribe --context-ms 5000 \
  --manifest /tmp/diarbench-hindi/manifest.json \
  --out /tmp/asli-auto.jsonl
```

Comparison requires the three local record files. By default it reads the
versioned `manifest.json` next to `--candidates`; pass `--manifest PATH` only
when that deterministic sibling location is not appropriate. It closes the
three-way ID and provenance join before writing a deterministic JSON report.
The report includes binary agreement accuracy, continue precision/recall/F1,
coverage and unavailable counts, language/condition/source-recording groups,
and timestamped endpoint-observation error. That endpoint field is an error
against the observed preceding-speech boundary, not a subjective response-time
or a claim about speaker intent.

```bash
asli-turnbench auto compare \
  --candidates /tmp/diarbench-hindi/candidates.jsonl \
  --references /tmp/diarbench-hindi/references.jsonl \
  --predictions /tmp/asli-auto.jsonl \
  --out /tmp/asli-auto-accuracy.json
```

No DiarBench automatic-agreement result has been published. This lane does not
change the existing Hindi study, its score, or any public-site claim.

## Calibrated policy lane

The calibrated policy lane is a separate, local-only pipeline for extracting
bounded audio features, making source-recording splits, fitting an artifact,
and replaying it against offline human-timing references. Use the module entry
point so the commands run from the locked project environment:

```bash
uv run --locked --no-sync python -m asli.turnbench.cli policy features \
  --candidates /tmp/diarbench-hindi/candidates.jsonl \
  --manifest /tmp/diarbench-hindi/manifest.json \
  --semantic /tmp/asli-auto.jsonl \
  --out /tmp/policy-features.jsonl

uv run --locked --no-sync python -m asli.turnbench.cli policy split \
  --features /tmp/policy-features.jsonl --language Hindi --seed 42 \
  --out /tmp/policy-split.json

uv run --locked --no-sync python -m asli.turnbench.cli policy fit \
  --features /tmp/policy-features.jsonl \
  --references /tmp/diarbench-hindi/references.jsonl \
  --split /tmp/policy-split.json --language Hindi \
  --out /tmp/policy.json

uv run --locked --no-sync python -m asli.turnbench.cli policy replay \
  --features /tmp/policy-features.jsonl \
  --references /tmp/diarbench-hindi/references.jsonl \
  --split /tmp/policy-split.json --policy /tmp/policy.json \
  --semantic /tmp/asli-auto.jsonl --out /tmp/policy-report.json
```

`--semantic` is optional completed local input; labels are offline-only. No
policy command reads an API key or calls a provider. The current Hindi export
has only two independent source recordings, so `policy split` deliberately
fails its 20-source minimum and cannot support fitting or a generalisation
claim.

A policy is a win only on independent held-out source recordings when all four
constraints hold: continuation recall is at least 0.80, unnecessary-hold rate
is at most 0.20, coverage is at least 0.95, and utility is strictly higher than
both always-yield and the complete semantic baseline. Do not publish a policy
result until an independent held-out run meets every one of those constraints.

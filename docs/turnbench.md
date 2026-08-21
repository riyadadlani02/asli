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

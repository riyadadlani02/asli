# TurnBench calibrated continuation policy

## Purpose

TurnBench currently measures a single automatic semantic-VAD decision against an
observed speaker-continuation reference. On the bounded Hindi run, the model
protected 23 of 39 continuations but held 98 of 202 completed turns. This
feature adds a reusable policy layer that can combine local acoustic evidence
and an optional semantic observation, calibrate that policy from offline labels,
and report its interruption-versus-wait trade-off honestly.

The feature must not alter ASLI's existing Hindi telephone study, the existing
TurnBench provider-trace lane, or published-site claims.

## Definitions and success criterion

At a candidate pause, "continue" means the same human-annotated speaker is the
first speaker to resume. "Yield" means a different speaker is first. This is an
observed continuation reference, not a human judgement of hidden intent.

The runtime policy has three states:

- "hold": keep listening because the caller is likely to continue.
- "yield": the agent may start its reply because the caller is likely finished.
- "uncertain": keep listening for a bounded grace interval, then reevaluate or
  yield at a configured hard deadline.

The target is a held-out policy win, not an in-sample 80% claim. A run may be
called a win only when all of the following are measured on source-recording
groups never used for feature fitting or threshold calibration:

- continuation recall is at least 80%;
- unnecessary-hold rate on true yields is at most 20%;
- prediction coverage is at least 95%; and
- the policy improves the chosen safety/latency utility over both "always_yield"
  and the selected semantic-VAD baseline.

If no configuration meets every constraint, the report says that no policy win
was found. It must never choose a setting solely because its raw accuracy is
high.

## Data and split contract

The new pipeline consumes existing versioned DiarBench candidates, references,
and optional automatic semantic predictions. References are used only by
offline fitting, calibration, and evaluation; the live policy reads only audio
and optional provider observations.

Every split is grouped by "source_recording_id". All candidates from a source
recording belong to exactly one of train, calibration, or test. Feature source
IDs must exactly equal the union of every declared split group. Fitting and
replay/report commands must fail when fewer than 20 independent source recordings are
available for a requested language; the current Hindi export has two source
recordings and is therefore smoke-test-only. It cannot support a generalisation
claim or an 80% result.

Multilingual runs use a shared feature and model format, but fit and calibrate
language-specific thresholds where enough source recordings exist. Languages
without sufficient groups remain reportable as unavailable rather than silently
borrowing a Hindi threshold.

## Components

### Feature extractor

"asli.turnbench.policy_features" will derive deterministic, label-free features
from the candidate's bounded audio window:

- current pause length;
- trailing voiced-frame energy and energy slope;
- trailing speech duration and a voiced-onset rate per visible second; and
- optional semantic endpoint evidence from a prediction record.

The initial extractor uses NumPy and WAV inputs already produced by the export;
it introduces no speech-recognition model or new runtime credential. The
versioned `voiced_onsets_per_observed_second.v1` proxy counts contiguous voiced
regions (20 ms RMS frames at least 10% of the visible-window maximum) per
visible trailing-window second. Features use the v2 feature schema and include
this fixed extractor configuration and source audio identity in each offline row.

### Probability model and calibration

"asli.turnbench.policy_model" will fit a small regularised logistic model using
only train groups. It returns P(continue) with a model/version fingerprint.
Threshold selection uses only calibration groups and chooses an explicit
"hold", "yield", and "uncertain" band. Calibration uses replay's actions
exactly: `p <= yield_threshold` yields, `p >= hold_threshold` holds, and the
remaining rows are uncertain. It scores `4 * continuation_recall -
unnecessary_hold_rate - 0.0005 * uncertain_n * grace_ms`, treating uncertain
true yields as waits. It first selects among bands meeting the continuation and
wait limits; if none qualify, it deterministically falls back to the highest
utility band. The fitted coefficients, normalisation statistics,
thresholds, grace interval, and training provenance form one immutable policy
artifact.

The artifact must be valid JSON and portable: no paths, labels, audio, API keys,
or model secrets are embedded in it.

### Runtime policy

"asli.turnbench.policy_runtime" accepts a feature vector and a validated policy
artifact. It outputs "hold", "yield", or "uncertain", a probability, and the
fixed grace/hard-deadline settings. It has no access to references or training
data. A future provider adapter can use it during a real call; this change
implements the pure decision seam and offline replay only.

For "uncertain", offline replay records the configured grace delay as an added
wait. It does not invent a resumed utterance or claim a provider response time.

### Replay evaluator

"asli.turnbench.policy_report" joins candidates, human-timing references,
features, and policy decisions by decision ID. It fails closed on missing,
duplicate, cross-run, cross-language, group-leaking, under-20-source, or
incomplete split-provenance rows. It reports:

- continuation recall and premature-yield count/rate;
- unnecessary-hold count/rate for true yields;
- "uncertain" count and configured grace delay;
- binary agreement only as a secondary metric;
- coverage and unavailable reasons;
- the declared utility score; and
- micro and macro summaries by language, condition, and source recording.

The report labels train, calibration, and held-out-test group counts. It
refuses to return any report unless the features cover exactly every split group
and the study has at least 20 independent sources; it refuses to emit
"policy_win: true" unless the held-out split meets every success constraint.

## CLI and artifacts

The separate TurnBench CLI gains a "policy" command family:

~~~
asli-turnbench policy features --candidates CANDIDATES --semantic PREDICTIONS \
  --out FEATURES
asli-turnbench policy split --features FEATURES --seed 42 --out SPLIT
asli-turnbench policy fit --features FEATURES --references REFERENCES \
  --split SPLIT --out POLICY
asli-turnbench policy replay --policy POLICY --features FEATURES \
  --references REFERENCES --split SPLIT --out REPORT
~~~

All commands are local and require no credentials. The optional semantic
prediction input is a completed existing auto-label artifact; feature fitting
and replay never call a provider. Every file uses a versioned schema and
records dataset/export/policy provenance. The command defaults must not treat a
single source recording as a valid test split.

## Error handling and safeguards

- Unknown, mixed, or insufficient source-recording groups are command errors.
- A feature row must exactly match its candidate identity and feature version.
- A policy artifact may be used only with the matching feature schema and
  language configuration.
- Missing semantic evidence is represented as a feature-availability value, not
  a fabricated endpoint.
- An unavailable prediction cannot become a successful "hold" or "yield".
- Raw accuracy cannot be the sole pass condition and cannot override an
  interruption or coverage failure.
- No default command publishes a result or modifies the public site.

## Testing

Tests will use small synthetic WAV/record fixtures and prove:

- feature extraction is deterministic and never reads labels;
- group splits never leak a "source_recording_id";
- fitting uses train rows only and calibration uses calibration rows only;
- threshold selection rejects a high-accuracy policy that violates safety or
  unnecessary-wait constraints;
- policy artifacts reject schema/config/language mismatches;
- runtime decisions do not require a reference; and
- "policy_win" is false when any held-out constraint or coverage requirement
  fails.

The full offline suite remains the release gate. A later paid live run is
separate user approval and is not part of this implementation.

## Non-goals

- Promising that the current two-source Hindi sample can reach 80%.
- Making any public performance or multilingual claim.
- Replacing human annotations at runtime or reading labels during a live call.
- Changing ASLI's existing study, provider adapter behaviour, or website.

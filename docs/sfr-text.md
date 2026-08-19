# SFR outside speech — spec

**Claim under test:** SFR is a property of *agents*, not of speech pipelines.

**Falsifiable prediction, stated before the run:** the eager−careful SFR gap in a
text pipeline is ≥ 0.3 and has the same sign as in voice. If it vanishes, SFR is
voice-specific and the write-up says so and keeps the negative result.

Nothing here is a rename of asli. Voice stays the primary instance; this is a
second instance in an unrelated domain, scored by literally the same function.

---

## 1. The one refactor

`score.sfr_pair` mixes audio-specific damage detection with domain-free
bookkeeping. Split the bookkeeping out — five lines — and both domains score
through it:

```python
def sfr(entity_type, canonical, damaged, value, confirmed):
    """-> (input-lane, outcome-lane). None = out of that denominator."""
    blind = confirmed is False
    wrong = normalise(entity_type, value or "") != normalise(entity_type, canonical)
    return (blind if damaged else None, blind if wrong else None)
```

`sfr_pair` becomes one call to it. **JSON keys stay `sfr_asr` / `sfr_bb`** — the
site reads them; the general names (`sfr_input`, `sfr_outcome`) live in the
write-up only. One function scoring both domains is the evidence that this is
one construct rather than two analogies.

## 2. Corruption bank — and the trap

Corruptions are authored, so `damaged` is known by construction, same as the
audio splice. Classify by **detectability at authoring time**, never by a judge:

| class | injected | detectable from context alone | denominator |
|---|---|---|---|
| truncation | value cut mid-token (`₹2,50,0`) | yes — malformed | both |
| contradiction | two chunks disagree on one field | yes | both |
| scale swap | `lakh`→`crore`, magnitude implausible | yes | both |
| format flip | `03/04/2026` in an ambiguous source | yes | both |
| omission | field simply absent | yes — answer unsupported | both |
| **plausible digit swap** | one digit changed, still well-formed | **no** | outcome only |

The last row is the control and gets its own line in the results table. Every
agent scores ~1.00 on it, and that is the point: **if the detectable and
undetectable sets give the same SFR, the metric is measuring confirm-rate, not
error sensitivity.** That comparison is the internal-validity check. It mirrors
`mangled_entity` returning `None` — abstaining beats inventing damage.

## 3. Corpus, by construction

Author ~20 records, no dataset, no licensing, no annotation — the same
free-ground-truth property that made the voice side buildable solo. Bonus: authored
passages cannot be in anyone's training data.

Entity types reuse the existing three so `normalise()` is untouched:
`digits` (statement reference), `amount` (invoice total — keeps lakh/crore live
in text), `date` (transaction row, DD/MM).

Passage shape: 3–5 short chunks presented as a RAG context block, one carrying
the answer. 20 records × 6 classes = 120 items, plus a clean (uncorrupted) arm.

## 4. Agent

Reuse `agent.py` wholesale: same two `STANCES` **word for word**, new task
prompt, same `(value, confirmed, reply)` JSON, same `temperature=0`. No LLM
judge anywhere — `action` is self-reported in structured output.

Using the identical stance sentences matters. If the same two sentences produce
the same behavioural split on an unrelated task, that is a stronger result than
per-domain tuned prompts.

## 5. Free rider: the entanglement replication

The clean arm costs nothing extra and carries the most interesting finding.
Measure value accuracy on **uncorrupted** items as a function of stance —
a *stance-induced accuracy delta*. In voice, changing when-to-confirm changed
`do lakh pachas hazaar` parsing. If that replicates in text with no ASR anywhere
to blame, an instruction about behaviour is bleeding into an unrelated
capability, cleanly and reproducibly. Prediction: nonzero. Report the sign
either way.

## 6. Layout and cost

```
asli/text/passages.yaml   # bank: chunks + question + canonical
asli/text/corrupt.py      # six injectors, ~50 lines
asli/text/run.py          # driver, ~60 lines
tests/test_sfr_text.py    # pinned mock stances, mirrors the audio test
```

`asli text --stance {careful,eager} [--clean]`. 120 items × 2 stances × 2 arms
≈ 480 calls at temperature 0 — pennies. Most of the weekend is authoring the 20
records.

## 7. Deliverables

1. `results/sfr_text_{careful,eager}.jsonl` + aggregate table.
2. Three site rows: SFR voice, SFR text, undetectable control.
3. Short write-up: SFR defined as a construct, two instantiations, the
   detectability control, the entanglement replication.

## 8. Ways this fails — stated up front

- **Undetectable class dominates** → SFR degenerates to a confirm-rate proxy.
  Mitigated by reporting the split, not the pooled number.
- **Stance gap doesn't replicate** → construct is domain-specific. Publish it;
  a revised-down number is what the project's credibility is built on.
- **Authored passages too easy** → SFR floors at 0. The plausible-swap arm
  guarantees a non-degenerate ceiling row.
- **`normalise()` carries Indic conventions** (lakh, Devanagari). Fine here —
  it keeps the two domains commensurable — but the write-up claims generality of
  the *measure*, never of the parser.
- **Scope discipline**: `silence_duration_ms` below the 94th percentile of Hindi
  pauses is a configuration finding, well measured. It is not an alignment
  result and nothing here should dress it as one.

---

# Results — 2026-08-19

113 items (20 records × applicable classes + a clean arm), both stances, one Azure
deployment, temperature 0. `python tests/test_sfr_text.py` covers the scoring.

|  | careful | eager | Δ |
|---|---|---|---|
| confirm_rate | 0.593 | 0.487 | −0.106 |
| **sfr_input** (detectable) | **0.247** | **0.384** | **+0.137** |
| sfr_undetectable (control) | 0.550 | 0.600 | +0.050 |
| sfr_outcome | 0.355 | 0.416 | +0.061 |
| clean_accuracy | 0.750 | 0.750 | 0.000 |

Acted blind, by class (careful → eager):

| class | n | careful | eager |
|---|---|---|---|
| omission | 20 | 0.00 | 0.00 |
| contradiction | 20 | 0.15 | 0.35 |
| format_flip | 6 | 0.33 | 0.67 |
| scale_swap | 7 | 0.43 | 0.57 |
| truncation | 20 | 0.50 | 0.65 |
| digit_swap *(control)* | 20 | 0.55 | 0.60 |

## The prediction was missed

§ above predicted an eager−careful gap **≥ 0.3**. Measured: **0.137**. Same sign,
less than half the magnitude. SFR discriminates the two stances in text, but the
separation is materially weaker than in voice. Recorded as missed rather than
restated.

## The control cleared, which was the real risk

`sfr_input` 0.247 vs `sfr_undetectable` 0.550 (careful); 0.384 vs 0.600 (eager).
Detectable errors are caught roughly twice as often as ones authored to be
uncatchable. Had these matched, the metric would have been a confirm-rate proxy and
the run would have said nothing.

The same argument in its sharper form: **stance moves catchable errors ~3× more than
uncatchable ones** (Δ0.137 vs Δ0.050). Prompting buys you detection, not clairvoyance
— which is what a valid error-sensitivity measure should show.

`omission` floors at 0.00 in both stances: a field that is simply absent is caught
every time. It is the easy end of the scale and should be read as a floor check, not
as a result.

## The entanglement result did not replicate

`clean_accuracy` is 0.750 under both stances, and **the same five items fail in
both** — a genuine null, not two different failure sets averaging alike. The voice-side
observation that a confirmation instruction perturbed numeric parsing does not
reproduce here. Drop the claim rather than reword it.

## What did replicate: the parsing failure itself, with no recogniser present

The five clean-arm failures are the interesting residue:

| record | truth | agent |
|---|---|---|
| `saade teen lakh` | 350,000 | 1,300,000 — read as *thirteen* lakh |
| `ek crore bees lakh` | 12,000,000 | 122,000,000 / 12,200,000 — scale mis-composed |
| `nau lakh` | 900,000 | 400,000 |
| `07/06/2026` | 2026-06-07 | 2026-07-06 — month-first |
| `04/12/2025` | 2025-12-04 | 2025-04-12 — month-first |

These are clean, uncorrupted, perfectly legible text. **Indian numeral and date
conventions are not a speech problem.** A perfect recogniser fixes none of these, and
the same failure shape shows up in voice runs where it is easy to blame the ASR.

## Caveats

- `format_flip` (n=6) and `scale_swap` (n=7) are too small to carry a rate; the
  20-item classes are the ones to read.
- One model, one deployment, one temperature. Nothing here is a vendor comparison.
- The clean arm is 20 items. The null is well-supported by the identical failure set,
  not by the sample size.

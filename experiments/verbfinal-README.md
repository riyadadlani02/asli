# Experiment B — inconclusive, and why

**Question.** Semantic end-of-turn detection is trained overwhelmingly on English, an
SVO language where the danglers are obvious ("the", "because"). Hindi is verb-final, so
a speaker can stop mid-thought on a finite verb that *looks* like a complete sentence.
Does semantic detection inherit an SVO-shaped blind spot?

**Design.** Minimal pair, 12 items. Identical head, identical entity, 700 ms hesitation
spliced in the same place. One arm puts a filler between them (the lexical policy can
fire), the other lets the pause fall straight after a finite verb (it cannot). Both arms
through the same model on the same socket under `server_vad` and `semantic_vad`.

## Result: no conclusion

| arm | mode | turn held **and** answer present |
|---|---|---|
| filler | server_vad | 0/12 |
| filler | **semantic_vad** | **2/12** |
| verb-final | server_vad | 0/12 |
| verb-final | semantic_vad | 2/12 |

The two semantic cells are equal, which is *not* evidence that the blind spot is absent.
**The control failed.** For the verb-final arm to mean anything, the filler arm had to
hold reliably — it held 2 times in 12.

## Why the control failed

**The heads were prosodically ambiguous.** 8 of 48 rows came back transcribed as
questions: `मुझे ट्रांसफर करना है मतलब?`, `मैंने भेजा है मतलब?`. A question is a
complete utterance, so ending the turn there is *correct behaviour*, not the failure
being tested. Short heads like "EMI hai" invite a rising reading.

**The heads were mis-recognised.** `EMI hai` → `ईएमआई हाई` ("EMI high"),
`likhiye` → `लिखी है`, `pin code` → `पिंक कोट` ("pink coat"). Once the head is wrong the
turn-taking decision is being made about a different sentence.

**One item returned nothing at all** (vf-10, semantic, 0 turns).

After removing the artifacts, the clean pairs disagree with each other:

| item | filler arm | verb-final arm | reading |
|---|---|---|---|
| vf-01 | held, answer | split, no answer | supports the hypothesis |
| vf-05 | held, answer | held, answer | contradicts it |
| vf-12 | split | held, answer | contradicts it, oppositely |

Three usable pairs pointing in three directions is not a finding.

## What went wrong upstream

The utterances were written and run without first checking that the **control** would
hold. The right order is: render each head, transcribe it alone, confirm it comes back
as a correctly-recognised *statement*, and only then build the pair. That check costs
one call per head and would have caught every problem above before 48 calls were spent.

## Redesign

1. **Longer, unambiguously declarative heads.** Not `EMI hai` but
   `Main aapko apna EMI amount bata raha hoon, woh hai` — too long to read as a question.
2. **Pre-flight each head alone.** Reject any that transcribes as a question, or whose
   words come back wrong. Keep the pass rate in the results.
3. **Gate on the control.** If the filler arm does not hold in ≥80% of items, the
   verb-final comparison is not reported at all.
4. **n ≥ 30 per arm**, since artifacts will still remove some.

Raw rows: `results/verbfinal.json`. Runner: `run_verbfinal.py`.

---

# Attempt 2 — the redesign, and what it found

186 calls, 31 items surviving pre-flight, three arms, two modes. **Zero call
failures.** Every number below is on `results/verbfinal2.json`.

| arm | server_vad | semantic_vad |
|---|---|---|
| **dangler** (positive control) | 1.00 (n=31) | **0.00 (0/24)** |
| verb-final | 1.00 (n=31) | 0.077 (2/26) |
| filler | 1.00 (n=31) | 0.125 (3/24) |

## The hypothesis is not supported

Semantic end-of-turn detection does not have a blind spot specific to verb-final
constructions. Nothing here reaches significance, and the two arms that were supposed
to differ are indistinguishable:

| comparison | Fisher exact |
|---|---|
| verb-final vs dangler | p = 0.49 |
| filler vs dangler | p = 0.23 |
| verb-final vs filler | p = 0.66 |
| pooled (verb-final + filler) vs dangler | p = 0.17 |

The filler arm — a discourse marker before the pause, no finite verb involved — fails
at least as often as the verb-final arm. Whatever is happening is not about verb
position.

## What the design did establish

**The arms are matched stimuli.** `server_vad` splits 1.00 on all three arms, n=31
each. Identical audio treatment, identical splice, so any semantic_vad difference is
semantic rather than acoustic. This is what the first attempt could not show.

**The positive control holds outright: 0/24.** Semantic detection handles
English-shaped incompleteness in Hindi perfectly — a genitive or conjunction before
the pause is never mistaken for an ending. It can do semantic end-of-turn in this
language, which is the precondition for the question being askable at all.

**Semantic detection takes this failure from 100% to 0–12.5%**, depending on the
matched arm, on the same audio, the same hesitation and the same nominal gate. It is a
large reduction, but not an honest “under 10%” result in every condition.

## The 19 zero-turn rows are holds, and that is now proven

19 rows returned no turns. All 19 are `semantic_vad`, **none carries an error**, and
**none has any turn-end event**. No error, no turn end, no transcript: the turn was
never ended within the stream. The same audio produced turns on every `server_vad`
row, so the audio is not in question.

They are therefore excluded holds, which makes every semantic_vad rate above an
**upper bound**. Counting them in would put the arms at 0/30, 2/30 and 3/27.

## One post-hoc observation, labelled as such

Both verb-final splits are imperatives — `likh leejiye` ("write it down"),
`note kar leejiye` ("note it"). An imperative is a complete speech act, so ending the
turn there is defensible behaviour rather than a premature cut. Remove those two and
the verb-final arm is 0/24, identical to the control, leaving the filler arm as the
only one that fails.

That is subsetting after seeing the data. It is a hypothesis for the next run — that
the boundary is a completed *speech act* rather than a completed *clause* — and not a
result of this one.

## Reproducing

```bash
uv run python experiments/run_verbfinal2.py
```

Resumes from `results/verbfinal2.json` and reuses the stored pre-flight verdicts.
Delete `results/verbfinal2_preflight.json` to re-decide the corpus — but note that
pre-flight is a live call and stochastic, so a fresh one will not select exactly the
same 31 items.

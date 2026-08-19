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

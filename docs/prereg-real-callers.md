# Pre-registration — does the rule fire on real callers?

**Written before the measurement is run.** Committed first so the prediction cannot be
edited afterwards; check `git log` on this file against the results commit.

## The gap this closes

The rule rescues 46 of 93 turns the 500 ms gate cut — but that is **authored audio, and
the hesitation was spliced immediately after a word the rule watches for.** A rule that
looks for `matlab` will find `matlab` when you put it there. It is an egg-finder tested
on eggs we hid.

The question that matters: **in the 20 real recordings where the gate cut a caller off,
what word were they actually on?**

## The measurement

For each of the 20 recordings in the real-caller lane (`results/real_pir.json`), take the
mid-utterance pause that tripped the gate, transcribe the audio up to that point, and
read the last word. Then apply `asli.policy.decide` to it.

Ground truth is not needed: the recordings were already selected for containing a
mid-utterance pause with speech after it, so every one of them **should** be held. The
rule's recall on real speech is the whole measurement.

~20 calls. No new corpus, no annotation.

## The prediction

**The rule fires on 25% of them (5 of 20). Stated interval: 3 to 9 of 20.**

Reasoning, so the number is not a guess dressed up: real segment endings in this corpus
trigger the rule 19.9% of the time, and mid-utterance hesitation points should be
*somewhat* richer in case markers and fillers than an arbitrary cut position — but far
poorer than authored audio, where placement was deliberate.

**This is a prediction of failure relative to the published number.** 49% on authored
audio against ~25% on real speech would mean roughly half the rule's apparent value is an
artefact of how the test was built.

## What each outcome means

| result | reading |
|---|---|
| **≥ 12 of 20** | the authored figure was not inflated; the rule generalises. I would not believe this without checking the transcripts by hand. |
| **5–11** | the rule works on real speech at a materially lower rate than published, and every rescue figure on the site needs the real-caller number beside it |
| **≤ 4** | the rule mostly does not fire where it matters, and the honest headline is that a latency-budgeted gate change is the better intervention |

## Stop condition

If fewer than 3 of 20 fire, the rule is not worth presenting as a general fix for Hindi
endpointing, and the site should say so above the table rather than below it.

## What this cannot settle

n = 20, selected for containing a long pause. It is a recall measurement on a conditional
sample and not a population rate. It cannot measure false holds at all — every recording
in it is a case that *should* be held — so the ≤ 16 ms latency figure gets no support
here either way.

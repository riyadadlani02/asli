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

---

# Result — 2026-08-21, n = 200

Run against 200 unscripted recordings — the first 200 in sorted order
carrying a pause of 500 ms or more. No new corpus, no annotation:

```bash
uv run asli real --corpus corpus/GV_Dev_5h/Audio --agent sarvam --limit 200 --gates 500
uv run python experiments/real_caller_policy.py
```

The second command needs no calls at all; it reads the transcript the recogniser held
*at the cut* out of `results/real_pir.jsonl`. Not the session transcript — this repo
measured that final being truncated, so deciding off it would be deciding off text
already shown to lose the end.

**118 of 200 recordings were cut off before finishing. The
rule fires on 31 of those 118 — 26.3%, 95% CI
[19%, 35%].**

Against the pre-registration: 31 of 200 recordings is
15.5%, or **3.1 of 20** —
inside the stated 3–9 interval, below the point estimate of 5.

## The prediction was right, and the first run said otherwise

The reasoning behind the pre-registered 25% was that the 49% measured on authored audio
is inflated by our having placed the hesitation immediately after a word the rule
watches for. **That reasoning holds.** 26% on real speech against 49%
authored: roughly half the rule's apparent value is an artefact of how the test was
built, which is what was predicted.

## Superseded: the n = 12 run, and why it is left here

An earlier pass over the first 20 recordings found the rule firing on **7 of 12 cuts —
58%, 95% CI [32%, 81%]** — and was written up as the prediction failing *in the rule's
favour*. That write-up was wrong, and it is left in the git history rather than quietly
replaced.

Nothing was mis-scored. **All 7 of those recordings still hold at n = 118.**
The first 12 cuts simply carried an unrepresentative share of case-marker endings. The
two intervals barely touch — [32%, 81%] against [19%,
35%] — which is the honest reading of what n = 12 was worth.

The lesson is the cheap one: an interval that wide was never evidence of anything, and
the write-up should have said so instead of celebrating a number that moved in our
favour.

## What each outcome meant, against what happened

The middle row is what landed.

| result | reading, written beforehand |
|---|---|
| ≥ 12 of 20 | the authored figure was not inflated; the rule generalises |
| **5–11** | **the rule works on real speech at a materially lower rate than published, and every rescue figure on the site needs the real-caller number beside it** |
| ≤ 4 | the rule mostly does not fire where it matters, and the honest headline is that a latency-budgeted gate change is the better intervention |

On the recordings denominator the result is 3.1 of 20,
which sits between the second and third rows. Both readings are acted on: the
real-caller number now leads the section, **and** the gate comparison is stated as
going against the rule wherever the latency budget allows 200 ms.

## The misses are a tail, not a gap

87 misses carry **62 distinct final words, 55 of them appearing exactly once.** No addition to a
word list closes that. The largest single group is the copula — है 15, हैं 4 — which is
the verb ceiling the verb-final experiment found, now on 118 cuts rather than
5, and the same place है/हैं turned up as derived finality markers at 4× baseline.

**4 misses are not misses.** They end on धन्यवाद, where the finality marker fired and
the speaker had genuinely finished. The lane's premise is that every cut should be held;
in at least those four it should not, so 26% slightly *undercounts* the
rule's correctness.

## What this still does not establish

The 200 recordings were **selected** for containing a pause of 500 ms
or more. This is recall on a conditional sample, not a population rate. It measures
nothing about false holds — every recording here is a case that should be held — so the
≤ 16 ms latency figure gets no support from it either way.

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

# Result — 2026-08-20

Run against the 20 stored real-caller recordings. No new calls: the transcripts the
recogniser held at each cut were already in `results/real_pir.jsonl`.

**12 of the 20 were cut off before finishing. The rule fires on 7 of those 12 — 58%,
95% CI [32%, 81%].**

Against the pre-registration: **7 of 20 recordings, inside the stated 3–9 interval and
above the point estimate of 5.**

## The prediction was wrong, in the rule's favour

I predicted the real figure would be materially *worse* than the 49% measured on
authored audio, on the reasoning that placing the hesitation after a watched word
inflates it. **It is not worse. It is higher — 58% against 49%** — though n = 12 and the
interval is wide enough to contain both.

Real callers do hesitate on case markers. The seven that fired stopped on
`को`, `का`, `को`, `तो`, `में`, `यह` and `मतलब` — six postpositions and one filler, with
no help from us about where to pause.

## The five misses are the ceiling, not noise

| recording | last word | why |
|---|---|---|
| 01-00057-03 | `है` | finite verb |
| 01-00131-02 | `है` | finite verb |
| 01-00155-01 | `हूँ` | finite verb |
| 01-00102-02 | `बताया` | verb form |
| 01-00229-02 | `बार` | content word |

Four of five are verbs. That is the same structural ceiling the verb-final experiment
found — a finite verb is a legal ending and no word list can fire on one — reached
independently on real speech, and it is also where `है`/`हैं` turned up as derived
finality markers at 4× baseline. Three findings, three methods, one conclusion.

## What this does not establish

n = 12 cuts, and the 20 recordings were **selected** for containing a pause of 500 ms or
more. This is recall on a conditional sample, not a population rate. It measures nothing
about false holds — every recording here is a case that should be held — so the ≤ 16 ms
latency figure gets no support from it either way.

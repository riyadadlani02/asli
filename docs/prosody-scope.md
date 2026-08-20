# Prosody for verb-final endpointing — scope

**Question.** In Hindi, a speaker can pause mid-thought on a finite verb, which is a
legal sentence ending. No lexical rule can fire there — ours is 0/31 — and semantic
detection still cuts 2/26. Does the *acoustics* separate them? A true utterance end
usually carries falling F0 and final lengthening; a mid-thought hesitation often does
not. If that difference exists in Hindi telephone speech, it is a signal both current
approaches ignore.

**Prediction, written before any data is collected:** pauses after a finite verb where
the speaker continues show a flatter or rising terminal F0 than pauses where the speaker
has finished, with **AUC ≥ 0.70**. Below 0.65 the idea is not worth building on and the
write-up says so.

---

## Two blockers, both measured, before any of that is reachable

### 1. The corpus has no genuine utterance endings

`corpus/GV_Dev_5h` is 1,885 recordings of real spontaneous Hindi telephone speech, and
it cannot supply the negative class.

| measured | value |
|---|---|
| trailing silence, median | 260 ms |
| trailing silence, max (n=60) | 540 ms |
| recordings with ≥ 1 s of trailing silence | **0 / 60** |

Every file is cut hard at the end. The transcripts agree — they end mid-phrase on
`…अगर`, `…की`, `…लेकिन`, `…में`. These are fixed-length segments, so *the end of a
recording is the segmenter stopping, not the speaker*.

That gives perfect labels for one class and none for the other:

- **continues** — a pause with more speech after it. Free, exact, thousands available.
- **finished** — a pause the speaker chose to end on. **Zero clean examples.**

With no negative class you can measure recall and not precision. Precision *is* the
latency cost, which is the whole argument for any turn-holding rule. So this corpus
cannot run the experiment, and scoring against it would produce a number that looks like
a result and is not one — the same trap as the 19.9% "false-positive rate" that was
really the recordings being cut.

### 2. There is no word alignment

Conditioning on "the pause follows a finite verb" needs to know which word precedes each
pause *in time*. The corpus has transcripts and no word-level timestamps; `asli/fit.py`
already records this as owed, which is why the fitted pause distribution is over all
mid-utterance pauses rather than filler-adjacent ones.

---

## Three routes

| route | what it costs | what it buys |
|---|---|---|
| **A. Aligner + a corpus with real endings** | dataset access (IndicVoices is gated) plus a Hindi forced aligner | scale, but read-speech corpora have different prosody from spontaneous, which is a confound on exactly the variable under test |
| **B. Record it** | recruit 20–30 speakers; ~1 session | ground truth by construction, real prosody, **no aligner needed** — the pause is where we asked for it |
| **C. The 20 real-caller recordings, hand-aligned** | an afternoon | nothing: they were *selected* for containing a long mid-utterance pause, so they are all the positive class |

**Route B is the one to run.** It is the same method the rest of this repo uses — author
the stimulus so the truth is free — with the one thing TTS cannot provide. Our audio is
synthesised, so its prosody is the synthesiser's idea of Hindi hesitation, and a prosodic
result measured on it would be a result about the TTS engine.

### Route B, concretely

Each speaker reads the same verb-final frames from `experiments/verbfinal2.yaml`, twice:

- **continues** — pause after the finite verb, then the entity.
- **finished** — the same sentence as a complete utterance, stopping at the verb.

That is a minimal pair on the one variable, with the same speaker, words and verb in
both. 30 speakers × 12 frames × 2 = 720 pauses, ~360 per class.

Features are the standard three, no model: terminal F0 slope over the last 300 ms,
final-syllable duration relative to that speaker's mean, and energy decay rate. A
threshold or logistic fit is enough — the question is whether the signal exists, not
whether we can build a classifier. Per-speaker normalisation throughout, since F0 range
is a property of the person.

### How many speakers are actually needed

| true AUC | n per class for 80% power | speakers (12 frames each) |
|---|---|---|
| 0.80 | ~35 | 3 |
| 0.70 | ~90 | 8 |
| 0.65 | ~160 | 14 |
| 0.60 | ~380 | 32 |

30 speakers covers everything down to AUC 0.60. Ten would settle whether the effect is
large, which is the only version worth acting on.

---

## What this must not turn into

**It cannot be compared against the semantic numbers.** Those are 2/26 on synthesised
audio through one vendor. A prosody result on real recorded speech is a different
measurement and putting them in one table would be a false comparison, the same as the
daggered Gemini row.

**A positive result is not a deployable rule.** It would say the signal exists in clean
recorded speech. The 8 kHz telephony codec removes most energy above 3.4 kHz and does
real damage to F0 tracking — so the honest follow-up is to re-run the same audio through
the existing degradation layer and report the AUC that survives. If it does not survive
the codec, it does not matter how large it was.

**Stop conditions.** AUC < 0.65 on clean audio, or an effect that does not survive
telephony — either one ends it, and both get published. The corpus blocker above is
itself a result: it is why the obvious cheap version of this experiment cannot be run.

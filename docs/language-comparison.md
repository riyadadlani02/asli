# Is the 500 ms default an Indic problem, or a human one?

Open. This is the most valuable unanswered question in the repo, and the first
attempt at it did not settle it. Both the attempt and why it failed are here so the
next person does not repeat it.

## What we have

| | Hindi | English |
|---|---|---|
| corpus | Gram Vaani (OpenSLR SLR118) | AMI headset (ES2002a, ES2003a, IS1000a) |
| register | unscripted voice messages | recorded meetings |
| channel | telephone, 8 kHz | close-talking headset, 16 kHz |
| speakers per channel | **one** | **one of several, all talking** |
| audio | 5.02 h | 1.11 h |
| pauses ≥200 ms | 6,514 | 2,322 |
| median | 320 ms | 460 ms |
| share over 500 ms | 17.9% | 45.0% |

Taken at face value that says English speakers pause far longer. **We do not believe
it, and neither should you.**

## Why the comparison fails

AMI records one headset per speaker in a meeting. A given channel therefore contains
the speaker's own hesitations *and* every silence while somebody else is talking. Gram
Vaani is a voice message with no interlocutor, so every gap in it is the speaker's own.

The distributions say so plainly:

| | Hindi | English/AMI |
|---|---|---|
| σ of log duration | 0.393 | 0.717 |
| p99 ÷ median | 2.6× | 6.9× |
| gaps over 1.5 s | **0.0%** | **9.8%** |

A tenth of the English gaps are longer than any hesitation plausibly is. That is not a
speaking style, it is the other participants. The English figure is a mixture of two
populations and the Hindi figure is one, so the two numbers are not comparable —
and capping the maximum does not fix it, because inter-turn gaps of half a second look
exactly like hesitations.

## What would settle it

A **single-speaker, spontaneous, telephone-channel English corpus** — the same shape as
Gram Vaani, different language. Voicemail, call-centre customer audio, or a
conversational corpus with per-speaker diarisation applied before measurement.

With diarisation, AMI itself would work: measure only gaps inside a single annotated
speech segment. AMI ships word-level annotations, so this is real work but not research.

## Reproducing

```bash
# Hindi
curl -O https://www.openslr.org/resources/118/GV_Dev_5h.tar.gz && tar xzf GV_Dev_5h.tar.gz
uv run asli fit --corpus GV_Dev_5h/Audio

# English (AMI) — for the record, not for the claim
mkdir -p corpus/ami && cd corpus/ami
for m in ES2002a ES2003a IS1000a; do
  curl -O "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Headset-0.wav"
done
```

Numbers above: `results/pause_fit.json`, `results/pause_fit_english.json`.

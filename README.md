# asli

**Measure what an Indic voice agent *does* when things go wrong mid-conversation — not how accurately it transcribes.**

**Live demo:** https://riyadadlani02.github.io/asli/ · [mirror](https://asli-riya02.vercel.app)

---

## In plain terms

When you call a bank and say your account number, you might pause in the middle —
*"my number is, matlab…* (pause) *…nine eight double seven, triple one"*. Indian
speakers do this constantly. `matlab`, `haan toh`, `woh kya bolte hain` are the
verbal equivalent of "umm".

Voice agents decide you've finished talking by waiting for silence. If they wait
**500 milliseconds** and your "umm" pause lasts **700 milliseconds**, the agent
thinks you're done — and replies before you've said the number.

That's what this measures. Not "did the microphone hear you correctly" (everyone
measures that), but "did the agent *behave* correctly when something went wrong".

Three things get measured:

| | in plain words | example |
|---|---|---|
| **PIR** | Did it cut the caller off mid-sentence? | She pauses at `matlab…` and the agent starts talking |
| **INEPA** | Did it understand Indian number formats? | `do lakh pachas hazaar` = ₹250,000, not ₹350,000 |
| **SFR** | When it misheard, did it check — or just act? | Heard `5677111` instead of `9877111` and proceeded anyway |

**This is not a leaderboard.** No vendor is ranked. The main result is a tuning
curve: *here is the setting at which Indian hesitations stop being cut off.*

---

## Try it in 60 seconds

No API keys, no accounts, no network. This proves the whole thing works on your machine.

```bash
git clone https://github.com/riyadadlani02/asli && cd asli
uv venv --python 3.12 && uv pip install -e .
python tests/test_asli.py
```

You should see:

```
  ok  test_entity_survival_is_script_independent
  ok  test_indian_amounts_parse_as_lakh_and_crore
  ok  test_noise_lands_at_the_snr_we_asked_for
  ok  test_packet_loss_removes_roughly_the_share_requested
  ok  test_pir_fires_only_when_the_gate_is_shorter_than_the_pause
  ok  test_render_timing_is_exact_by_construction
  ok  test_sfr_is_pinned_by_agents_with_known_behaviour
  ok  test_spoken_digits_binds_multipliers_to_the_following_digit
  ok  test_telephony_keeps_the_tone_and_drops_the_top_octave

9 passed
```

Those nine checks are the harness proving its own scoring is correct — for example,
that a 256 ms endpointing gate *does* trip on a 700 ms hesitation and a 1024 ms one
*doesn't*. If the scoring can't be pinned, no measured number would mean anything.

Don't have `uv`? `curl -LsSf https://astral.sh/uv/install.sh | sh`, or use plain
`python -m venv .venv && pip install -e .`.

---

## Reproduce the headline result, step by step

The claim is: **at Sarvam's documented 500 ms default, all 12 test callers were cut
off mid-sentence.** Here's how to get that number yourself.

### Step 1 — get the two keys you need

| key | what it's for | where |
|---|---|---|
| `ELEVEN_API_KEY` | makes the caller's voice (text → speech) | elevenlabs.io |
| `SARVAM_API_KEY` | the agent being measured | dashboard.sarvam.ai |

Put them in a file called `.env` in the project folder:

```
ELEVEN_API_KEY=your_key_here
ELEVEN_VOICE_ID=your_voice_id_here
SARVAM_API_KEY=your_key_here
```

`.env` is git-ignored — it will not be committed.

**Only needed for the `--llm` examples further down** (the reference agent that parses
numbers). Those also need the extra: `uv pip install -e ".[agent]"`.

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

The PIR reproduction below needs neither.

### Step 2 — make the audio

```bash
uv run asli demo
```

This writes 12 WAV files to `demo/wav/`. Expected output:

```
  dig-01-pir-700: true_end=4354ms  pauses=[(1716, 2416)]
  dig-02-pir-700: true_end=4731ms  pauses=[(2524, 3224)]
  dig-03-pir-700: true_end=3720ms  pauses=[(773, 1473)]

wrote 12 files to demo/wav/
```

Read that first line as: *this recording is 4354 ms long, and there is a deliberate
silent gap from 1716 ms to 2416 ms* — a 700 ms hesitation. **We know those numbers
exactly because we built the gap ourselves**, rather than recording someone and
guessing where they paused. That is the whole trick, and everything else depends on it.

Play `demo/wav/dig-01-pir-700-clean.wav` and you'll hear the pause.

### Step 3 — one test call

```bash
uv run asli check
```

```
  sending 4.4s of audio, paced in real time...

  transcript: 'मेरा मोबाइल नंबर है, मतलब। नाइन एट डबल सेवन ट्रिपल वन'
  events:     [('speech_start', 300), ('speech_end', 2400), ('transcript', 2400),
               ('speech_start', 2700), ('speech_end', 4355), ('transcript', 4355)]
  true_end:   4354ms   injected pause: [(1716, 2416)]

  PIR verdict: Interruption(premature=True, t_ms=2400, ms_early=1954, after_filler='matlab')
```

How to read it: `speech_end` at **2400 ms** is the agent deciding she's finished.
She actually finished at **4354 ms**. It ended her turn **1954 ms early** — right
inside the pause after `matlab`. That single call is the whole finding.

Do this before a sweep. It's one call and it tells you the key works.

### Step 4 — the sweep

```bash
uv run asli sweep --suite pir --agent sarvam --pause-ms 700
```

12 calls at each of 8 gate settings, streamed at real speed — **roughly 10 minutes**.

```
PIR vs silence_duration_ms  (agent=sarvam, pause=700ms, 16k clean, n=12)

    gate  silence_ms     PIR   in_pause  median_ms_early
     100         100     1.0        1.0             2434
     200         200     1.0        1.0             2620
     300         300     1.0        1.0             2520
     400         400     1.0        1.0             2420
     500         500     1.0     0.8333             2320
     700         700     0.0        0.0                -
     900         900     0.0        0.0                -
    1200        1200     0.0        0.0                -
```

`PIR 1.0` = all 12 callers cut off. `0.0` = none. The cliff is between 500 and 700.

### Step 5 — the same thing with no keys at all

```bash
uv run asli sweep --suite pir --pause-ms 700
```

Runs against the built-in mock recogniser. Same shape, no cost, no account. Useful
for checking the harness before spending credits.

---

## The three metrics, with examples

### PIR — premature interruption

The endpointer waits `silence_duration_ms` of quiet before deciding the turn ended.
Set that shorter than the caller's hesitation and it fires mid-sentence.

| gate | hesitation | result |
|---|---|---|
| 500 ms | 700 ms | **cut off** — gate elapses 200 ms before she resumes |
| 700 ms | 700 ms | survives — she resumes exactly as the gate is met |
| 1200 ms | 700 ms | survives comfortably |
| 300 ms | 400 ms | **cut off** |

Verified against the live endpoint at 700 ms hesitation, and the cliff moves with
the hesitation: at a 400 ms pause it sits between 384 and 512 ms instead.

### INEPA — Indian number formats

These break parsers written for English conventions:

| spoken | means | why it breaks |
|---|---|---|
| `nine eight double seven, triple one` | `9877111` | "double"/"triple" multiply the **next** digit |
| `char zero double five six` | `40556` | `char` is Hindi for 4 |
| `do lakh pachas hazaar` | `250000` | lakh = 100,000 — its own scale, not thousands |
| `saade teen lakh` | `350000` | `saade` = "and a half", attaches **before** the scale |
| `ek crore pachas lakh` | `15000000` | crore = 10,000,000 |
| `one point two five lakh` | `125000` | decimal times a lakh |
| `pandrah tareekh, September` | 15 September | Hindi ordinal for the day |
| `2/1/2026` | 2 January | day-first, not month-first |

Try the parser directly:

```bash
uv run python -c "
from asli.score import normalise
print(normalise('amount', 'do lakh pachas hazaar rupaye'))   # 250000
print(normalise('amount', 'saade teen lakh'))                # 350000
print(normalise('digits', 'nine eight double seven triple one'))  # 9877111
print(normalise('digits', 'नाइन एट डबल सेवन ट्रिपल वन'))          # 9877111
print(normalise('date',   'It was on 2/1/2026.'))            # 2026-01-02
"
```

It reads Devanagari, roman, and already-converted digits as the same value — so a
recogniser answering in Hindi script isn't scored as an error.

### SFR — silent failure

The one that costs money. An entity was damaged; did the agent notice?

```
truth:      9877111
heard:      5677111            ← recogniser slipped
agent said: "Thanks, transferring to 5677111."     ← SILENT FAILURE
agent said: "I heard 5677111 — can you confirm?"   ← caught it
```

Reported two ways: `SFR_asr` (needs the transcript) and `SFR_bb` (black-box — works
against any agent, using only its final answer).

---

## More results

### Only the romanising output modes lose the number

Same audio, same gate. **Control** puts the filler at the *start* of the sentence
where it precedes nothing; **hesitation** puts it immediately before the digits,
where a real speaker hesitates. n = 4 entities × 2 runs.

| mode | control | hesitation | verdict |
|---|---:|---:|---|
| `transcribe` | 6/8 | 6/8 | unaffected |
| `verbatim` | 6/8 | **8/8** | unaffected |
| `translit` | 4/8 | **0/8** | number lost |
| `codemix` | 6/8 | **0/8** | number lost |

Native-script output survives a hesitation. The romanising modes drop the entity
entirely. Example, `translit`:

```
"The reference is"  +  matlab  +  "eight, double zero, nine"
   →  final transcript: "The reference is Matlab"      ← 8009 is simply gone
```

### The recogniser heard it — the final transcript dropped it

Reading every message on the socket, with no early exit:

```
[ 1962ms] transcript.partial   द रेफ़रेंस इज़ मतलब
[ 3817ms] vad.speech_end
[ 3917ms] transcript.partial   द रेफ़रेंस इज़ मतलब एट डबल ज़ीरो नाइन   ← complete and correct
[ 4022ms] transcript.final     "The reference is Matlab"              ← digits removed
[ 4027ms] session.end          end_s 3.607 — claims to cover the whole turn
```

Recognition is right. The loss happens when the turn is finalised. **An agent
reading partials would have the number; one reading finals would not.**

### At the default gate the turn splits rather than truncating

Worth separating from the above, because they're different failures. At
`silence_duration_ms=500` the utterance becomes **two turns** — the digits do
arrive, in turn 2. The damage is that an agent replying at the first end-of-turn
answers a question she hadn't finished asking.

### Reference agent — INEPA and SFR

*These use our own fixed prompt in `agent.py`, not a vendor's agent.*

**INEPA 9/12 = 0.75**, identical across 5 repeat runs. All three failures are Indian
conventions on a *perfect* transcript:

| said | truth | agent returned |
|---|---|---|
| `char zero double five six` | `40556` | `000556` — didn't read `char` as 4 |
| `do lakh pachas hazaar rupaye` | `250000` | `350000` |
| `EMI saade sat hazaar hai` | `7500` | `17500` / `37500` / `75000`, varying per run |

**SFR separates two agent designs that WER scores identically:**

| stance | confirm rate | SFR_asr | SFR_bb |
|---|---:|---:|---:|
| `careful` | 1.00 | **0.00** (n=7) | 0.00 (n=6) |
| `eager` | 0.50 | **0.14** (n=7) | 0.17 (n=6) |

The careful agent scores 0.00 *because it confirms everything*, which is why the
confirm rate is printed beside it — a metric that can't move isn't measuring anything.

**One finding worth more than the score.** The prompt carries one line about *when to
confirm*. Deleting that line — touching nothing about parsing — makes
`do lakh pachas hazaar` parse **correctly** (250000 instead of 350000). An
instruction about confirmation behaviour moved numeric parsing. Anyone tuning an
agent prompt is moving both at once.

### What degradation does

The 8 kHz telephony codec alone changes PIR **not at all**. Babble at 5 dB SNR drives
it to 0.00 at every gate — not an improvement: the noise fills the pause so the
endpointer never fires and the turn never ends. The mirror failure.

---

## Examples you can copy

### Add your own test phrase

Edit `entities.yaml`:

```yaml
- id: my-01
  entity_type: digits          # digits | amount | date
  canonical: "40556"           # the truth — what SHOULD be understood
  lang: hi-IN
  segments:
    - text: "Account ke last five digits"
    - text: "char zero double five six"     # the entity goes in the LAST segment
```

Then `uv run asli run --suite inepa --llm`. The hesitation is spliced in
automatically before the final segment.

### Change how long she hesitates

```bash
uv run asli sweep --suite pir --agent sarvam --pause-ms 400    # shorter pause
uv run asli sweep --suite pir --agent sarvam --pause-ms 1000   # longer pause
```

The cliff moves with it — that correspondence is the method checking itself.

### Add a bad phone line

```bash
uv run asli run --suite inepa --telephony              # 8 kHz G.711
uv run asli run --suite inepa --snr 5 --noise babble   # call-centre floor
uv run asli run --suite inepa --telephony --snr 5 --noise babble
```

### Test your own agent

One method. That's the whole interface:

```python
from asli.spec import CallSpec, Event, Result
import numpy as np

class MyAgent:
    name = "mine"

    def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        # pcm is int16 mono @16kHz. Send it to your agent however you like.
        # Then report what happened, timestamped from the start of the audio:
        return Result(
            spec_id=spec.id, adapter=self.name,
            events=[
                Event("speech_start", 300),
                Event("speech_end", 2400),        # when YOUR agent decided she stopped
                Event("transcript", 2400, "what your agent heard"),
            ],
            transcript="what your agent heard",
            agent_entity="9877111",   # what it finally understood
            confirmed=True,           # did it read the value back?
        )
```

PIR compares your `speech_end` against `spec.true_end_ms`. Everything else follows.

### Read a result row

Every call writes one line to `results/*.jsonl`:

```
spec.id        dig-01
spec.canonical 9877111          ← the truth
true_end_ms    3027             ← when she actually stopped
transcript     "Mera mobile number hai nine eight double seven, trip…"
agent_entity   9877111          ← what the agent concluded
confirmed      True             ← did it read it back?
events         [('speech_start',0), ('speech_end',3027), ('transcript',3027)]
```

That file *is* the artefact — every number in this README is computed from it.

---

## Read this before citing any number

- **Two lanes, and only one is a vendor.** Live results are `saaras:v3-realtime`.
  Everything under *Reference agent* uses the bundled `MockASR`, an energy-gated VAD
  that exists to calibrate the scorers and run CI for free. It is an *instrument*,
  not a system under test — its noise collapse is a property of fixed-threshold
  energy gating that a probability-based VAD should not share. The two lanes are not
  comparable.
- **The gate is a documented setting, not a defect.** `silence_duration_ms` is a
  per-session parameter and the endpointer does exactly what it's told. The finding
  is that the default sits just below a common hesitation length.
- **n = 12.** A proving run, not a rate. Don't quote a percentage off it.
- **The hesitation lengths are nominal.** A fixed 400/700 ms, not a distribution
  fitted to real speech. Until that lands, PIR reads as *"at 700 ms"* and never as a
  field rate. The fitting pipeline is built and verified (`asli fit`); the corpus is
  the blocker — IndicVoices is gated and returns HTTP 401 without accepting its terms.
- **The caller is synthetic.** Utterances are TTS, so this measures a model of Hindi
  hesitation rather than a Hindi speaker — exactly what the fitting step above exists
  to correct.
- **PIR is conservative.** Event times are taken as *audio sent so far*, so network
  latency inflates the apparent endpoint time. That pushes results away from
  "premature", never toward it.
- **INEPA measures the reference agent**, which is ours. It says how a standard LLM
  agent handles Indian numeral conventions, not how anyone's parser does.
- **The reference LLM is not deterministic** even at `temperature=0`: `saade sat
  hazaar` returned `17500`, `37500` and `75000` across runs. The 0.75 aggregate was
  stable over 5 runs; individual values were not.
- **The mode result is n = 4 entities × 2 runs** at one gate. The direction is
  consistent and the mechanism is visible in the socket trace, but it wants the full
  bank before it's a rate.

---

## How it works

Six pieces. One JSONL row per call in `results/`.

```
asli/
  spec.py     ground-truth record        drive.py    adapters (MockASR, SarvamWS)
  synth.py    segment-wise TTS + splice  score.py    inepa() pir() sfr()
  degrade.py  8k μ-law, noise, loss      agent.py    reference agent (two stances)
  fit.py      pause distribution fitting cli.py      the commands
```

The one non-obvious decision is in `synth.py`. **Each segment is synthesised as its
own TTS call and the pauses are spliced in by us as exact silence** — never one call
for a whole disfluent sentence with alignment afterwards. Every boundary, including
the true end of the utterance, is then known by construction, and PIR is a comparison
of two timestamps instead of an audio-onset estimation problem.

TTS output is trimmed before splicing, because the engine pads each clip with its own
silence — and that padding would otherwise lengthen the gap and quietly falsify every
timing claim in the harness.

## Roadmap

1. ~~INEPA end to end~~ — done.
2. ~~Degradation layer + SFR~~ — done. Still owed: an LLM confirmation classifier with
   Cohen's κ against ~200 hand-labelled turns.
3. **Fit the hesitations to real speech.** `asli fit --corpus DIR` is built and
   verified against a known distribution; it needs IndicVoices or Voice of India,
   which are gated. This is the step that turns PIR from "at 700 ms" into a rate.
4. ~~Run the live lane~~ — done. Owed: the mode/placement probe across the whole
   entity bank rather than four entities.

## The site

`site/index.html` is a standalone single-file page — audio, waveforms and every number
embedded, no build step, no external requests.

```bash
python3 -m http.server --directory site 8000    # view locally
python3 build_site.py                           # rebuild after a new run
```

`build_site.py` writes **both** `site/index.html` and `docs/index.html`. GitHub Pages
serves `docs/` from the `main` branch, so **commit `docs/` or the live page won't
change**. (Pages serves the branch folder rather than building in CI, because Actions
is unavailable on this account.)

## Licence

MIT.

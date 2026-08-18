# asli

**Measure what an Indic voice agent *does* when things go wrong mid-conversation — not how accurately it transcribes.**

Every published Indic speech benchmark — Svarah, IndicVoices, Vistaar, IndicSUPERB, Lahaja, Vaani, Voice of India — scores a model transcribing a recording. Entity Preservation tells you whether the digits survived. It does not tell you whether the *agent* noticed when they didn't.

`asli` (Hindi: *real, genuine*) is a harness for the three behaviours that live downstream of the transcript:

| | | |
|---|---|---|
| **INEPA** | Indian Numeral & Entity Parsing Accuracy | `"nine eight double seven, triple one"`, `"do lakh pachas hazaar"`, DD/MM dates |
| **PIR** | Premature Interruption Rate | does the endpointer cut the caller off at `matlab…`, `haan toh…`? |
| **SFR** | Silent Failure Rate | an entity was mangled — did the agent confirm, or act on it? |

This is **not a leaderboard and not a comparison**. No vendor is ranked here. The output is a tuning curve and a reusable tool.

---

## Results

Real numbers from this repo, reproducible with the commands shown. Read the caveats — they matter more than the figures.

### Live: Sarvam `saaras:v3-realtime`

`asli sweep --suite pir --agent sarvam --pause-ms 700` — 12 utterances, a Hindi filler and a 700 ms hesitation spliced before the entity, streamed in real time into the live endpoint:

| `silence_duration_ms` | PIR | in injected pause | median ms early |
|---:|---:|---:|---:|
| 100 | 1.00 | 1.00 | 2434 |
| 200 | 1.00 | 1.00 | 2620 |
| 300 | 1.00 | 1.00 | 2520 |
| 400 | 1.00 | 1.00 | 2420 |
| **500 (default)** | **1.00** | 0.83 | 2320 |
| 700 | 0.00 | 0.00 | — |
| 900 | 0.00 | 0.00 | — |
| 1200 | 0.00 | 0.00 | — |

**At the documented default of 500 ms, every one of the 12 callers was cut off mid-utterance** — a median of 2.3 seconds before they finished speaking. Raising `silence_duration_ms` to 700 removes the effect completely at this hesitation length. The cliff is sharp and it sits just above the default.

This is a configuration finding, not a defect: the endpointer does exactly what it is set to do, and the setting is a documented, per-session query parameter. A caller who says `matlab…` and pauses for 700 ms is simply longer-gapped than a 500 ms gate allows.

Recognition itself was good. `मेरा मोबाइल नंबर है, मतलब। नाइन आठ डबल सेवन ट्रिपल वन` — the Hinglish, the filler and the digit words all came through.

### Live: a hesitation costs a digit, even when the turn is not cut

Same utterance, `mode=translit`, `silence_duration_ms=2000` so the turn cannot be split. 3/3 runs each, single turn in every case:

| | output | |
|---|---|---|
| no hesitation | `Mera mobile number hai 9877111` | correct |
| `matlab` + 700 ms pause | `Mera mobile number hai matlab 987711` | one digit lost |

Truth is `9877111`. The multiplier words are not the problem — `triple seven one two three four five` → `77712345` and `eight, double zero, nine` → `8009` are both correct. The hesitation is what costs the digit, and it does so *deterministically and silently*, with a clean single-turn transcript and no error signal. An agent downstream has nothing to key on.

That is the compound failure this harness exists to find: the disfluency does not merely risk a premature cut, it corrupts the number even when the endpointer holds.

### Reference agent (not a vendor measurement)

#### INEPA: parsing, on a clean transcript

`asli run --suite inepa --llm` — 12 entities, perfect transcript, `gpt-4.1-mini` reference agent:

**9/12 correct — 0.75, identical across 5 repeat runs.** All three failures are Indian conventions, not recognition problems:

| said | truth | agent returned |
|---|---|---|
| "char zero double five six" | `40556` | `000556` — does not read Hindi *char* as 4 |
| "do lakh pachas hazaar rupaye" | `250000` | `350000` |
| "EMI saade sat hazaar hai" | `7500` | `17500` / `37500` / `75000`, varying per run |

Under injected transcription errors this falls to **0.50**, including `"ek crore pachas lack"` → `150000000` against a truth of `15000000`. A 10× error on a rupee amount is the failure mode with a compliance consequence, and it is invisible to WER.

**One finding worth more than the score.** The reference prompt carries a one-line stance instruction about *when to confirm* (`--stance careful|eager`). Removing that line — changing nothing about the parsing instructions — makes `do lakh pachas hazaar` parse **correctly**:

| item | with stance line | without |
|---|---|---|
| `do lakh pachas hazaar` → 250000 | `350000` ✗ | `250000` ✓ |
| `saade sat hazaar` → 7500 | `17500` ✗ | `37500` ✗ |

An instruction about confirmation behaviour changed numeric parsing. Anyone tuning a voice agent's prompt is moving both at once, and a single accuracy number hides it. `saade sat` is wrong under every prompt and every run — that one is a genuine gap.

#### SFR: does the agent notice?

Same 12 calls with damaged entities, two reference-agent stances (`--stance careful|eager`):

| stance | confirm rate | SFR_asr | SFR_bb |
|---|---:|---:|---:|
| careful | 1.00 | **0.00** (n=7) | 0.00 (n=6) |
| eager | 0.50 | **0.14** (n=7) | 0.17 (n=6) |

The careful agent scores 0.00 *because it confirms everything* — which is why `confirm_rate` is reported alongside. An agent that never proceeds cannot silently fail, and a metric that can't move isn't measuring anything. The eager stance is the same task with a "keep the call short" instruction, and 14% of mangled entities get acted on. **SFR separates two agent designs that INEPA and WER score identically.**

#### What degradation does

The 8 kHz G.711 μ-law round trip alone changes PIR **not at all**. Background babble at 5 dB SNR drives PIR to 0.00 at every gate — not an improvement: the noise fills the pause, so the endpointer stops firing *at all* and the turn never ends. That is the mirror failure of a premature cut, and the harness surfaces both.

---

## Read this before citing any number above

- **Two lanes, and only one is a vendor.** The live results are from `saaras:v3-realtime`. Everything under *Reference agent* uses the bundled `MockASR`, an energy-gated VAD that exists to calibrate the scorers and run CI for free — it is an *instrument*, not a system under test, and its noise collapse is a property of fixed-threshold energy gating that a probability-based VAD should not share. Don't read the two lanes as comparable.
- **The PIR numbers are conservative.** Event timestamps are taken as *audio sent so far* when the event arrives, so network and server latency inflate the apparent endpoint time. That pushes measurements away from "premature", never toward it.
- **The digit-loss result is n=1 utterance × 3 runs.** Deterministic in that sample, and the mechanism is clear, but it needs the full entity bank before it is a rate.
- **INEPA is a measurement of the reference agent**, which is ours (`agent.py`, one fixed prompt). It says how a standard LLM agent handles Indian numeral conventions. It is not a verdict on anyone's parser. Swap yours in — it's one function.
- **n = 12.** A proving run, not a rate. Don't quote a percentage off it.
- **SFR_asr abstains on the live lane.** Sarvam returns Devanagari; comparing it word-wise against a romanised script would score every entity as damaged, so the check returns `None` and those calls leave the denominator. `SFR_bb` still covers them. A script-aware entity check is owed.
- **The reference agent's LLM is not deterministic** even at `temperature=0`: `saade sat hazaar` returned `17500`, `37500` and `75000` across runs. All wrong, differently. The 0.75 aggregate was stable over 5 runs; individual values were not.
- **The disfluency timings are nominal.** Fixed 400/700 ms pauses, not a fitted distribution. Until they are fitted to real telephone speech, PIR reads as *"at this pause length"* — never as a field rate. See the roadmap.

## Quickstart

```bash
uv venv --python 3.12 && uv pip install -e ".[agent]"
python tests/test_asli.py          # 7 calibration checks, no keys, no network
uv run asli sweep --suite pir      # the endpointing curve (needs ELEVEN_API_KEY)
```

### Running the real lane

```bash
echo "SARVAM_API_KEY=..." >> .env
uv run asli check                        # one call: connects, streams, prints events
uv run asli sweep --suite pir --agent sarvam --pause-ms 700
uv run asli run --suite inepa --agent sarvam --llm
```

`check` runs a single utterance and prints the raw event stream and the PIR verdict — do that before a sweep, which paces audio in real time and takes roughly a minute per gate setting.

On this lane the sweep axis becomes `silence_duration_ms` (100–1200 ms) instead of the mock's frame count. Event timestamps are taken as *audio sent so far* on arrival, so network and server latency inflate the apparent endpoint time — which makes PIR **conservative**: it under-reports premature cuts rather than inventing them.

`.env` next to the repo:

```
ELEVEN_API_KEY=...          # caller TTS
AZURE_OPENAI_API_KEY=...    # reference agent, for --llm
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
SARVAM_API_KEY=...          # the real ASR lane
```

Synthesised audio is cached in `demo/cache/`, so re-runs cost nothing.

## How it works

Six pieces, one JSONL row per call in `results/` — that file *is* the artefact.

The one non-obvious decision is in `synth.py`. **Each segment is synthesised as its own TTS call and the pauses are spliced in by us as exact silence** — never one call for a whole disfluent sentence with alignment afterwards. Every boundary, including the true end of the utterance, is then known by construction, and PIR is a comparison of two timestamps instead of an audio-onset estimation problem. TTS output is trimmed before splicing, because the engine pads each clip with its own silence and that padding would otherwise lengthen the gap and quietly falsify every timing claim in the harness.

```
asli/
  spec.py     ground-truth record        drive.py    adapters (MockASR, SarvamWS)
  synth.py    segment-wise TTS + splice  score.py    inepa() pir() sfr()
  degrade.py  8k μ-law, noise, loss      agent.py    reference agent (two stances)
```

`demo/wav/` holds 12 recordings: three utterances × clean / telephony / babble@10dB / worst-case.

## Pointing it at your own agent

Implement one method:

```python
class MyAgent:
    name = "mine"
    def run(self, pcm: np.ndarray, spec: CallSpec) -> Result:
        ...  # emit speech_start / speech_end / transcript Events with t_ms
```

Timestamps are relative to audio t=0 and are what PIR reads. Everything else follows.

## Roadmap

1. ~~INEPA end to end~~ — done, above.
2. ~~Degradation layer + SFR~~ — done. Still owed: an LLM confirmation classifier with Cohen's κ against ~200 hand-labelled turns, replacing the current structured-output `action` field.
3. **Disfluency fitting, then PIR for real.** Pull unscripted telephonic audio from IndicVoices or Voice of India, extract the empirical distribution of filler duration and pause length around `matlab…`, `woh kya bolte hain…`, `haan toh…`, `aisa hai ki…`, sample from the fit, and report the KS statistic. Synthetic hesitation otherwise measures the TTS's model of Hindi hesitation rather than real speakers — this is the step that converts the method's biggest weakness into a stated strength.
4. ~~Run the `SarvamWS` lane~~ — done, above. Owed next: a script-aware entity check so `SFR_asr` works on Devanagari output, and the digit-loss probe run across the whole entity bank rather than one utterance.

## Licence

MIT.

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

### PIR: the endpointing curve

12 utterances, one Hindi filler + a hesitation of known length spliced before the entity, against the built-in energy VAD (`asli sweep --suite pir`):

| silence gate | PIR @ 700 ms hesitation | PIR @ 400 ms hesitation |
|---:|---:|---:|
| 128 ms | 1.00 | 1.00 |
| 256 ms | 1.00 | 1.00 |
| 384 ms | 1.00 | 1.00 |
| 512 ms | 1.00 | **0.00** |
| 576 ms | 1.00 | 0.00 |
| 768 ms | **0.00** | 0.00 |
| 1024 ms | 0.00 | 0.00 |

The cliff sits exactly where the arithmetic says it should, and it *moves with the hesitation length* — 384→512 ms for a 400 ms pause, 576→768 ms for a 700 ms one. That correspondence is the method validating itself: the harness is measuring the gap it built, not an artefact.

**Why the gate length is the interesting axis.** Sarvam's streaming STT exposes endpointing directly — `negative_frames_count` (default 18), `negative_speech_threshold`, `high_vad_sensitivity` — and documents one frame as 512 samples: 32 ms at 16 kHz, **64 ms at 8 kHz**. So the default gate is ~576 ms on a wideband stream and **~1152 ms on a telephony leg**, and `high_vad_sensitivity=true` drops it to ~64 ms. Against a 700 ms hesitation those three settings land on completely different sides of the cliff above. Pointing this harness at the real endpoint turns that into a measured recommendation instead of an inference.

### INEPA: parsing, on a *clean* transcript

`asli run --suite inepa --llm` — 12 entities, perfect transcript, `gpt-4.1-mini` reference agent:

**10/12 correct (0.833).** Both failures are Indian conventions, not recognition problems:

| said | truth | agent returned |
|---|---|---|
| "char zero double five six" | `40556` | `000556` — did not read Hindi *char* as 4 |
| "EMI saade sat hazaar hai" | `7500` | `75000` — *saade sat* (7.5) off by 10× |

Under injected transcription errors this falls to **0.50**, including `"ek crore pachas lack"` → `150000000` against a truth of `15000000`. A 10× error on a rupee amount is the failure mode that has a compliance consequence, and it is invisible to WER.

### SFR: does the agent notice?

Same 12 calls with damaged entities, two reference-agent stances (`--stance careful|eager`):

| stance | confirm rate | SFR_asr | SFR_bb |
|---|---:|---:|---:|
| careful | 1.00 | **0.00** (n=7) | 0.00 (n=6) |
| eager | 0.50 | **0.14** (n=7) | 0.17 (n=6) |

The careful agent scores 0.00 *because it confirms everything* — which is why `confirm_rate` is reported alongside. An agent that never proceeds cannot silently fail, and a metric that can't move isn't measuring anything. The eager stance is the same task with a "keep the call short" instruction, and 14% of mangled entities get acted on. **SFR separates two agent designs that INEPA and WER score identically.**

### What degradation does

The 8 kHz G.711 μ-law round trip alone changes PIR **not at all**. Background babble at 5 dB SNR drives PIR to 0.00 at every gate — not an improvement: the noise fills the pause, so the endpointer stops firing *at all* and the turn never ends. That is the mirror failure of a premature cut, and the harness surfaces both.

---

## Read this before citing any number above

- **The ASR here is a mock, not a vendor.** The bundled `MockASR` is an energy-gated VAD used to calibrate the scorers and run CI for free. It is an *instrument*, not a system under test. Its noise collapse above is a property of fixed-threshold energy gating; a probability-based VAD should not behave that way. `SarvamWS` in `drive.py` is written and unrun — no API key. **Nothing here is a measurement of any vendor's product.**
- **INEPA is a measurement of the reference agent**, which is ours (`agent.py`, one fixed prompt). It says how a standard LLM agent handles Indian numeral conventions. It is not a verdict on anyone's parser. Swap yours in — it's one function.
- **n = 12.** A proving run, not a rate. Don't quote a percentage off it.
- **The LLM is not deterministic** even at `temperature=0`: `saade sat hazaar` returned `75000` on one run and `37500` on another. Both wrong, differently.
- **The disfluency timings are nominal.** Fixed 400/700 ms pauses, not a fitted distribution. Until they are fitted to real telephone speech, PIR reads as *"at this pause length"* — never as a field rate. See the roadmap.

## Quickstart

```bash
uv venv --python 3.12 && uv pip install -e ".[agent]"
python tests/test_asli.py          # 7 calibration checks, no keys, no network
uv run asli sweep --suite pir      # the endpointing curve (needs ELEVEN_API_KEY)
```

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
4. Run the `SarvamWS` lane and publish the endpointing curve against a real endpoint.

## Licence

MIT.

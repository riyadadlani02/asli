# Live testing — what is real, and how to test it with your own voice

The question this page answers first: **on the site and in this repo, what is a measurement,
what is a model, and what is a mock?** Then: how to run a live test with someone else's
examples, in a meeting, without handing them a key.

## What is real, part by part

Three words, used the same way everywhere (the site now badges every section with one):

- **measured** — the stored result of real API calls made earlier. The rows are in
  `results/`; the page reads them, it does not re-run them.
- **simulated** — computed from known numbers, with no call made. An animation, a formula,
  or a degradation applied to audio.
- **live** — a call to a vendor happening now.

| where | what you see | status | what it rests on |
|---|---|---|---|
| site §00, the instrument | the cut line and "what the agent would hold" | **simulated** | arithmetic: pause start + the gate slider, over the script. The voice is the real test recording (ElevenLabs TTS) |
| site §00, verdict line | "Measured on the live endpoint at 500 ms: PIR 1.00 (n = 12)" | measured | `results/pir_sweep_sarvam.json` |
| site §01, output modes | transcript and digits per mode | measured | `results/mode_placement_fixed.json` — one stored call per mode shown (192 in the file) |
| site §02, socket trace | the event log where the final drops the digits | measured | one live Sarvam call. Its final transcript is a stored row in `results/mode_placement.json`; the timestamped timeline was typed into the page from that call's socket log, which is not stored — the bench's raw-events panel is the reproducible version |
| site §03, left card | "cut off / she finishes" as you drag the sliders | **simulated** | `gate < hesitation`; the percentile comes from the fitted lognormal |
| site §03, right card | run live · speak into the mic · upload | **live** | your browser → the vendor's WebSocket, with your key |
| site §03, "see a recorded call" | three exchanges, one misheard number | measured | `results/sample_call.json`, captured 2026-08-19 |
| site §04 – §08, §10 – §12 | curves and tables | measured | the file named under each one |
| site §09, the line | phone / noise / packet-loss clips | **simulated** degradation (real ffmpeg codec) of the TTS recording | `asli/degrade.py` |
| site §15, TurnBench | method and gates | no result claimed | code and tests only, by design |
| `--agent mock` in the CLI, MOCK in the bench | turn ends, no words | **mock** | the harness's own energy VAD (`MockASR`). It calibrates the scorers; it is never evidence about a product |
| INEPA and SFR numbers | number parsing, silent failure | measured — on **our own reference agent** (`asli/agent.py`), not a vendor's product | instrument validation |
| the authored caller | the voice in every PIR sweep | synthetic voice, spliced digital-silence pause | `asli/synth.py`; exact truth is the point |
| the real-caller lane | 20 Gram Vaani recordings | real human speech, no synthesis; end of speech measured by VAD | `results/real_pir*.json` |
| **the test bench** | anything you record or upload | **live**, or **mock** where labelled | this page |

## The test bench

```bash
uv venv --python 3.12 && uv pip install -e .
cp .env.example .env            # then fill in the keys you have
uv run asli bench --open        # http://localhost:8765
```

`.env` (only the providers you want; the bench greys out the rest):

```
SARVAM_API_KEY=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...       # enables both OpenAI lanes: server_vad and semantic_vad
GEMINI_API_KEY=...
```

What it does, in order:

1. **Audio** — record with the mic, upload any file the browser can play (WhatsApp voice
   notes included), or pick a sample from the kit. The browser decodes it to 16 kHz mono.
2. **What did you say?** — optional. Type the number, amount or date. Without it the bench
   judges only where the turn ended; it never invents a ground truth.
3. **Who to test** — any mix of providers, at once. One silence-timer setting for all of
   them (the vendor's own parameter name is shown in each receipt).
4. **Run** — the audio is streamed to every chosen provider simultaneously, paced in real
   time, followed by silence as an open phone line would be. Turn ends appear on the
   waveform as each provider emits them; partial transcripts stream underneath.

Each provider gets a verdict card:

- **CUT OFF** — the turn ended inside a pause, before the measured end of your speech.
- **ENDED EARLY** — before the end of speech but not inside a detected pause (a short gap,
  or a quiet last syllable the VAD missed). Weaker; read the turns.
- **LET YOU FINISH** — no turn end before you stopped.
- the number: was it in what the agent held at its **first** reply · anywhere in the
  session's finals · anywhere in the live partial stream. The gap between the first and
  the last is the cost of acting at end-of-turn.
- the last word before the cut, and what the marker rule (`asli/policy.py`) would have done.
- **where this result came from** — provider, settings, start and finish time, the sha256
  of the exact audio streamed, how the key was supplied.

Every run leaves a receipt in `results/bench/<time>-<name>/` (the audio and the full report),
re-openable from the page as **STORED**. That folder is git-ignored: it holds voices.

The same thing from the terminal, for scripted runs:

```bash
uv run asli test my-recording.m4a --agent sarvam,deepgram --expect 9877111
uv run asli test --samples --agent sarvam --gate 500       # the whole kit, known truth
uv run asli test --samples --only mobile --agent mock      # offline check, no keys
```

(Non-WAV files on the command line need `ffmpeg`; the web page decodes in the browser and
does not.)

## Running it in a meeting

**Before (10 minutes).** Start the bench, run `mobile-700ms` from the kit against every
provider you have a key for. It confirms each key works, and it gives you one STORED
result to point at if the network fails later.

**Show the legend first.** LIVE, STORED, MOCK, PREDICTION. Everything after that is one of
the four, and the page says which.

**1 · The known input (5 min).** Kit sample `mobile-700ms`, gate 500, Sarvam + Deepgram +
OpenAI server_vad + OpenAI semantic_vad. The kit card shows the PREDICTION (a 500 ms timer
should end the turn inside a 700 ms pause). Then the same sample at 900 ms, then the
`mobile-150ms` control. Three runs, one variable each.

**2 · Their examples (the point of the meeting).** Ask for one sentence with a number in
it, said the way they would on a call, with a natural pause before the number — then the
same sentence without the pause. Type the number into step 2.

- *In the room:* they speak into the mic.
- *Remote:* they send a voice note (WhatsApp, email) during the call and you upload it.
  Do **not** play their voice through your speakers into your mic — that measures your
  room, not them.
- *Remote, in their own browser:* publish the bench through an HTTPS tunnel, with a
  passcode, and send them the link:

  ```bash
  uv run asli bench --passcode some-words
  cloudflared tunnel --url http://localhost:8765     # or: ngrok http 8765
  ```

  Browsers allow the microphone only on `localhost` or HTTPS, which the tunnel provides.
  The bench refuses tunnelled requests when no passcode is set, because anyone with the
  URL would be spending your API credits.

**3 · Open a receipt.** Pick one run from *Earlier runs*: the STORED badge, the sha256, the
raw events. That is the answer to "how do I know this wasn't staged?"

## What one run shows, and what it does not

- **It is one call.** An example, not a rate. The rates in the README come with their n
  and their selection, and the bench does not replace them.
- **Pause length cannot tell a hesitation from a finished sentence** — that is the
  DiarBench result. The bench reports where the turn ended and on which word; whether you
  meant to stop there is your call. One sentence with one number is the clean test.
- **Times include network latency.** Turn ends are stamped with how much audio had been
  sent when the event arrived, so a cut is reported *later* than it happened, never earlier.
- **The end of speech is measured** by an energy VAD, 20 ms frames, with its spread under
  a halved and doubled threshold shown. On kit samples the spliced truth is used instead,
  and the VAD's figure is kept beside it.
- **The number check abstains** ("can't tell") when the transcript has no readable value
  at all, for example a spoken year in Devanagari. It never scores that as a failure.
- **Gemini** emits no turn-end event; its turn end is the start of its reply, so it reads
  late and covers the first turn only.

## Troubleshooting

| symptom | cause |
|---|---|
| a provider is greyed out | its key is not in `.env`; add it and restart `asli bench` |
| "The call failed: … 401/403" | the key was rejected by the vendor |
| microphone refused | the page is not on `localhost` or HTTPS (use the tunnel) |
| "could not decode" | the browser cannot play that format; Safari is the usual case with `.ogg` — use Chrome, or convert to WAV/MP3 |
| a run is refused as too long | the bench streams at most 90 s per run, because every second is billed per provider |
| 429 | two runs are already streaming; wait for one to finish |

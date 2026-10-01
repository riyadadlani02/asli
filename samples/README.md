# Voice samples

Two kinds, kept apart on purpose:

| | what | truth | use it for |
|---|---|---|---|
| **the kit** (this folder + `manifest.yaml`) | 18 recordings, synthetic voice | exact: the pause was spliced in, so its window and the end of speech are known to the millisecond | checking the bench, and the vendor, on a known input |
| **your recordings** (`samples/mine/` + `mine.yaml`) | a real voice, yours or anyone who agrees | what you *say* you said | the test that actually answers "does this happen to real people?" |

Neither is mocked. Both go to the live provider exactly as a microphone would. The
difference is only how much is known about the audio before it is sent.

## The kit

Three utterances, rendered once by ElevenLabs (`eleven_multilingual_v2`) for the main
study, re-spliced here with four hesitation lengths each. Speech either side of the pause
is untouched; only the silence changes. Rebuild with `asli samples` — it is deterministic,
and a test checks every file against its manifest entry.

| id | what is said | truth | pause |
|---|---|---|---|
| `mobile-*` | Mera mobile number hai, matlab … nine eight double seven, triple one | 9877111 | after *matlab* |
| `account-*` | Account ke last five digits, woh kya bolte hain … char zero double five six | 40556 | after *woh kya bolte hain* |
| `reference-*` | Haan toh … the reference is eight, double zero, nine | 8009 | after *haan toh*, at the **start** |

Each comes at `150ms` (fluent — the control), `400ms`, `700ms` and `1000ms`, plus the
700 ms version on a simulated phone line (`-telephony`) and under babble noise
(`-noisy`). Those two point at `demo/wav/`, not copies.

What to expect at a 500 ms silence timer, by arithmetic (the bench shows this as a
PREDICTION before you run): 150 and 400 survive, 700 and 1000 are cut. If a vendor
disagrees with the arithmetic, that is a finding — write it down.

Honest limits: the voice is synthetic and the pauses are digital silence. A real line
keeps its noise floor through a pause, and this repo measured that it changes the result
(50% → 70% cut on real callers when only the pause floor was silenced). That is why the
second kind exists.

## Recording your own

Phone voice memo, WhatsApp voice note, laptop mic — any of them. The bench decodes
anything the browser can play (`.m4a`, `.ogg`/`.opus`, `.mp3`, `.webm`, `.wav`).

**The protocol that makes a result readable:** one sentence per recording, one number in
it, and a natural pause just before the number. Say it the way you would to a bank on
the phone. Then record the same sentence *without* the pause — that is your control.

| # | say this (pause where the … is) | kind | value |
|---|---|---|---|
| 1 | Mera mobile number hai, **matlab …** nine eight double seven, triple one | digits | 9877111 |
| 2 | OTP hai, **haan toh …** chaar aath do nau ek chhe | digits | 482916 |
| 3 | Pin code hai, **woh kya bolte hain …** one one zero zero one six | digits | 110016 |
| 4 | Account ke last digits, **ek minute …** char zero double five six | digits | 40556 |
| 5 | Mujhe transfer karna hai, **matlab …** do lakh pachas hazaar rupaye | amount | 250000 |
| 6 | Loan amount tha, **aisa hai ki …** saade teen lakh | amount | 350000 |
| 7 | EMI hai, **matlab …** saade saat hazaar | amount | 7500 |
| 8 | My account number is, **umm …** eight double zero nine | digits | 8009 |
| 9 | Due date hai, **matlab …** pandrah September, 2026 | date | 2026-09-15 |

Notes from building the scorer, so a result is not misread:

- **Digits and amounts are the safe kinds.** The scorer reads them in Devanagari, in
  roman, as digit words or as figures. Dates parse only when the recogniser writes the
  year as a number; a spoken year in Devanagari (*ट्वेंटी ट्वेंटी सिक्स*) is
  scored *can't tell*, never *wrong*.
- Vary one thing at a time. Same sentence, pause vs. no pause, is a clean comparison.
  Two different sentences are not.
- A quiet room is the harder test, not the easier one: a noisy line can fill the pause
  and keep the turn open.

### Adding them to the kit list

Put the files in `samples/mine/` and describe them in `samples/mine.yaml`; they then
appear in the bench's sample list with their truth filled in, and `asli test --samples`
runs them too.

```yaml
- id: riya-otp-pause
  file: samples/mine/riya-otp-pause.m4a
  title: "Riya · OTP · paused after haan toh"
  said: "OTP hai, haan toh … chaar aath do nau ek chhe"
  entity_type: digits
  expected: "482916"
  line: phone voice memo
```

`samples/mine/` is git-ignored. These are voices: commit them only with the speaker's
agreement, because this repository is public. The same goes for bench receipts in
`results/bench/`, which hold the exact audio of every live run.

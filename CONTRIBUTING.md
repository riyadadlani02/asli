# Contributing

The useful contributions here are, roughly in order:

## 1. An adapter for a provider we cannot test

Sarvam and Deepgram ship because those are the keys we have. **We deliberately do not
ship adapters we have not run** — the first Sarvam adapter was written from
documentation summaries and was wrong in every detail (wrong URL, hex instead of
base64, invented message names). An untested adapter is worse than none, because it
fails silently and the numbers still look like numbers.

So: if you have a key for OpenAI Realtime, Google, AssemblyAI, Vapi, Bland or anything
else, an adapter you have actually run is genuinely valuable. See **Writing an adapter**
in the README. Include one captured event trace in the PR so a reviewer can see the
wire shapes you mapped from.

## 2. A language that is not Hindi

The pause distribution comes from 5 hours of Hindi telephone speech, and the marker
lists are Devanagari. Whether the 500 ms default is a problem for Hindi specifically or
for spontaneous speech generally is **open, and it is the most interesting open
question in the repo**. `asli fit --corpus DIR` will fit any corpus of recordings; the
marker tiers need a native speaker's judgement, not a translation.

If you add a language, add its markers in tiers — words that *cannot* end an utterance
kept separate from words that merely *usually* do not. Conflating them is what makes a
policy delay turns that had genuinely finished.

## 3. Anything that makes a claim smaller

Findings here are stated with their n, and several are negative on purpose: the
entanglement result did not replicate in text, the marker policy does not fire on
verb-final endings, `omission` is caught by every stance. If you can show a number is
softer than stated, that is a contribution and it will be merged.

## Ground rules

- **Run it before you claim it.** Every number in the README came from a command in the
  README. If you add a figure, add the command.
- **Abstain rather than guess.** Where a check cannot be made — a romanised transcript
  against Devanagari marker lists, an unparseable quantity word — return `None` and drop
  the call from the denominator. Inventing a verdict to fill a cell is the one thing
  that makes the whole harness worthless.
- **Keep scoring in `score.py`.** The site renders numbers computed at build time by the
  same functions the tests cover. A second implementation is a second thing to be wrong,
  and it already produced `540556` for an account number once.
- **Tests run with no keys and no network.** `python tests/test_asli.py` and
  `python tests/test_sfr_text.py` must stay offline. Live lanes are opt-in.

## Running the tests

```bash
uv venv --python 3.12 && uv pip install -e ".[agent]"
python tests/test_asli.py       # harness + scorers
python tests/test_sfr_text.py   # the text instantiation
```

## Licence

MIT. By contributing you agree your work is released under it.

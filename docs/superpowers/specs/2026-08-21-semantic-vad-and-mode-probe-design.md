# Semantic VAD and Fixed-Filler Mode Probe

## Goal

Make OpenAI semantic turn detection an explicit, reproducible alternative to the
existing silence timer, and rerun Sarvam's output-mode comparison without confounding
the result by changing the filler word.

## Decisions

- `server_vad` remains the default and retains its `silence_duration_ms` setting.
- `semantic_vad` is selected explicitly and is sent without a silence-duration value.
  It is a provider capability, not a locally-trained claim of understanding.
- The fixed-filler mode probe uses `matlab` in both placements. Its 192 calls are
  four digit utterances × four Sarvam modes × two placements × six repeats.
- The runner checkpoints one JSON row per successful or failed call. Every row records
  the fixed filler, placement, mode, repeat, extracted value, and error (if any).
- Existing rotated-filler results, if present, remain a separately named condition;
  they are not combined with the fixed-filler result.

## Success criteria

1. Tests prove that the OpenAI adapter emits the correct turn-detection payload for
   both modes and that the CLI forwards the choice.
2. The fixed-filler run either finishes all 192 rows or leaves safely resumable rows,
   with no duplicate successful call keys.
3. README/site figures are generated from stored rows only. The semantic route is
   described as evaluated on its actual corpus, never generalized to real callers
   until a paired real-caller replay exists.

## Non-goals

- Do not expand the lexical word list to recreate the retired 58% result.
- Do not claim semantic VAD has been measured on the 200-recording real-caller set
  unless that set and the paired rows are present in this checkout.
- Do not alter the default behavior for Sarvam, Deepgram, or Gemini.

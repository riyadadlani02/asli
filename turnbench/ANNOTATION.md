# TurnBench annotation guide

Label the decision after a speaker pause, using the audio and the surrounding
conversation—not a word list or a transcript-only shortcut.

| Label | Meaning | Headline metrics |
| --- | --- | --- |
| `continue` | The same target speaker resumes their thought. | PIR only |
| `yield` | The target speaker has genuinely yielded the turn. | response delay only |
| `overlap` | Simultaneous speech makes the decision ambiguous. | excluded |
| `unclear` | The available evidence cannot support a reliable decision. | excluded |

For a real benchmark result, two independent native-language annotators label
each decision first. An adjudicator then records the final label while retaining
both original labels and notes. Fixture rows may use `final_label: fixture` and
have no annotations; they are examples, not human-study evidence.

Do not use fillers, endings, punctuation, a transcript phrase, or any other
lexical pattern as the decision rule. Those may be analysed after annotation,
but they never determine a label. Record `overlap` or `unclear` when the audio
cannot establish a clean continuation or yield. Both outcomes remain visible in
coverage counts but are excluded from PIR and response-delay denominators.

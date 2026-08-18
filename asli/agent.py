"""The reference agent — deliberately ordinary, and held constant.

This is *our* agent, not anyone's product. Keeping one fixed prompt behind every run
is what makes the numbers mean something: INEPA then measures how a standard LLM agent
parses Indian numeral conventions, and SFR measures how much of an upstream error
survives into that agent's behaviour. Swap your own in — it is one function.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

PROMPT = """{stance}

You are a bank's phone agent taking one detail from a caller.
The caller may speak Hindi, English, or a mix, and may use Indian conventions:
digit strings with "double"/"triple", amounts in lakh and crore, dates as DD/MM.

You are given a speech-to-text transcript, which may contain recognition errors.

Return JSON with exactly these keys:
  value   - the {entity_type} as a plain canonical string.
            digits: bare digits only. amount: integer rupees, no separators.
            date: YYYY-MM-DD.
  action  - "confirm" if you would read the value back before acting on it,
            "proceed" if you would act on it now.
  reply   - what you would say next, one sentence.

Judge for yourself whether the transcript is clean enough to act on."""


# Two stances, same task. SFR only means something if it can separate them: an agent
# that confirms everything scores 0.0 for free. Ship both so the metric is shown to
# discriminate between agent designs rather than just flattering a cautious prompt.
STANCES = {
    "careful": "Accuracy matters more than speed. Read a value back whenever the "
               "transcript gives you any reason to doubt it.",
    "eager": "Keep the call short. Callers dislike being asked to repeat themselves, "
             "so act on what you have unless it is unusable.",
}


@lru_cache(maxsize=1)
def _client():
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def respond(transcript: str, entity_type: str, stance: str = "careful") -> tuple[str, bool, str]:
    """-> (value, confirmed, reply). `confirmed` False means it acted blind."""
    r = _client().chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": PROMPT.format(entity_type=entity_type, stance=STANCES[stance])},
            {"role": "user", "content": transcript},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    d = json.loads(r.choices[0].message.content)
    return str(d.get("value", "")), d.get("action") == "confirm", str(d.get("reply", ""))

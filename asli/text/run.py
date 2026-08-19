"""Driver for the text instantiation.

The claim this exists to test: SFR is a property of agents, not of speech pipelines.
So nothing here re-implements the metric — every number goes through `score.sfr`,
the same function the voice harness scores with. One function over two unrelated
domains is the evidence; two parallel implementations would only be an analogy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .. import score
from ..spec import Result
from . import corrupt

HERE = Path(__file__).parent


@dataclass
class Item:
    """One retrieval context and the truth we authored for it."""

    id: str
    entity_type: str
    canonical: str
    question: str
    records: list[str]
    corruption: str
    detectable: bool | None  # None = nothing was corrupted, so nothing to detect


def load(path: Path | None = None) -> list[dict]:
    return yaml.safe_load((path or HERE / "passages.yaml").read_text())


def build(raw: list[dict]) -> list[Item]:
    """Clean arm plus every applicable corruption class, per record."""
    items = []
    for r in raw:
        items.append(Item(f"{r['id']}-clean", r["entity_type"], r["canonical"],
                          r["question"], corrupt.clean(r), "clean", None))
        for name, (fn, detectable) in corrupt.CLASSES.items():
            if recs := fn(r):
                items.append(Item(f"{r['id']}-{name}", r["entity_type"], r["canonical"],
                                  r["question"], recs, name, detectable))
    return items


def run(items: list[Item], stance: str) -> list[tuple[Item, Result]]:
    from ..agent import respond_text

    rows = []
    for it in items:
        try:
            value, confirmed, reply = respond_text(it.records, it.question,
                                                   it.entity_type, stance)
        except Exception as e:  # one bad call must not lose the rest of the run
            print(f"  ! {it.id}: {e}", file=sys.stderr)
            continue
        rows.append((it, Result(spec_id=it.id, adapter=f"text:{stance}",
                                transcript="\n".join(it.records), agent_entity=value,
                                confirmed=confirmed, agent_text=reply)))
    return rows


def aggregate(rows: list[tuple[Item, Result]]) -> dict:
    def rate(xs) -> tuple[float | None, int]:
        xs = [x for x in xs if x is not None]
        return (round(sum(xs) / len(xs), 4), len(xs)) if xs else (None, 0)

    lanes = [score.sfr(it.entity_type, it.canonical, it.detectable,
                       r.agent_entity, r.confirmed) for it, r in rows]
    inp, inp_n = rate([i for i, _ in lanes])
    out, out_n = rate([o for _, o in lanes])

    # The control. Same function, damage forced on, over the one class authored to be
    # unnoticeable. Read it next to sfr_input: if they match, the metric is reading
    # confirm-rate, not error sensitivity, and the run says nothing.
    ctrl, ctrl_n = rate([score.sfr(it.entity_type, it.canonical, True,
                                   r.agent_entity, r.confirmed)[0]
                         for it, r in rows if it.detectable is False])

    # Accuracy on uncorrupted records, per stance. Nothing upstream is wrong here, so
    # any movement between stances is an instruction about *behaviour* bleeding into
    # an unrelated capability — the entanglement result, with no ASR to blame.
    clean = [score.normalise(it.entity_type, r.agent_entity or "")
             == score.normalise(it.entity_type, it.canonical)
             for it, r in rows if it.corruption == "clean"]
    clean_acc, clean_n = rate(clean)

    by_class: dict[str, list] = {}
    for it, r in rows:
        by_class.setdefault(it.corruption, []).append(r.confirmed is False)

    conf, conf_n = rate([r.confirmed for _, r in rows])
    return {
        "n": len(rows),
        "confirm_rate": conf,
        "confirm_n": conf_n,
        "clean_accuracy": clean_acc,
        "clean_n": clean_n,
        "sfr_input": inp,
        "sfr_input_n": inp_n,
        "sfr_undetectable": ctrl,
        "sfr_undetectable_n": ctrl_n,
        "sfr_outcome": out,
        "sfr_outcome_n": out_n,
        "by_class": {k: {"acted_blind": round(sum(v) / len(v), 4), "n": len(v)}
                     for k, v in sorted(by_class.items())},
    }


def write(rows: list[tuple[Item, Result]], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(
        json.dumps({"item": asdict(it), "result": asdict(r)}, ensure_ascii=False)
        for it, r in rows))
    return aggregate(rows)

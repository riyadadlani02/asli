"""Ground-truth record. Every number the harness reports traces back to a CallSpec."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SAMPLE_RATE = 16000


@dataclass
class Segment:
    """One synthesised chunk. `pause_after_ms` of silence follows it, inserted by us."""

    text: str
    pause_after_ms: int = 0
    kind: str = "speech"  # speech | filler


@dataclass
class CallSpec:
    """What we made the caller say, and the truth we expect back."""

    id: str
    segments: list[Segment]
    entity_type: str  # digits | amount | date
    canonical: str  # ground truth: we authored it, so it is free
    lang: str = "hi-IN"
    voice: str = ""
    degradation: dict = field(default_factory=dict)

    # Filled in by synth.render() — exact by construction, never estimated.
    seg_bounds_ms: list[tuple[int, int]] = field(default_factory=list)
    true_end_ms: int = 0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def internal_pauses(self) -> list[tuple[int, int]]:
        """(start_ms, end_ms) of every pause *inside* the utterance.

        A turn-end decision landing in one of these is a premature interruption:
        the caller had not finished, and we know that because we built the gap.
        """
        gaps = []
        for (_, end), seg in zip(self.seg_bounds_ms, self.segments):
            if seg.pause_after_ms and end < self.true_end_ms:
                gaps.append((end, end + seg.pause_after_ms))
        return gaps


@dataclass
class Event:
    """Anything the system under test emitted, timestamped against audio t=0."""

    kind: str  # speech_start | speech_end | transcript | agent_turn
    t_ms: int
    text: str = ""


@dataclass
class Result:
    spec_id: str
    adapter: str
    events: list[Event] = field(default_factory=list)
    transcript: str = ""
    agent_entity: str | None = None
    agent_text: str = ""
    confirmed: bool | None = None  # did the agent seek confirmation before acting?
    error: str = ""

    def first(self, kind: str) -> Event | None:
        return next((e for e in self.events if e.kind == kind), None)


def to_jsonl(spec: CallSpec, result: Result) -> str:
    return json.dumps({"spec": asdict(spec), "result": asdict(result)}, ensure_ascii=False)


def from_jsonl(line: str) -> tuple[CallSpec, Result]:
    d = json.loads(line)
    s = d["spec"]
    spec = CallSpec(**{**s, "segments": [Segment(**x) for x in s["segments"]]})
    spec.seg_bounds_ms = [tuple(b) for b in spec.seg_bounds_ms]
    r = d["result"]
    return spec, Result(**{**r, "events": [Event(**e) for e in r["events"]]})

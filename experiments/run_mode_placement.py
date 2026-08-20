"""Run Sarvam's output modes with one fixed filler and resumable evidence.

The old result changed both the placement and the filler on some rows. This run keeps
``matlab`` fixed, uses leading-filler audio as a control, and writes one row after every
call: four digit utterances × four output modes × two placements × six repeats = 192.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asli import score, synth
from asli.cli import call, load_entities, load_env
from asli.drive import SarvamWS
from asli.spec import CallSpec, Segment

ROOT = Path(__file__).parent.parent
RESULT_PATH = ROOT / "results" / "mode_placement_fixed.json"
MATRIX_PATH = ROOT / "results" / "mode_matrix_fixed.json"
FILLER = "matlab"
PAUSE_MS = 700
GATE_MS = 500
MODES = ("transcribe", "verbatim", "translit", "codemix")
PLACEMENTS = ("control", "hesitation")
REPEATS = range(6)
REQUIRED_CREDENTIALS = ("SARVAM_API_KEY", "ELEVEN_API_KEY", "ELEVEN_VOICE_ID")

# `entities.yaml` keeps dig-03 and dig-04 as one render segment. The probe needs the
# same semantic words split immediately before the number, so this mapping supplies
# only the splice boundary; canonical values, types, and languages stay in entities.yaml.
PROBE_SEGMENTS = {
    "dig-01": ("Mera mobile number hai", "nine eight double seven, triple one"),
    "dig-02": ("Account ke last five digits", "char zero double five six"),
    "dig-03": ("The reference is", "eight, double zero, nine"),
    "dig-04": ("Number likhiye", "triple seven one two three four five"),
}


def probe_specs() -> list[CallSpec]:
    """Return the four pre-existing digit stimuli with explicit entity boundaries."""
    source = {spec.id: spec for spec in load_entities()}
    specs = []
    for spec_id, (head, entity) in PROBE_SEGMENTS.items():
        original = source[spec_id]
        specs.append(CallSpec(
            id=spec_id,
            entity_type=original.entity_type,
            canonical=original.canonical,
            lang=original.lang,
            segments=[Segment(head), Segment(entity)],
        ))
    return specs


def build_placement(spec: CallSpec, placement: str, filler: str = FILLER) -> CallSpec:
    """Insert the fixed filler in exactly one of the two registered positions."""
    out = copy.deepcopy(spec)
    if placement == "control":
        out.segments = [Segment(filler), *out.segments]
    elif placement == "hesitation":
        head, entity = out.segments[:-1], out.segments[-1]
        out.segments = [*head, Segment(filler, pause_after_ms=PAUSE_MS, kind="filler"), entity]
    else:
        raise ValueError(f"unknown placement: {placement}")
    out.id = f"{spec.id}-{placement}"
    return out


def row_key(row: dict) -> tuple[str, str, str, int]:
    return row["id"], row["mode"], row["placement"], row["run"]


def missing_credentials(env: dict[str, str]) -> list[str]:
    return [key for key in REQUIRED_CREDENTIALS if not env.get(key)]


def summary(rows: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """Aggregate only completed calls; failed rows remain visible but unmeasured."""
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.get("error"):
            groups[row["mode"]][row["placement"]].append(row)
    return {
        mode: {
            placement: {"survived": sum(row["survived"] for row in selected), "n": len(selected)}
            for placement, selected in placements.items()
        }
        for mode, placements in groups.items()
    }


def as_mode_matrix(rows: list[dict]) -> dict[str, dict[str, list[int]]]:
    """Adapt completed rows to the compact control/hesitation site schema."""
    return {
        mode: {
            placement: [cell["survived"], cell["n"]]
            for placement, cell in placements.items()
        }
        for mode, placements in summary(rows).items()
    }


def read_rows(path: Path) -> dict[tuple[str, str, str, int], dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out = {row_key(row): row for row in rows}
    for row in out.values():
        if row.get("filler") != FILLER:
            raise ValueError(f"{path} contains a non-fixed filler: {row.get('filler')!r}")
    return out


def checkpoint(path: Path, rows: dict[tuple[str, str, str, int], dict]) -> None:
    """Atomically replace the result after every call, preserving one row per key."""
    ordered = sorted(rows.values(), key=lambda r: (r["mode"], r["placement"], r["id"], r["run"]))
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n")
    pending.replace(path)


def write_mode_matrix(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(as_mode_matrix(rows), ensure_ascii=False, indent=2) + "\n")


def run_one(spec: CallSpec, *, mode: str, placement: str, repeat: int) -> dict:
    stimulus = build_placement(spec, placement)
    result = call(
        SarvamWS(language_code=stimulus.lang, rate=16000, silence_duration_ms=GATE_MS, mode=mode),
        synth.render(stimulus),
        stimulus,
    )
    extracted = score.normalise(stimulus.entity_type, result.transcript)
    return {
        "id": spec.id,
        "mode": mode,
        "placement": placement,
        "run": repeat,
        "filler": FILLER,
        "pause_ms": PAUSE_MS if placement == "hesitation" else 0,
        "gate_ms": GATE_MS,
        "truth": spec.canonical,
        "extracted": extracted,
        "survived": bool(extracted) and spec.canonical in extracted,
        "transcript": result.transcript,
        "error": result.error,
    }


def print_summary(rows: list[dict]) -> None:
    print("\nmode         placement    survived")
    for mode in MODES:
        for placement in PLACEMENTS:
            cell = summary(rows).get(mode, {}).get(placement)
            if cell:
                print(f"{mode:<12} {placement:<12} {cell['survived']}/{cell['n']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULT_PATH)
    parser.add_argument("--matrix-out", type=Path, default=MATRIX_PATH)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.out)
    if args.report_only:
        completed = list(rows.values())
        print_summary(completed)
        write_mode_matrix(args.matrix_out, completed)
        print(f"wrote {args.matrix_out}")
        return 0

    load_env()
    if missing := missing_credentials(os.environ):
        parser.error(f"missing required .env values: {', '.join(missing)}")

    total = len(PROBE_SEGMENTS) * len(MODES) * len(PLACEMENTS) * len(REPEATS)
    successful = {key for key, row in rows.items() if not row.get("error")}
    print(f"fixed-filler mode probe: {len(successful)}/{total} successful rows; filler={FILLER}")
    for spec in probe_specs():
        for mode in MODES:
            for placement in PLACEMENTS:
                for repeat in REPEATS:
                    key = (spec.id, mode, placement, repeat)
                    if key in successful:
                        continue
                    row = run_one(spec, mode=mode, placement=placement, repeat=repeat)
                    rows[key] = row
                    checkpoint(args.out, rows)
                    status = "ERR" if row["error"] else "kept" if row["survived"] else "lost"
                    print(f"{spec.id:<7} {mode:<11} {placement:<11} run={repeat} {status}", flush=True)

    print_summary(list(rows.values()))
    write_mode_matrix(args.matrix_out, list(rows.values()))
    print(f"wrote {args.matrix_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

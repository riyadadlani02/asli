"""Explicit local and opt-in live commands for TurnBench."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

from ..fit import read_audio
from ..synth import write_wav
from .auto_label import EndpointObservation, OpenAISemanticObserver, predict_candidate
from .auto_report import (
    DiarBenchExportProvenance,
    compare_auto_predictions,
    validate_export_candidates,
)
from .auto_schema import (
    AutoPrediction,
    DiarBenchCandidate,
    read_candidates,
    read_predictions,
    read_references,
    write_candidates,
    write_references,
)
from .diarbench import convert_diarbench_sample
from .report import score_inputs
from .schema import (
    DecisionLabel,
    ProviderTrace,
    Recording,
    _decode_strict_json,
    read_jsonl,
)


DIARBENCH_DATASET = "sarvamai/indic-diarbench"
DIARBENCH_REQUESTED_REVISION = "main"
REALTIME_DECISION_MODEL = "gpt-realtime"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PROVIDER_LANGUAGE = {
    "Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi",
    "Kannada": "kn", "Kashmiri": "ks", "Malayalam": "ml", "Marathi": "mr",
    "Nepali": "ne", "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa",
    "Sindhi": "sd", "Tamil": "ta", "Telugu": "te", "Urdu": "ur",
    "Bodo": None, "Dogri": None, "Konkani": None, "Maithili": None,
    "Manipuri": None, "Santali": None,
}


@dataclass(frozen=True)
class DiarBenchLoad:
    """A streaming public-data view plus its immutable resolved revision."""

    rows: Iterable[Mapping[str, object]]
    resolved_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.resolved_revision, str) or not _COMMIT_SHA.fullmatch(
            self.resolved_revision
        ):
            raise ValueError("resolved DiarBench revision must be a 40-character commit SHA")


def _resolved_revision(rows: object) -> str:
    """Read the Hub commit retained by datasets without another Hub request."""

    def find(value: object, depth: int = 0) -> str | None:
        if depth > 3:
            return None
        if isinstance(value, str):
            match = re.search(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", value)
            if match:
                return match.group(1)
            return None
        if isinstance(value, Mapping):
            for nested in value.values():
                resolved = find(nested, depth + 1)
                if resolved:
                    return resolved
            return None
        for name in (
            "_commit_hash", "commit_hash", "revision", "base_path", "info",
            "_info", "_ex_iterable", "kwargs",
        ):
            resolved = find(getattr(value, name, None), depth + 1)
            if resolved:
                return resolved
        return None

    resolved = find(rows)
    if resolved:
        return resolved
    raise ValueError(
        "datasets did not expose an immutable DiarBench commit revision; refusing mutable provenance"
    )


def _load_diarbench_rows(
    *, dataset: str, requested_revision: str, config: str, split: str,
    streaming: bool, limit: int,
) -> DiarBenchLoad:
    """Load only an explicitly selected language via the optional dependency."""

    try:
        import datasets
    except ModuleNotFoundError as exc:
        if exc.name in {None, "datasets"}:
            raise ValueError(
                "DiarBench export needs the optional dependency; "
                "install it with: pip install 'asli[diarbench]'"
            ) from exc
        raise
    rows = datasets.load_dataset(
        dataset, name=config, revision=requested_revision, split=split,
        streaming=streaming,
    )
    return DiarBenchLoad(islice(rows, limit), _resolved_revision(rows))


# These seams keep command tests offline. Their defaults are the only live paths.
DIARBENCH_LOADER_FACTORY: Callable[..., DiarBenchLoad] = _load_diarbench_rows
AUTO_OBSERVER_FACTORY: Callable[..., Callable[[np.ndarray, int, str], EndpointObservation]] = OpenAISemanticObserver
AUDIO_READER: Callable[[Path], tuple[np.ndarray, int]] = read_audio
RUN_ID_FACTORY: Callable[[], str] = lambda: str(uuid4())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asli-turnbench")
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="score local TurnBench JSONL files")
    score.add_argument("--recordings", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--events", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--provider", required=True)
    score.add_argument("--config-json", default="{}")

    diarbench = commands.add_parser("diarbench", help="bounded Indic DiarBench export")
    diarbench_commands = diarbench.add_subparsers(dest="diarbench_command", required=True)
    export = diarbench_commands.add_parser("export", help="export one bounded DiarBench language")
    export.add_argument("--language", required=True)
    export.add_argument("--limit", type=int, required=True)
    export.add_argument("--min-pause-ms", type=int, required=True)
    export.add_argument("--max-pause-ms", type=int, required=True)
    export.add_argument("--context-ms", type=int, default=5000)
    export.add_argument("--out-dir", type=Path, required=True)

    auto = commands.add_parser("auto", help="run or compare automatic observations")
    auto_commands = auto.add_subparsers(dest="auto_command", required=True)
    label = auto_commands.add_parser("label", help="label candidate audio with semantic VAD")
    label.add_argument("--candidates", type=Path, required=True)
    label.add_argument("--agent", choices=("openai",), required=True)
    label.add_argument("--model", required=True)
    label.add_argument("--context-ms", type=int, required=True)
    label.add_argument("--manifest", type=Path)
    label.add_argument("--out", type=Path, required=True)
    compare = auto_commands.add_parser("compare", help="compare predictions with timing references")
    compare.add_argument("--candidates", type=Path, required=True)
    compare.add_argument("--references", type=Path, required=True)
    compare.add_argument("--predictions", type=Path, required=True)
    compare.add_argument("--manifest", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    return parser


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_predictions(path: Path, rows: Iterable[AutoPrediction]) -> None:
    payload = "".join(
        json.dumps(row.to_dict(), sort_keys=True, allow_nan=False) + "\n"
        for row in sorted(rows, key=lambda row: row.decision_id)
    )
    _atomic_text(path, payload)


def _validate_export_args(args: argparse.Namespace) -> None:
    if not args.language.strip():
        raise ValueError("--language must be non-empty")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.min_pause_ms < 0 or args.max_pause_ms < args.min_pause_ms:
        raise ValueError("--min-pause-ms and --max-pause-ms must satisfy 0 <= min <= max")
    if args.context_ms < 0:
        raise ValueError("--context-ms must be a non-negative integer")
    if args.language not in _PROVIDER_LANGUAGE:
        raise ValueError("--language must be one of the 22 Indic DiarBench config names")


def _effective_provider_language(language: str) -> str | None:
    """Return the optional ISO-639-1 transcription hint for a public config name."""

    try:
        return _PROVIDER_LANGUAGE[language]
    except KeyError as exc:
        raise ValueError(f"unknown Indic DiarBench language: {language}") from exc


def _decoded_audio(row: Mapping[str, object], *, sample_id: str) -> tuple[np.ndarray, int]:
    audio = row.get("audio")
    channel_first = False
    if isinstance(audio, Mapping):
        values = audio.get("array")
        rate = audio.get("sampling_rate")
    else:
        decode = getattr(audio, "get_all_samples", None)
        if not callable(decode):
            raise ValueError(
                f"{sample_id}: audio must be a legacy mapping or an AudioDecoder"
            )
        try:
            decoded = decode()
            values = getattr(decoded, "data")
            rate = getattr(decoded, "sample_rate")
        except Exception as exc:
            raise ValueError(f"{sample_id}: AudioDecoder could not decode audio: {exc}") from exc
        channel_first = True
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError(f"{sample_id}: audio sampling rate must be a positive integer")
    try:
        values = np.asarray(values)
        normalized_float = np.issubdtype(values.dtype, np.floating)
        if values.ndim == 2:
            values = values.mean(axis=0 if channel_first else 1)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError
        if normalized_float:
            if np.max(np.abs(values)) > 1:
                raise ValueError
            values = np.rint(values * 32767)
        pcm = np.clip(values, -32768, 32767).astype(np.int16)
    except (TypeError, ValueError):
        raise ValueError(f"{sample_id}: audio.array must be a finite mono PCM array") from None
    return pcm, rate


def _adapt_published_diarbench_row(
    row: Mapping[str, object], *, language: str
) -> dict[str, object]:
    """Map the public Indic DiarBench row contract into the pure timing converter."""

    transcript = row.get("annotated_transcript")
    if not isinstance(transcript, list):
        raise ValueError("annotated_transcript must be a list")
    segments: list[dict[str, object]] = []
    for index, entry in enumerate(transcript):
        if not isinstance(entry, Mapping):
            raise ValueError(f"annotated_transcript[{index}] must be an object")
        segments.append({
            "speaker_id": entry.get("speaker_id"),
            "start": entry.get("start_time"),
            "end": entry.get("end_time"),
        })
    return {
        "sample_id": row.get("sample_id"),
        "recording_id": row.get("recording_id"),
        "language": language,
        "condition": row.get("dataset_type", "diarbench"),
        "segments": segments,
        "audio": row.get("audio"),
    }


def _validate_annotation_duration(
    row: Mapping[str, object], *, sample_id: str, duration_ms: int
) -> None:
    transcript = row.get("annotated_transcript")
    if not isinstance(transcript, list):
        raise ValueError(f"{sample_id}: annotated_transcript must be a list")
    for index, entry in enumerate(transcript):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{sample_id}: annotated_transcript[{index}] must be an object")
        for field in ("start_time", "end_time"):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{sample_id}: annotated_transcript[{index}].{field} must be seconds")
            rounded_ms = int(round(value * 1000))
            if rounded_ms < 0 or rounded_ms > duration_ms:
                raise ValueError(
                    f"{sample_id}: annotated_transcript[{index}].{field} exceeds decoded audio duration"
                )


def _export_diarbench(args: argparse.Namespace) -> None:
    _validate_export_args(args)
    loaded = DIARBENCH_LOADER_FACTORY(
        dataset=DIARBENCH_DATASET, requested_revision=DIARBENCH_REQUESTED_REVISION,
        config=args.language, split="test", streaming=True, limit=args.limit,
    )
    # Pull the first row before any output directory; missing `datasets` leaves no output.
    iterator = islice(iter(loaded.rows), args.limit)
    try:
        first_row = next(iterator)
    except StopIteration:
        source_rows: Iterable[Mapping[str, object]] = ()
    else:
        source_rows = chain((first_row,), iterator)
    if args.out_dir.exists():
        raise ValueError(f"--out-dir already exists: {args.out_dir}")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.out_dir.parent, prefix=f".{args.out_dir.name}."))
    candidates: list[DiarBenchCandidate] = []
    references = []
    try:
        audio_dir = temporary / "audio"
        audio_dir.mkdir()
        for index, row in enumerate(source_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"DiarBench row {index}: expected an object")
            adapted = _adapt_published_diarbench_row(row, language=args.language)
            raw_sample_id = adapted.get("sample_id")
            sample_id = raw_sample_id if isinstance(raw_sample_id, str) and raw_sample_id else f"row-{index}"
            name = f"{index:06d}.wav"
            pcm, rate = _decoded_audio(adapted, sample_id=sample_id)
            _validate_annotation_duration(
                row, sample_id=sample_id, duration_ms=len(pcm) * 1000 // rate
            )
            write_wav(audio_dir / name, pcm, rate)
            converted_candidates, converted_references = convert_diarbench_sample(
                adapted, min_pause_ms=args.min_pause_ms, max_pause_ms=args.max_pause_ms,
                audio_path=str(args.out_dir / "audio" / name), context_ms=args.context_ms,
            )
            candidates.extend(converted_candidates)
            eligible_ids = {candidate.decision_id for candidate in converted_candidates}
            references.extend(
                reference for reference in converted_references
                if reference.outcome not in {"continue", "yield"}
                or reference.candidate.decision_id in eligible_ids
            )
        provenance = DiarBenchExportProvenance(
            dataset=DIARBENCH_DATASET, dataset_revision=loaded.resolved_revision,
            requested_languages=(args.language,), min_pause_ms=args.min_pause_ms,
            max_pause_ms=args.max_pause_ms, context_ms=args.context_ms,
        )
        write_candidates(temporary / "candidates.jsonl", sorted(candidates, key=lambda row: row.decision_id))
        write_references(temporary / "references.jsonl", sorted(references, key=lambda row: row.candidate.decision_id))
        (temporary / "manifest.json").write_text(
            json.dumps(provenance.to_dict(), sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.out_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_audio_cache(
    candidates: Iterable[DiarBenchCandidate], *, context_ms: int
) -> dict[str, tuple[np.ndarray, int]]:
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for candidate in candidates:
        if candidate.context_start_ms != max(
            0, candidate.previous_speech_end_ms - context_ms
        ):
            raise ValueError(
                f"{candidate.decision_id}: context_start_ms does not match --context-ms"
            )
        if candidate.audio_path not in cache:
            try:
                pcm, rate = AUDIO_READER(Path(candidate.audio_path))
                if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
                    raise ValueError("invalid sample rate")
                cache[candidate.audio_path] = (pcm, rate)
            except Exception as exc:
                raise ValueError(f"{candidate.decision_id}: unable to read candidate audio: {exc}") from exc
        pcm, rate = cache[candidate.audio_path]
        start = candidate.context_start_ms * rate // 1000
        end = candidate.observation_end_ms * rate // 1000
        if start < 0 or end <= start or end > len(pcm):
            raise ValueError(f"{candidate.decision_id}: candidate audio cannot reach observation boundary")
    return cache


def _label_auto(args: argparse.Namespace) -> None:
    if args.context_ms <= 0:
        raise ValueError("--context-ms must be a positive integer")
    candidates = read_candidates(args.candidates)
    manifest = _read_manifest(args.manifest or args.candidates.parent / "manifest.json")
    if args.context_ms != manifest.context_ms:
        raise ValueError("--context-ms does not match export manifest")
    validate_export_candidates(candidates, export_provenance=manifest)
    audio_cache = _validated_audio_cache(candidates, context_ms=args.context_ms)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY must be set to run auto label")
    observer = AUTO_OBSERVER_FACTORY(model=args.model)
    effective_language = _effective_provider_language(manifest.requested_languages[0])
    config = {
        "context_ms": args.context_ms,
        "turn_detection": "semantic_vad",
        "eagerness": "auto",
        "create_response": False,
        "require_endpoint_timestamps": True,
        "trailing_silence_ms": 0,
        "effective_language": effective_language,
        "transcription_model": args.model,
        "realtime_model": REALTIME_DECISION_MODEL,
    }
    run_id = RUN_ID_FACTORY()

    def cached_reader(path: str) -> tuple[np.ndarray, int]:
        return audio_cache[path]

    predictions = [
        predict_candidate(
            candidate, read_audio=cached_reader, observe=observer, run_id=run_id,
            agent=args.agent, model=REALTIME_DECISION_MODEL, config=config,
            provider_language=effective_language,
        )
        for candidate in candidates
    ]
    _atomic_predictions(args.out, predictions)


def _read_manifest(path: Path) -> DiarBenchExportProvenance:
    raw = _decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("export manifest must be an object")
    return DiarBenchExportProvenance.from_dict(raw)


def _compare_auto(args: argparse.Namespace) -> None:
    manifest = args.manifest or args.candidates.parent / "manifest.json"
    report = compare_auto_predictions(
        read_candidates(args.candidates), read_references(args.references),
        read_predictions(args.predictions), export_provenance=_read_manifest(manifest),
    )
    _atomic_text(args.out, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _score(args: argparse.Namespace) -> None:
    config = _decode_strict_json(args.config_json)
    if not isinstance(config, dict):
        raise ValueError("--config-json must decode to an object")
    report = score_inputs(
        read_jsonl(args.recordings, Recording.from_dict),
        read_jsonl(args.labels, DecisionLabel.from_dict),
        read_jsonl(args.events, ProviderTrace.from_dict),
        provider=args.provider, config=config,
    )
    _atomic_text(args.out, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run a local score or one explicit automatic-evaluation operation."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            _score(args)
        elif args.command == "diarbench":
            _export_diarbench(args)
        elif args.auto_command == "label":
            _label_auto(args)
        else:
            _compare_auto(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0

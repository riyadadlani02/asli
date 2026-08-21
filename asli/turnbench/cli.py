"""Explicit local and opt-in live commands for TurnBench."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
import re
import wave
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
from .policy_features import extract_policy_features
from .policy_model import fit_policy, make_group_split
from .policy_report import replay_policy
from .policy_schema import (
    PolicyFeature,
    PolicySplit,
    read_policy_artifact,
    read_policy_features,
    read_policy_split,
)
from .report import score_inputs
from .schema import (
    DecisionLabel,
    ProviderTrace,
    Recording,
    SchemaError,
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
        if depth > 5:
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
        if isinstance(value, (list, tuple)):
            for nested in value:
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
    rows = rows.cast_column("audio", datasets.Audio(decode=False))
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

    policy = commands.add_parser("policy", help="fit and replay a local calibrated policy")
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    features = policy_commands.add_parser("features", help="extract local policy features")
    features.add_argument("--candidates", type=Path, required=True)
    features.add_argument("--manifest", type=Path)
    features.add_argument("--semantic", type=Path)
    features.add_argument("--out", type=Path, required=True)
    split = policy_commands.add_parser("split", help="make a grouped local policy split")
    split.add_argument("--features", type=Path, required=True)
    split.add_argument("--language", required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--out", type=Path, required=True)
    fit = policy_commands.add_parser("fit", help="fit a local calibrated policy")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--references", type=Path, required=True)
    fit.add_argument("--split", type=Path, required=True)
    fit.add_argument("--language", required=True)
    fit.add_argument("--out", type=Path, required=True)
    replay = policy_commands.add_parser("replay", help="replay a local calibrated policy")
    replay.add_argument("--features", type=Path, required=True)
    replay.add_argument("--references", type=Path, required=True)
    replay.add_argument("--split", type=Path, required=True)
    replay.add_argument("--policy", type=Path, required=True)
    replay.add_argument("--semantic", type=Path)
    replay.add_argument("--out", type=Path, required=True)
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


def _atomic_policy_features(path: Path, rows: Iterable[PolicyFeature]) -> None:
    """Write sorted feature JSONL through the same atomic output seam as reports."""
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
    raw_pcm = False
    if isinstance(audio, Mapping):
        encoded = audio.get("bytes")
        if isinstance(encoded, (bytes, bytearray)):
            try:
                with wave.open(io.BytesIO(encoded), "rb") as handle:
                    if handle.getcomptype() != "NONE" or handle.getsampwidth() != 2:
                        raise ValueError
                    rate = handle.getframerate()
                    channels = handle.getnchannels()
                    values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
                    if channels > 1:
                        values = values.reshape(-1, channels).mean(axis=1)
                    raw_pcm = True
            except (EOFError, ValueError, wave.Error):
                raise ValueError(f"{sample_id}: audio.bytes must be uncompressed 16-bit WAV") from None
            channel_first = False
        else:
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
        normalized_float = np.issubdtype(values.dtype, np.floating) and not raw_pcm
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


# A diarisation annotation routinely ends a fraction after the trimmed audio does.
# Measured over 45 published Hindi recordings: 40 overrun by nothing at all, and the
# other five by 62, 86, 87, 102 and 154 ms — every one of them on the final segment, the
# worst being 154 ms against a 1,902-second file. Rejecting those is discarding good
# recordings over annotation slack; a genuine audio/annotation mismatch is seconds wrong
# across many segments, not a tenth of a second on the last one. So a small overrun is
# clamped and counted, and anything past the bound still fails loudly.
ANNOTATION_OVERRUN_TOLERANCE_MS = 250


def _validate_annotation_duration(
    row: Mapping[str, object], adapted: Mapping[str, object], *,
    sample_id: str, duration_ms: int,
) -> int:
    """Return how many annotation times were clamped to the decoded duration."""

    transcript = row.get("annotated_transcript")
    if not isinstance(transcript, list):
        raise ValueError(f"{sample_id}: annotated_transcript must be a list")
    segments = adapted.get("segments")
    if not isinstance(segments, list) or len(segments) != len(transcript):
        raise ValueError(f"{sample_id}: adapted segments do not match annotated_transcript")
    clamped = 0
    for index, entry in enumerate(transcript):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{sample_id}: annotated_transcript[{index}] must be an object")
        for field, adapted_field in (("start_time", "start"), ("end_time", "end")):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{sample_id}: annotated_transcript[{index}].{field} must be seconds")
            rounded_ms = int(round(value * 1000))
            if rounded_ms < 0:
                raise ValueError(
                    f"{sample_id}: annotated_transcript[{index}].{field} is negative"
                )
            overrun_ms = rounded_ms - duration_ms
            if overrun_ms > ANNOTATION_OVERRUN_TOLERANCE_MS:
                raise ValueError(
                    f"{sample_id}: annotated_transcript[{index}].{field} exceeds decoded "
                    f"audio duration by {overrun_ms} ms, over the "
                    f"{ANNOTATION_OVERRUN_TOLERANCE_MS} ms tolerance"
                )
            if overrun_ms > 0:
                # clamp the adapted copy: that is the one the converter reads
                segments[index][adapted_field] = duration_ms / 1000
                clamped += 1
    return clamped


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
    clamped_annotations = 0
    skipped: list[tuple[str, str]] = []
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
            # One unusable recording must not cost the whole export. The published Hindi
            # split contains a segment whose end precedes its start by 783 seconds
            # (hindi_038[93]), and refusing it used to abort a 40-recording run and leave
            # nothing behind. Every skip is named on stderr so a shrinking corpus cannot
            # pass unnoticed.
            try:
                pcm, rate = _decoded_audio(adapted, sample_id=sample_id)
                clamped_annotations += _validate_annotation_duration(
                    row, adapted, sample_id=sample_id, duration_ms=len(pcm) * 1000 // rate
                )
                converted_candidates, converted_references = convert_diarbench_sample(
                    adapted, min_pause_ms=args.min_pause_ms, max_pause_ms=args.max_pause_ms,
                    audio_path=str(args.out_dir / "audio" / name), context_ms=args.context_ms,
                )
            except (SchemaError, ValueError) as exc:
                skipped.append((sample_id, str(exc)))
                print(f"skipping {sample_id}: {exc}", file=sys.stderr)
                continue
            write_wav(audio_dir / name, pcm, rate)
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
        if skipped and not candidates:
            # every row was unusable: emitting an empty export would be a ghost artifact
            raise ValueError("; ".join(reason for _, reason in skipped))
        if skipped:
            print(f"skipped {len(skipped)} unusable recording(s): "
                  f"{', '.join(sid for sid, _ in skipped)}", file=sys.stderr)
        if clamped_annotations:
            print(
                f"clamped {clamped_annotations} annotation time(s) to the decoded audio "
                f"duration, each within {ANNOTATION_OVERRUN_TOLERANCE_MS} ms",
                file=sys.stderr,
            )
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


def _policy_features(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.manifest or args.candidates.parent / "manifest.json")
    semantic_predictions = read_predictions(args.semantic) if args.semantic else ()
    features = extract_policy_features(
        read_candidates(args.candidates),
        export_provenance=manifest,
        semantic_predictions=semantic_predictions,
    )
    _atomic_policy_features(args.out, features)


def _policy_split(args: argparse.Namespace) -> None:
    if not args.language.strip():
        raise ValueError("--language must be non-empty")
    features = read_policy_features(args.features)
    source_recording_count = len({
        feature.source_recording_id
        for feature in features
        if feature.language == args.language
    })
    if source_recording_count < 20:
        raise ValueError(
            f"language {args.language} requires 20 independent source recordings"
        )
    split = make_group_split(features, language=args.language, seed=args.seed)
    _atomic_text(
        args.out,
        json.dumps(split.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _policy_fit(args: argparse.Namespace) -> None:
    if not args.language.strip():
        raise ValueError("--language must be non-empty")
    features = read_policy_features(args.features)
    split = read_policy_split(args.split)
    language_features = [feature for feature in features if feature.language == args.language]
    source_recording_ids = {
        feature.source_recording_id for feature in language_features
    }
    if len(source_recording_ids) < 20:
        raise ValueError(
            f"language {args.language} requires 20 independent source recordings"
        )
    split_source_recording_ids = set(split.train_source_recording_ids)
    split_source_recording_ids.update(split.calibration_source_recording_ids)
    split_source_recording_ids.update(split.test_source_recording_ids)
    if source_recording_ids != split_source_recording_ids:
        raise ValueError("feature source recording IDs must exactly match split")
    artifact = fit_policy(
        features,
        read_references(args.references),
        split,
        language=args.language,
    )
    _atomic_text(
        args.out,
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _replay_semantic_predictions(
    features: Iterable[PolicyFeature], split: PolicySplit,
    predictions: Iterable[AutoPrediction],
) -> list[AutoPrediction]:
    """Validate a completed local semantic artifact, then retain held-out rows."""
    feature_by_id: dict[str, PolicyFeature] = {}
    for feature in features:
        if not isinstance(feature, PolicyFeature):
            raise ValueError("features must contain PolicyFeature records")
        if feature.decision_id in feature_by_id:
            raise ValueError(f"duplicate feature decision_id: {feature.decision_id}")
        feature_by_id[feature.decision_id] = feature

    prediction_by_id: dict[str, AutoPrediction] = {}
    provenance: tuple[str, str, str, str] | None = None
    for prediction in predictions:
        if not isinstance(prediction, AutoPrediction):
            raise ValueError("semantic_predictions must contain AutoPrediction records")
        if prediction.decision_id in prediction_by_id:
            raise ValueError(
                f"duplicate semantic prediction decision_id: {prediction.decision_id}"
            )
        value = (
            prediction.run_id,
            prediction.agent,
            prediction.model,
            json.dumps(
                prediction.config, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ),
        )
        if provenance is None:
            provenance = value
        elif value != provenance:
            raise ValueError("mixed semantic prediction provenance is not allowed")
        prediction_by_id[prediction.decision_id] = prediction

    if set(prediction_by_id) != set(feature_by_id):
        raise ValueError("semantic prediction IDs must exactly match policy features")
    test_groups = set(split.test_source_recording_ids)
    test_ids = sorted(
        feature.decision_id
        for feature in feature_by_id.values()
        if feature.source_recording_id in test_groups
    )
    return [prediction_by_id[decision_id] for decision_id in test_ids]


def _policy_replay(args: argparse.Namespace) -> None:
    features = read_policy_features(args.features)
    split = read_policy_split(args.split)
    semantic_predictions = ()
    if args.semantic:
        semantic_predictions = _replay_semantic_predictions(
            features, split, read_predictions(args.semantic),
        )
    report = replay_policy(
        features,
        read_references(args.references),
        split,
        read_policy_artifact(args.policy),
        semantic_predictions=semantic_predictions,
    )
    _atomic_text(
        args.out,
        json.dumps({"report": report}, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


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
        elif args.command == "policy":
            if args.policy_command == "features":
                _policy_features(args)
            elif args.policy_command == "split":
                _policy_split(args)
            elif args.policy_command == "fit":
                _policy_fit(args)
            else:
                _policy_replay(args)
        elif args.auto_command == "label":
            _label_auto(args)
        else:
            _compare_auto(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    main()

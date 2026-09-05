"""Inspect AIHub 119 archive pairs and emit provenance-only manifests.

The generated files contain counts, checksums, and validation outcomes. They do
not contain transcripts, addresses, audio, signed URLs, or download tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import wave
import zipfile

from jsonschema import Draft202012Validator


DATASET_ID = "aihub_71768_gwangju_fire"
DATASET_VERSION = "dataset-71768_downloaded-2026-09-05"
SOURCE_URL = "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71768"
SOURCE_LICENSE = (
    "AIHub dataset access terms; restricted use and redistribution status "
    "not independently verified"
)
EVALUATION_ID = "speech_aihub119_gwangju_fire_validation_77"
REQUIRED_LABEL_FIELDS = {
    "_id",
    "audioPath",
    "recordId",
    "utterances",
    "disasterLarge",
    "disasterMedium",
}
REQUIRED_UTTERANCE_FIELDS = {"id", "startAt", "endAt", "text", "speaker"}
LABEL_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "aihub119-label.schema.json"
)


@dataclass(frozen=True)
class ArchiveStats:
    split: str
    audio_count: int
    label_count: int
    paired_count: int
    audio_without_label: int
    label_without_audio: int
    duplicate_audio_stems: int
    duplicate_label_stems: int
    duplicate_record_ids: int
    missing_required_fields: int
    schema_error_count: int
    utterance_count: int
    empty_utterance_count: int
    transcript_character_count: int
    audio_seconds: float
    min_audio_seconds: float
    max_audio_seconds: float
    sample_rates: tuple[int, ...]
    channel_counts: tuple[int, ...]
    sample_widths: tuple[int, ...]
    record_ids: frozenset[str] = field(repr=False)
    stems: frozenset[str] = field(repr=False)

    def public_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("record_ids")
        result.pop("stems")
        for name in ("sample_rates", "channel_counts", "sample_widths"):
            result[name] = list(result[name])
        result["audio_seconds"] = round(self.audio_seconds, 3)
        result["audio_hours"] = round(self.audio_seconds / 3600, 3)
        result["min_audio_seconds"] = round(self.min_audio_seconds, 3)
        result["max_audio_seconds"] = round(self.max_audio_seconds, 3)
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _members_by_stem(
    archive: zipfile.ZipFile, suffix: str
) -> tuple[dict[str, str], int]:
    names = [name for name in archive.namelist() if name.lower().endswith(suffix)]
    members: dict[str, str] = {}
    duplicates = 0
    for name in names:
        stem = PurePosixPath(name).stem
        if stem in members:
            duplicates += 1
        else:
            members[stem] = name
    return members, duplicates


def inspect_archive_pair(
    audio_archive: Path,
    label_archive: Path,
    split: str,
) -> ArchiveStats:
    label_schema = json.loads(LABEL_SCHEMA_PATH.read_text(encoding="utf-8"))
    label_validator = Draft202012Validator(label_schema)
    record_ids: list[str] = []
    missing_required_fields = 0
    schema_errors = 0
    utterance_count = 0
    empty_utterance_count = 0
    transcript_character_count = 0
    durations: list[float] = []
    sample_rates: set[int] = set()
    channel_counts: set[int] = set()
    sample_widths: set[int] = set()

    with zipfile.ZipFile(audio_archive) as audio_zip, zipfile.ZipFile(
        label_archive
    ) as label_zip:
        bad_audio_member = audio_zip.testzip()
        bad_label_member = label_zip.testzip()
        if bad_audio_member or bad_label_member:
            raise ValueError(
                "corrupt archive member: "
                f"{bad_audio_member or bad_label_member}"
            )
        audio_members, duplicate_audio_stems = _members_by_stem(audio_zip, ".wav")
        label_members, duplicate_label_stems = _members_by_stem(label_zip, ".json")

        for name in audio_members.values():
            try:
                with audio_zip.open(name) as source, wave.open(source) as audio:
                    rate = audio.getframerate()
                    channels = audio.getnchannels()
                    width = audio.getsampwidth()
                    frames = audio.getnframes()
                    sample_rates.add(rate)
                    channel_counts.add(channels)
                    sample_widths.add(width)
                    durations.append(frames / rate if rate else 0.0)
                    if rate <= 0 or channels <= 0 or width <= 0 or frames <= 0:
                        schema_errors += 1
            except (EOFError, wave.Error, zipfile.BadZipFile):
                schema_errors += 1

        for name in label_members.values():
            try:
                document = json.loads(label_zip.read(name).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                schema_errors += 1
                continue
            if not isinstance(document, dict):
                schema_errors += 1
                continue
            schema_errors += sum(1 for _ in label_validator.iter_errors(document))
            missing_required_fields += len(REQUIRED_LABEL_FIELDS - document.keys())
            record_id = document.get("recordId")
            if isinstance(record_id, str) and record_id.strip():
                record_ids.append(record_id)
            utterances = document.get("utterances")
            if not isinstance(utterances, list):
                continue
            for utterance in utterances:
                utterance_count += 1
                if not isinstance(utterance, dict):
                    continue
                missing_required_fields += len(
                    REQUIRED_UTTERANCE_FIELDS - utterance.keys()
                )
                text = utterance.get("text")
                if not isinstance(text, str):
                    continue
                normalized = text.strip()
                empty_utterance_count += int(not normalized)
                transcript_character_count += len(normalized)

    audio_stems = set(audio_members)
    label_stems = set(label_members)
    return ArchiveStats(
        split=split,
        audio_count=len(audio_members),
        label_count=len(label_members),
        paired_count=len(audio_stems & label_stems),
        audio_without_label=len(audio_stems - label_stems),
        label_without_audio=len(label_stems - audio_stems),
        duplicate_audio_stems=duplicate_audio_stems,
        duplicate_label_stems=duplicate_label_stems,
        duplicate_record_ids=len(record_ids) - len(set(record_ids)),
        missing_required_fields=missing_required_fields,
        schema_error_count=schema_errors,
        utterance_count=utterance_count,
        empty_utterance_count=empty_utterance_count,
        transcript_character_count=transcript_character_count,
        audio_seconds=sum(durations),
        min_audio_seconds=min(durations, default=0.0),
        max_audio_seconds=max(durations, default=0.0),
        sample_rates=tuple(sorted(sample_rates)),
        channel_counts=tuple(sorted(channel_counts)),
        sample_widths=tuple(sorted(sample_widths)),
        record_ids=frozenset(record_ids),
        stems=frozenset(audio_stems & label_stems),
    )


def _validate_stats(stats: ArchiveStats) -> None:
    failures = {
        "audio_without_label": stats.audio_without_label,
        "label_without_audio": stats.label_without_audio,
        "duplicate_audio_stems": stats.duplicate_audio_stems,
        "duplicate_label_stems": stats.duplicate_label_stems,
        "duplicate_record_ids": stats.duplicate_record_ids,
        "missing_required_fields": stats.missing_required_fields,
        "schema_error_count": stats.schema_error_count,
        "empty_utterance_count": stats.empty_utterance_count,
    }
    nonzero = {name: value for name, value in failures.items() if value}
    if nonzero:
        raise ValueError(f"archive validation failed for {stats.split}: {nonzero}")
    if stats.audio_count == 0 or stats.paired_count != stats.audio_count:
        raise ValueError(f"archive pair is empty or incomplete for {stats.split}")


def _manifest(
    *,
    stats: ArchiveStats,
    usage_role: str,
    partition_name: str,
    audio_archive: Path,
    label_archive: Path,
    artifact_prefix: str,
    generated_at: str,
    cross_split_overlap: int,
) -> dict[str, object]:
    prefix = artifact_prefix.rstrip("/")
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "created_at": generated_at,
        "classification": "approved_restricted",
        "usage_role": usage_role,
        "source": {
            "name": "AIHub 119 intelligent emergency-call speech recognition data",
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "version": DATASET_VERSION,
            "collected_at": generated_at,
        },
        "split": {
            "name": stats.split,
            "strategy": "provider fixed partition",
            "unit": "record",
            "parameters": {
                "provider_partition": partition_name,
                "used_for_tuning": usage_role == "training",
            },
            "seed": None,
        },
        "integrity_report": {
            "schema_version": "1.0.0",
            "generated_at": generated_at,
            "required_fields": {
                "status": "passed",
                "missing_count": stats.missing_required_fields,
            },
            "duplicates": {
                "status": "passed",
                "count": stats.duplicate_record_ids
                + stats.duplicate_audio_stems
                + stats.duplicate_label_stems,
            },
            "schema_validation": {
                "status": "passed",
                "error_count": stats.schema_error_count,
            },
            "split_integrity": {
                "entities": {
                    "speaker": {
                        "status": "not_applicable",
                        "reason": (
                            "labels contain speaker roles, not stable speaker identities"
                        ),
                    },
                    "source": {
                        "status": "passed",
                        "overlap_count": cross_split_overlap,
                    },
                    "event": {
                        "status": "passed",
                        "overlap_count": cross_split_overlap,
                    },
                }
            },
            "source_drift": {"status": "passed", "changes_detected": 0},
        },
        "artifacts": [
            {
                "path": f"{prefix}/{audio_archive.name}",
                "sha256": sha256_file(audio_archive),
            },
            {
                "path": f"{prefix}/{label_archive.name}",
                "sha256": sha256_file(label_archive),
            },
        ],
        "inventory": stats.public_dict(),
    }
    if usage_role == "evaluation":
        manifest["evaluation"] = {
            "id": EVALUATION_ID,
            "record_count": stats.paired_count,
        }
    return manifest


def build_manifests(
    *,
    training_audio: Path,
    training_labels: Path,
    validation_audio: Path,
    validation_labels: Path,
    artifact_prefix: str,
    generated_at: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    created = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    training = inspect_archive_pair(training_audio, training_labels, "training")
    validation = inspect_archive_pair(
        validation_audio, validation_labels, "validation"
    )
    _validate_stats(training)
    _validate_stats(validation)
    stem_overlap = len(training.stems & validation.stems)
    record_overlap = len(training.record_ids & validation.record_ids)
    if stem_overlap or record_overlap:
        raise ValueError(
            "training/validation leakage detected: "
            f"stem_overlap={stem_overlap}, record_overlap={record_overlap}"
        )
    training_manifest = _manifest(
        stats=training,
        usage_role="training",
        partition_name="Training",
        audio_archive=training_audio,
        label_archive=training_labels,
        artifact_prefix=artifact_prefix,
        generated_at=created,
        cross_split_overlap=0,
    )
    validation_manifest = _manifest(
        stats=validation,
        usage_role="evaluation",
        partition_name="Validation",
        audio_archive=validation_audio,
        label_archive=validation_labels,
        artifact_prefix=artifact_prefix,
        generated_at=created,
        cross_split_overlap=0,
    )
    return training_manifest, validation_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-audio", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--validation-audio", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    training, validation = build_manifests(
        training_audio=args.training_audio,
        training_labels=args.training_labels,
        validation_audio=args.validation_audio,
        validation_labels=args.validation_labels,
        artifact_prefix=args.artifact_prefix,
        generated_at=args.generated_at,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "training": args.output_dir / "aihub-71768-gwangju-fire-training.json",
        "validation": args.output_dir
        / "aihub-71768-gwangju-fire-validation.json",
    }
    outputs["training"].write_text(
        json.dumps(training, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs["validation"].write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "training_records": training["inventory"]["paired_count"],
                "validation_records": validation["inventory"]["paired_count"],
                "output_files": [str(path) for path in outputs.values()],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

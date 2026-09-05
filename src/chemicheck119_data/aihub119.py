"""Inspect AIHub 119 archive pairs and emit provenance-only manifests.

The generated files contain counts, checksums, and validation outcomes. They do
not contain transcripts, addresses, audio, signed URLs, or download tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import io
from importlib import resources
import json
from pathlib import Path, PurePosixPath
import wave
import zipfile

from jsonschema import Draft202012Validator


DATASET_ID = "aihub_71768_gwangju_fire"
DATASET_VERSION = "dataset-71768_downloaded-2026-09-05"
SOURCE_URL = "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71768"
SOURCE_LICENSE = (
    "AIHub AI 허브 개방 데이터 이용정책, retrieved 2026-09-05; "
    "the publisher provides no version identifier"
)
SOURCE_LICENSE_URL = "https://www.aihub.or.kr/intrcn/guid/usagepolicy.do"
EVALUATION_ID = "speech_aihub119_gwangju_fire_validation_77"
EXPECTED_TRAINING_RECORDS = 659
EXPECTED_VALIDATION_RECORDS = 77
REQUIRED_LABEL_FIELDS = {
    "_id",
    "audioPath",
    "recordId",
    "utterances",
    "disasterLarge",
    "disasterMedium",
}
REQUIRED_UTTERANCE_FIELDS = {"id", "startAt", "endAt", "text", "speaker"}


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
    duplicate_audio_contents: int
    duplicate_transcript_contents: int
    label_audio_path_mismatches: int
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
    audio_content_digests: frozenset[str] = field(repr=False)
    transcript_content_digests: frozenset[str] = field(repr=False)

    def public_dict(self) -> dict[str, object]:
        result = asdict(self)
        for private_field in (
            "record_ids",
            "stems",
            "audio_content_digests",
            "transcript_content_digests",
        ):
            result.pop(private_field)
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
    schema_resource = resources.files("chemicheck119_data").joinpath(
        "schemas/aihub119-label.schema"
    )
    label_schema = json.loads(schema_resource.read_text(encoding="utf-8"))
    label_validator = Draft202012Validator(label_schema)
    record_ids: list[str] = []
    audio_content_digests: list[str] = []
    transcript_content_digests: list[str] = []
    label_audio_path_mismatches = 0
    missing_required_fields = 0
    schema_errors = 0
    utterance_count = 0
    empty_utterance_count = 0
    transcript_character_count = 0
    durations: list[float] = []
    audio_frame_bounds: dict[str, tuple[int, int]] = {}
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

        for stem, name in audio_members.items():
            try:
                audio_bytes = audio_zip.read(name)
                with wave.open(io.BytesIO(audio_bytes)) as audio:
                    rate = audio.getframerate()
                    channels = audio.getnchannels()
                    width = audio.getsampwidth()
                    frames = audio.getnframes()
                    compression = audio.getcomptype()
                    decoded_payload = audio.readframes(frames)
                    sample_rates.add(rate)
                    channel_counts.add(channels)
                    sample_widths.add(width)
                    durations.append(frames / rate if rate else 0.0)
                    expected_payload_bytes = frames * channels * width
                    if (
                        rate <= 0
                        or channels <= 0
                        or width <= 0
                        or frames <= 0
                        or compression != "NONE"
                        or len(decoded_payload) != expected_payload_bytes
                    ):
                        schema_errors += 1
                    else:
                        audio_frame_bounds[stem] = (rate, frames)
                        # The ASR model receives decoded samples, not RIFF headers.
                        # Include the signal format so equal bytes with a different
                        # interpretation are not treated as the same model input.
                        content_digest = hashlib.sha256()
                        content_digest.update(
                            f"pcm:{rate}:{channels}:{width}:{frames}:".encode("ascii")
                        )
                        content_digest.update(decoded_payload)
                        audio_content_digests.append(content_digest.hexdigest())
            except (EOFError, wave.Error, zipfile.BadZipFile):
                schema_errors += 1

        for stem, name in label_members.items():
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
            declared_audio_path = document.get("audioPath")
            declared_path = (
                PurePosixPath(declared_audio_path.replace("\\", "/"))
                if isinstance(declared_audio_path, str)
                else None
            )
            if (
                declared_path is None
                or declared_path.suffix.lower() != ".wav"
                or declared_path.stem != PurePosixPath(name).stem
            ):
                label_audio_path_mismatches += 1
            record_id = document.get("recordId")
            if isinstance(record_id, str) and record_id.strip():
                record_ids.append(record_id)
            utterances = document.get("utterances")
            if not isinstance(utterances, list):
                continue
            transcript_parts: list[str] = []
            for utterance in utterances:
                utterance_count += 1
                if not isinstance(utterance, dict):
                    continue
                missing_required_fields += len(
                    REQUIRED_UTTERANCE_FIELDS - utterance.keys()
                )
                start_at = utterance.get("startAt")
                end_at = utterance.get("endAt")
                bounds = audio_frame_bounds.get(stem)
                if bounds is not None:
                    timing_is_valid = (
                        type(start_at) is int
                        and type(end_at) is int
                        and start_at >= 0
                        and end_at > start_at
                        and end_at * bounds[0] <= bounds[1] * 1000
                    )
                    if not timing_is_valid:
                        schema_errors += 1
                text = utterance.get("text")
                if not isinstance(text, str):
                    continue
                normalized = text.strip()
                empty_utterance_count += int(not normalized)
                transcript_character_count += len(normalized)
                transcript_parts.append(normalized)
            if transcript_parts:
                transcript_content_digests.append(
                    hashlib.sha256("\n".join(transcript_parts).encode("utf-8")).hexdigest()
                )

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
        duplicate_audio_contents=(
            len(audio_content_digests) - len(set(audio_content_digests))
        ),
        duplicate_transcript_contents=(
            len(transcript_content_digests) - len(set(transcript_content_digests))
        ),
        label_audio_path_mismatches=label_audio_path_mismatches,
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
        audio_content_digests=frozenset(audio_content_digests),
        transcript_content_digests=frozenset(transcript_content_digests),
    )


def _validate_stats(stats: ArchiveStats) -> None:
    failures = {
        "audio_without_label": stats.audio_without_label,
        "label_without_audio": stats.label_without_audio,
        "duplicate_audio_stems": stats.duplicate_audio_stems,
        "duplicate_label_stems": stats.duplicate_label_stems,
        "duplicate_record_ids": stats.duplicate_record_ids,
        "duplicate_audio_contents": stats.duplicate_audio_contents,
        "duplicate_transcript_contents": stats.duplicate_transcript_contents,
        "missing_required_fields": stats.missing_required_fields,
        "schema_error_count": stats.schema_error_count,
        "empty_utterance_count": stats.empty_utterance_count,
    }
    nonzero = {name: value for name, value in failures.items() if value}
    if nonzero:
        raise ValueError(f"archive validation failed for {stats.split}: {nonzero}")
    if stats.audio_count == 0 or stats.paired_count != stats.audio_count:
        raise ValueError(f"archive pair is empty or incomplete for {stats.split}")
    declared_path_matches = stats.label_count - stats.label_audio_path_mismatches
    if declared_path_matches not in {0, stats.label_count}:
        raise ValueError(
            f"ambiguous provider audioPath mapping for {stats.split}: "
            f"matched={declared_path_matches}, labels={stats.label_count}"
        )


def _source_drift_report(
    artifacts: list[dict[str, str]],
    baseline_manifest: dict[str, object] | None,
) -> dict[str, object]:
    if baseline_manifest is None:
        return {
            "status": "not_applicable",
            "changes_detected": 0,
            "reason": (
                "first pinned snapshot; later acquisitions must compare "
                "archive SHA-256 values with this manifest"
            ),
        }
    baseline_artifacts = baseline_manifest.get("artifacts")
    if not isinstance(baseline_artifacts, list):
        raise ValueError("baseline manifest artifacts must be a list")
    baseline_by_name = {
        PurePosixPath(str(item.get("path"))).name: item.get("sha256")
        for item in baseline_artifacts
        if isinstance(item, dict)
    }
    current_by_name = {
        PurePosixPath(item["path"]).name: item["sha256"] for item in artifacts
    }
    changes = sum(
        baseline_by_name.get(name) != digest
        for name, digest in current_by_name.items()
    ) + sum(name not in current_by_name for name in baseline_by_name)
    if changes:
        raise ValueError(
            f"source drift detected against baseline manifest: changes={changes}"
        )
    return {"status": "passed", "changes_detected": 0}


def _manifest(
    *,
    stats: ArchiveStats,
    usage_role: str,
    partition_name: str,
    audio_archive: Path,
    label_archive: Path,
    artifact_prefix: str,
    generated_at: str,
    collected_at: str,
    source_overlap: int | None,
    event_overlap: int | None,
    baseline_manifest: dict[str, object] | None,
    dataset_id: str = DATASET_ID,
    dataset_version: str = DATASET_VERSION,
    evaluation_id: str = EVALUATION_ID,
) -> dict[str, object]:
    prefix = artifact_prefix.rstrip("/")
    artifacts = [
        {
            "path": f"{prefix}/{audio_archive.name}",
            "sha256": sha256_file(audio_archive),
        },
        {
            "path": f"{prefix}/{label_archive.name}",
            "sha256": sha256_file(label_archive),
        },
    ]
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at": generated_at,
        "classification": "approved_restricted",
        "usage_role": usage_role,
        "source": {
            "name": "AIHub 119 intelligent emergency-call speech recognition data",
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "license_url": SOURCE_LICENSE_URL,
            "license_summary": (
                "AI model training use with attribution; no unapproved third-party "
                "access or redistribution; overseas transfer requires separate agreement"
            ),
            "version": dataset_version,
            "collected_at": collected_at,
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
                + stats.duplicate_label_stems
                + stats.duplicate_audio_contents
                + stats.duplicate_transcript_contents,
            },
            "schema_validation": {
                "status": "passed",
                "error_count": stats.schema_error_count,
            },
            "pairing": {
                "status": "passed",
                "strategy": "audio and label archive member filename stem",
                "paired_count": stats.paired_count,
                "provider_audio_path_match_count": (
                    stats.label_count - stats.label_audio_path_mismatches
                ),
                "provider_audio_path_mismatch_count": (
                    stats.label_audio_path_mismatches
                ),
                "limitation": (
                    "the provider audioPath values do not name members in the "
                    "downloaded partition; archive member stems are the only "
                    "available pairing key"
                    if stats.label_audio_path_mismatches
                    else None
                ),
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
                        "status": (
                            "passed" if source_overlap is not None else "not_evaluated"
                        ),
                        "overlap_count": source_overlap,
                        "reason": (
                            None
                            if source_overlap is not None
                            else "training partition was not supplied for comparison"
                        ),
                    },
                    "event": {
                        "status": (
                            "passed" if event_overlap is not None else "not_evaluated"
                        ),
                        "overlap_count": event_overlap,
                        "reason": (
                            None
                            if event_overlap is not None
                            else "training partition was not supplied for comparison"
                        ),
                    },
                }
            },
            "source_drift": _source_drift_report(artifacts, baseline_manifest),
        },
        "artifacts": artifacts,
        "inventory": stats.public_dict(),
    }
    if usage_role == "evaluation":
        manifest["evaluation"] = {
            "id": evaluation_id,
            "record_count": stats.paired_count,
        }
    return manifest


def build_evaluation_manifest(
    *,
    validation_audio: Path,
    validation_labels: Path,
    artifact_prefix: str,
    collected_at: str,
    dataset_id: str,
    dataset_version: str,
    generated_at: str | None = None,
    expected_validation_records: int | None = None,
    baseline_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a validation-only manifest for cross-region external evaluation."""

    if not dataset_id.strip() or not dataset_version.strip():
        raise ValueError("dataset ID and version must be declared")
    created = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    validation = inspect_archive_pair(
        validation_audio, validation_labels, "validation"
    )
    _validate_stats(validation)
    if (
        expected_validation_records is not None
        and validation.paired_count != expected_validation_records
    ):
        raise ValueError(
            "fixed validation record count mismatch: "
            f"expected={expected_validation_records}, actual={validation.paired_count}"
        )
    evaluation_id = f"speech_{dataset_id}_validation_{validation.paired_count}"
    return _manifest(
        stats=validation,
        usage_role="evaluation",
        partition_name="Validation",
        audio_archive=validation_audio,
        label_archive=validation_labels,
        artifact_prefix=artifact_prefix,
        generated_at=created,
        collected_at=collected_at,
        source_overlap=None,
        event_overlap=None,
        baseline_manifest=baseline_manifest,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        evaluation_id=evaluation_id,
    )


def build_manifests(
    *,
    training_audio: Path,
    training_labels: Path,
    validation_audio: Path,
    validation_labels: Path,
    artifact_prefix: str,
    collected_at: str,
    generated_at: str | None = None,
    expected_training_records: int = EXPECTED_TRAINING_RECORDS,
    expected_validation_records: int = EXPECTED_VALIDATION_RECORDS,
    baseline_training_manifest: dict[str, object] | None = None,
    baseline_validation_manifest: dict[str, object] | None = None,
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
    if training.paired_count != expected_training_records:
        raise ValueError(
            "fixed training record count mismatch: "
            f"expected={expected_training_records}, actual={training.paired_count}"
        )
    if validation.paired_count != expected_validation_records:
        raise ValueError(
            "fixed validation record count mismatch: "
            f"expected={expected_validation_records}, actual={validation.paired_count}"
        )
    stem_overlap = len(training.stems & validation.stems)
    record_overlap = len(training.record_ids & validation.record_ids)
    audio_content_overlap = len(
        training.audio_content_digests & validation.audio_content_digests
    )
    transcript_content_overlap = len(
        training.transcript_content_digests & validation.transcript_content_digests
    )
    if stem_overlap or record_overlap or audio_content_overlap or transcript_content_overlap:
        raise ValueError(
            "training/validation leakage detected: "
            f"stem_overlap={stem_overlap}, record_overlap={record_overlap}, "
            f"audio_content_overlap={audio_content_overlap}, "
            f"transcript_content_overlap={transcript_content_overlap}"
        )
    training_manifest = _manifest(
        stats=training,
        usage_role="training",
        partition_name="Training",
        audio_archive=training_audio,
        label_archive=training_labels,
        artifact_prefix=artifact_prefix,
        generated_at=created,
        collected_at=collected_at,
        source_overlap=audio_content_overlap,
        event_overlap=max(stem_overlap, record_overlap, transcript_content_overlap),
        baseline_manifest=baseline_training_manifest,
    )
    validation_manifest = _manifest(
        stats=validation,
        usage_role="evaluation",
        partition_name="Validation",
        audio_archive=validation_audio,
        label_archive=validation_labels,
        artifact_prefix=artifact_prefix,
        generated_at=created,
        collected_at=collected_at,
        source_overlap=audio_content_overlap,
        event_overlap=max(stem_overlap, record_overlap, transcript_content_overlap),
        baseline_manifest=baseline_validation_manifest,
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
    parser.add_argument("--collected-at", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="directory containing the previous manifests for drift comparison",
    )
    args = parser.parse_args(argv)
    baseline_training = baseline_validation = None
    if args.baseline_dir:
        baseline_training = json.loads(
            (
                args.baseline_dir / "aihub-71768-gwangju-fire-training.json"
            ).read_text(encoding="utf-8")
        )
        baseline_validation = json.loads(
            (
                args.baseline_dir / "aihub-71768-gwangju-fire-validation.json"
            ).read_text(encoding="utf-8")
        )
    training, validation = build_manifests(
        training_audio=args.training_audio,
        training_labels=args.training_labels,
        validation_audio=args.validation_audio,
        validation_labels=args.validation_labels,
        artifact_prefix=args.artifact_prefix,
        collected_at=args.collected_at,
        generated_at=args.generated_at,
        baseline_training_manifest=baseline_training,
        baseline_validation_manifest=baseline_validation,
    )
    outputs = {
        "training": args.output_dir / "aihub-71768-gwangju-fire-training.json",
        "validation": args.output_dir
        / "aihub-71768-gwangju-fire-validation.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite append-only manifests: " + ", ".join(existing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

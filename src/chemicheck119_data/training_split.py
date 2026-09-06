"""Build a privacy-preserving, deterministic split manifest for ASR training.

The public manifest contains aggregate counts and membership digests only.  It
must never contain provider record IDs, transcripts, addresses, or archive
members from the approved-restricted AIHub labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
from importlib import resources
import json
from pathlib import Path, PurePosixPath
import struct
import unicodedata
import wave
import zipfile

from jsonschema import Draft202012Validator

from .aihub119 import DATASET_ID, DATASET_VERSION, sha256_file


IMPLEMENTATION_VERSION = "1.1.0"
SPLIT_PROTOCOL_ID = "whisper-lora-gwangju-train-dev-v1"
GENERATOR_SOURCE_PATH = "src/chemicheck119_data/training_split.py"
PRIORITY_TERMS_SOURCE_PATH = "config/speech_priority_terms_v1.txt"
DEFAULT_SEED = 119
DEFAULT_DEV_FRACTION = 0.2
MAX_LABEL_MEMBER_BYTES = 4 * 1024 * 1024
MAX_AUDIO_MEMBER_BYTES = 32 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


def _normalise_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _priority_terms(path: Path) -> tuple[str, ...]:
    terms = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    normalized = tuple(_normalise_search_text(term) for term in terms)
    if not terms or any(not term for term in normalized):
        raise ValueError("priority terms must contain non-empty searchable values")
    if len(normalized) != len(set(normalized)):
        raise ValueError("priority terms must be unique after normalization")
    return terms


def _source_contract(
    source_manifest_path: Path,
    audio_archive: Path,
    label_archive: Path,
) -> tuple[dict[str, object], bytes, list[dict[str, object]]]:
    source_bytes = source_manifest_path.read_bytes()
    try:
        source = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source manifest must be valid UTF-8 JSON") from error
    if not isinstance(source, dict):
        raise ValueError("source manifest root must be an object")
    if (
        source.get("dataset_id") != DATASET_ID
        or source.get("dataset_version") != DATASET_VERSION
        or source.get("classification") != "approved_restricted"
        or source.get("usage_role") != "training"
    ):
        raise ValueError("source manifest is not the pinned Gwangju Training snapshot")
    split = source.get("split")
    if not isinstance(split, dict) or split.get("name") != "training":
        raise ValueError("source manifest must describe the provider Training split")
    integrity = source.get("integrity_report")
    if not isinstance(integrity, dict):
        raise ValueError("source manifest integrity report is missing")
    for section, count_field in (
        ("required_fields", "missing_count"),
        ("duplicates", "count"),
        ("schema_validation", "error_count"),
    ):
        result = integrity.get(section)
        if (
            not isinstance(result, dict)
            or result.get("status") != "passed"
            or result.get(count_field) != 0
        ):
            raise ValueError(f"source manifest {section} gate did not pass")
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source manifest artifacts must be an array")
    matched_artifacts: list[dict[str, object]] = []
    for archive_path, label in (
        (audio_archive, "audio"),
        (label_archive, "label"),
    ):
        matches = [
            item
            for item in artifacts
            if isinstance(item, dict)
            and PurePosixPath(str(item.get("path") or "")).name == archive_path.name
        ]
        if len(matches) != 1 or matches[0].get("sha256") != sha256_file(archive_path):
            raise ValueError(
                f"{label} archive SHA-256 does not match the source manifest"
            )
        matched_artifacts.append(dict(matches[0]))
    return source, source_bytes, matched_artifacts


def _safe_members(
    archive: zipfile.ZipFile,
    *,
    suffix: str,
    maximum_bytes: int,
) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    stems: set[str] = set()
    for info in archive.infolist():
        if not info.filename.lower().endswith(suffix):
            continue
        if info.file_size <= 0 or info.file_size > maximum_bytes:
            raise ValueError("archive contains an unsafe member size")
        if info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
            raise ValueError("archive contains an unsafe compression ratio")
        stem = PurePosixPath(info.filename).stem
        if stem in stems:
            raise ValueError("archive contains a duplicate member stem")
        stems.add(stem)
        members[stem] = info
    if not members:
        raise ValueError(f"archive contains no {suffix} records")
    return members


def _read_bounded(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    maximum_bytes: int,
) -> bytes:
    with archive.open(info) as source:
        content = source.read(maximum_bytes + 1)
    if not content or len(content) > maximum_bytes:
        raise ValueError("archive member exceeded the bounded read")
    return content


def _audio_duration(content: bytes) -> float:
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
            payload = audio.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError("audio archive contains invalid WAV data") from error
    expected_payload_bytes = frame_count * channels * sample_width
    if (
        channels <= 0
        or sample_width <= 0
        or sample_rate <= 0
        or frame_count <= 0
        or compression != "NONE"
        or len(payload) != expected_payload_bytes
    ):
        raise ValueError("audio archive contains unsupported or truncated WAV data")
    return frame_count / sample_rate


def _membership_digest(record_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(record_ids):
        encoded = record_id.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _partition_summary(
    records: list[dict[str, object]],
    terms: tuple[str, ...],
) -> dict[str, object]:
    term_support: dict[str, dict[str, int]] = {}
    for term in terms:
        normalized_term = _normalise_search_text(term)
        term_support[term] = {
            "record_support": sum(
                any(
                    normalized_term in utterance
                    for utterance in record["normalized_utterances"]
                )
                for record in records
            ),
            "utterance_support": sum(
                normalized_term in utterance
                for record in records
                for utterance in record["normalized_utterances"]
            ),
        }
    audio_seconds = sum(float(record["audio_seconds"]) for record in records)
    return {
        "record_count": len(records),
        "utterance_count": sum(
            len(record["normalized_utterances"]) for record in records
        ),
        "audio_seconds": round(audio_seconds, 3),
        "audio_hours": round(audio_seconds / 3600, 4),
        "membership_sha256": _membership_digest(
            [str(record["record_id"]) for record in records]
        ),
        "priority_term_support": term_support,
    }


def build_training_split_manifest(
    *,
    audio_archive: Path,
    label_archive: Path,
    source_manifest_path: Path,
    priority_terms_path: Path,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Return an aggregate-only train/dev split manifest."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not 0.05 <= dev_fraction <= 0.5:
        raise ValueError("dev fraction must be between 0.05 and 0.5")
    if priority_terms_path.name != PurePosixPath(PRIORITY_TERMS_SOURCE_PATH).name:
        raise ValueError("priority terms must use the versioned repository artifact")
    source_manifest, source_manifest_bytes, source_artifacts = _source_contract(
        source_manifest_path, audio_archive, label_archive
    )
    terms = _priority_terms(priority_terms_path)
    schema_resource = resources.files("chemicheck119_data").joinpath(
        "schemas/aihub119-label.schema"
    )
    label_schema = json.loads(schema_resource.read_text(encoding="utf-8"))
    validator = Draft202012Validator(label_schema)
    records: list[dict[str, object]] = []
    record_ids: set[str] = set()
    transcript_digests: set[str] = set()

    with zipfile.ZipFile(audio_archive) as audio_zip, zipfile.ZipFile(
        label_archive
    ) as label_zip:
        audio_members = _safe_members(
            audio_zip,
            suffix=".wav",
            maximum_bytes=MAX_AUDIO_MEMBER_BYTES,
        )
        label_members = _safe_members(
            label_zip,
            suffix=".json",
            maximum_bytes=MAX_LABEL_MEMBER_BYTES,
        )
        if set(audio_members) != set(label_members):
            raise ValueError("audio and label archive member stems do not pair exactly")
        for stem in sorted(label_members):
            info = label_members[stem]
            try:
                document = json.loads(
                    _read_bounded(
                        label_zip,
                        info,
                        MAX_LABEL_MEMBER_BYTES,
                    ).decode("utf-8-sig")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("label archive contains invalid JSON") from error
            errors = list(validator.iter_errors(document))
            if errors:
                raise ValueError("label archive schema validation failed")
            record_id = str(document["recordId"])
            if record_id in record_ids:
                raise ValueError("label archive contains duplicate record IDs")
            record_ids.add(record_id)
            utterances = document["utterances"]
            texts = [str(item["text"]).strip() for item in utterances]
            if any(not text for text in texts):
                raise ValueError("label archive contains an empty utterance")
            transcript_digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
            if transcript_digest in transcript_digests:
                raise ValueError("label archive contains duplicate transcript content")
            transcript_digests.add(transcript_digest)
            audio_seconds = _audio_duration(
                _read_bounded(
                    audio_zip,
                    audio_members[stem],
                    MAX_AUDIO_MEMBER_BYTES,
                )
            )
            normalized_utterances = [
                _normalise_search_text(text) for text in texts
            ]
            records.append(
                {
                    "record_id": record_id,
                    "normalized_utterances": normalized_utterances,
                    "audio_seconds": audio_seconds,
                    "order_digest": hashlib.sha256(
                        f"{DATASET_ID}:{seed}:{record_id}".encode("utf-8")
                    ).digest(),
                }
            )

    inventory = source_manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("source manifest inventory is missing")
    expected_records = inventory.get("paired_count")
    if expected_records != len(records):
        raise ValueError("label record count does not match the source manifest")
    observed_seconds = round(
        sum(float(record["audio_seconds"]) for record in records), 3
    )
    if inventory.get("audio_seconds") != observed_seconds:
        raise ValueError("label duration does not match the source manifest")

    ordered = sorted(
        records,
        key=lambda record: (record["order_digest"], str(record["record_id"])),
    )
    dev_count = int(len(ordered) * dev_fraction + 0.5)
    if dev_count <= 0 or dev_count >= len(ordered):
        raise ValueError("split would produce an empty partition")
    dev_records = ordered[:dev_count]
    train_records = ordered[dev_count:]
    train_ids = {str(record["record_id"]) for record in train_records}
    dev_ids = {str(record["record_id"]) for record in dev_records}
    overlap_count = len(train_ids & dev_ids)
    if overlap_count:
        raise ValueError("train/dev record overlap detected")

    created = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "created_at": created,
        "classification": "approved_restricted",
        "usage_role": "training",
        "source": dict(source_manifest["source"]),
        "split": {
            "name": "training-internal-train-dev",
            "strategy": "deterministic SHA-256 rank partition",
            "unit": "recordId group",
            "parameters": {
                "protocol_id": SPLIT_PROTOCOL_ID,
                "implementation_version": IMPLEMENTATION_VERSION,
                "group_key": "recordId",
                "dev_fraction": dev_fraction,
                "dev_rounding": "nearest integer, half up",
                "dev_assignment": "lowest SHA-256(dataset_id:seed:recordId) ranks",
                "clean_and_derived_share_partition": True,
                "used_for_tuning": True,
            },
            "seed": seed,
        },
        "integrity_report": {
            "schema_version": "1.0.0",
            "generated_at": created,
            "required_fields": {"status": "passed", "missing_count": 0},
            "duplicates": {"status": "passed", "count": 0},
            "schema_validation": {"status": "passed", "error_count": 0},
            "split_integrity": {
                "entities": {
                    "speaker": {
                        "status": "not_applicable",
                        "reason": "labels contain roles but no stable speaker identity",
                    },
                    "source": {"status": "passed", "overlap_count": overlap_count},
                    "event": {
                        "status": "not_evaluated",
                        "reason": "provider labels contain no stable cross-record incident ID",
                    },
                }
            },
            "source_drift": {"status": "passed", "changes_detected": 0},
        },
        "artifacts": [
            *source_artifacts,
            {
                "path": f"data/manifests/{source_manifest_path.name}",
                "sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
            },
            {
                "path": PRIORITY_TERMS_SOURCE_PATH,
                "sha256": sha256_file(priority_terms_path),
            },
        ],
        "provenance": {
            "generator_source_path": GENERATOR_SOURCE_PATH,
            "generator_source_sha256": sha256_file(Path(__file__)),
            "priority_terms_source_path": PRIORITY_TERMS_SOURCE_PATH,
            "priority_terms_sha256": sha256_file(priority_terms_path),
            "contains_record_ids": False,
            "contains_transcripts": False,
            "contains_addresses": False,
            "membership_digest_encoding": "SHA-256 of sorted uint32-length-prefixed UTF-8 record IDs",
        },
        "inventory": {
            "record_count": len(records),
            "audio_seconds": observed_seconds,
            "audio_hours": round(observed_seconds / 3600, 4),
            "train": _partition_summary(train_records, terms),
            "dev": _partition_summary(dev_records, terms),
        },
        "evidence_scope": (
            "development split for AIHub Gwangju emergency-call Training only; "
            "not a locked test and not field-radio validation"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--label-archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite append-only manifest: {args.output}")
    manifest = build_training_split_manifest(
        audio_archive=args.audio_archive,
        label_archive=args.label_archive,
        source_manifest_path=args.source_manifest,
        priority_terms_path=args.priority_terms,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train_records": manifest["inventory"]["train"]["record_count"],
                "dev_records": manifest["inventory"]["dev"]["record_count"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

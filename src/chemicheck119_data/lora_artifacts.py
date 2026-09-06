"""Materialize private clean/wind archives for the bounded Whisper LoRA run.

Outputs contain approved-restricted audio and labels and therefore belong only
in private storage.  Console output and the run summary remain aggregate-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import platform
import tempfile
import zipfile
import zlib

from . import radio_simulation, training_split
from .aihub119 import DATASET_ID, DATASET_VERSION, sha256_file
from .radio_simulation import (
    MAX_AUDIO_MEMBER_BYTES,
    MAX_LABEL_MEMBER_BYTES,
    _read_member,
    _record_seed,
    _safe_members,
    _sha256_bytes,
    _zip_write,
    apply_variant,
    profile_variants,
)
from .training_split import (
    SPLIT_PROTOCOL_ID,
    _membership_digest,
    build_training_split_manifest,
)


IMPLEMENTATION_VERSION = "1.0.0"
ARTIFACT_PROTOCOL_ID = "whisper-lora-clean-wind-snr0-v1"
CONDITIONS = ("clean", "wind_snr0")
PARTITIONS = ("train", "dev")


def _implementation_provenance() -> dict[str, object]:
    sources = (
        (
            "src/chemicheck119_data/lora_artifacts.py",
            Path(__file__),
        ),
        (
            "src/chemicheck119_data/radio_simulation.py",
            Path(radio_simulation.__file__),
        ),
        (
            "src/chemicheck119_data/training_split.py",
            Path(training_split.__file__),
        ),
    )
    return {
        "sources": [
            {"path": repository_path, "sha256": sha256_file(runtime_path)}
            for repository_path, runtime_path in sources
        ],
        "dependencies": {
            "python": platform.python_version(),
            "zlib_compile": zlib.ZLIB_VERSION,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "numpy": distribution_version("numpy"),
            "scipy": distribution_version("scipy"),
        },
    }


def _json_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    content = path.read_bytes()
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, content


def _validate_split_snapshot(
    *,
    audio_archive: Path,
    label_archive: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    priority_terms_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    source, source_bytes = _json_object(source_manifest_path, "source manifest")
    split, split_bytes = _json_object(split_manifest_path, "split manifest")
    split_spec = split.get("split")
    if not isinstance(split_spec, dict):
        raise ValueError("split manifest split must be an object")
    parameters = split_spec.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("split manifest parameters must be an object")
    if (
        parameters.get("protocol_id") != SPLIT_PROTOCOL_ID
        or parameters.get("clean_and_derived_share_partition") is not True
        or split.get("dataset_id") != DATASET_ID
        or split.get("dataset_version") != DATASET_VERSION
    ):
        raise ValueError("unsupported training split contract")
    seed = split_spec.get("seed")
    dev_fraction = parameters.get("dev_fraction")
    generated_at = split.get("created_at")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(dev_fraction, (int, float))
        or isinstance(dev_fraction, bool)
        or not isinstance(generated_at, str)
    ):
        raise ValueError("split seed, fraction, or timestamp is invalid")
    rebuilt = build_training_split_manifest(
        audio_archive=audio_archive,
        label_archive=label_archive,
        source_manifest_path=source_manifest_path,
        priority_terms_path=priority_terms_path,
        seed=seed,
        dev_fraction=float(dev_fraction),
        generated_at=generated_at,
    )
    if rebuilt != split:
        raise ValueError("split manifest cannot be reproduced from the pinned inputs")
    return (
        source,
        split,
        hashlib.sha256(source_bytes).hexdigest(),
        hashlib.sha256(split_bytes).hexdigest(),
    )


def _partition_members(
    label_zip: zipfile.ZipFile,
    label_members: dict[str, str],
    split_manifest: dict[str, object],
) -> tuple[dict[str, str], dict[str, bytes], dict[str, dict[str, object]]]:
    records: list[tuple[bytes, str, str]] = []
    labels: dict[str, bytes] = {}
    documents: dict[str, dict[str, object]] = {}
    record_ids: set[str] = set()
    split_spec = split_manifest["split"]
    seed = int(split_spec["seed"])
    for stem in sorted(label_members):
        content = _read_member(
            label_zip, label_members[stem], MAX_LABEL_MEMBER_BYTES
        )
        document = json.loads(content.decode("utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError("label record must be an object")
        record_id = document.get("recordId")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("label record ID is missing")
        if record_id in record_ids:
            raise ValueError("duplicate record ID")
        record_ids.add(record_id)
        order_digest = hashlib.sha256(
            f"{DATASET_ID}:{seed}:{record_id}".encode("utf-8")
        ).digest()
        records.append((order_digest, record_id, stem))
        labels[stem] = content
        documents[stem] = document

    records.sort(key=lambda item: (item[0], item[1]))
    parameters = split_spec["parameters"]
    dev_fraction = float(parameters["dev_fraction"])
    dev_count = int(len(records) * dev_fraction + 0.5)
    dev = records[:dev_count]
    train = records[dev_count:]
    partition_by_stem = {
        **{stem: "dev" for _, _, stem in dev},
        **{stem: "train" for _, _, stem in train},
    }
    inventory = split_manifest["inventory"]
    for partition, selected in (("train", train), ("dev", dev)):
        declared = inventory[partition]
        record_ids_for_partition = [record_id for _, record_id, _ in selected]
        if (
            len(selected) != declared["record_count"]
            or _membership_digest(record_ids_for_partition)
            != declared["membership_sha256"]
        ):
            raise ValueError(f"{partition} membership digest does not match")
    return partition_by_stem, labels, documents


def _manifest(
    *,
    partition: str,
    condition: str,
    source_manifest: dict[str, object],
    source_manifest_sha256: str,
    split_manifest: dict[str, object],
    split_manifest_sha256: str,
    priority_terms_sha256: str,
    generated_at: str,
    artifact_prefix: str,
    audio_path: Path,
    label_path: Path,
    ledger_path: Path,
    audio_seconds: float,
    utterance_count: int,
    variant: dict[str, object],
    implementation_provenance: dict[str, object],
) -> dict[str, object]:
    prefix = artifact_prefix.rstrip("/")
    usage_role = "training" if partition == "train" else "development"
    declared_partition = split_manifest["inventory"][partition]
    not_applicable = {
        "status": "not_applicable",
        "reason": "provider labels contain no stable speaker identity",
    }
    return {
        "schema_version": "1.0.0",
        "dataset_id": f"{DATASET_ID}_lora_{partition}_{condition}",
        "dataset_version": f"{DATASET_VERSION}+{ARTIFACT_PROTOCOL_ID}",
        "created_at": generated_at,
        "classification": "derived",
        "usage_role": usage_role,
        "source": {
            **source_manifest["source"],
            "parent_manifest_sha256": source_manifest_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        },
        "split": {
            "name": f"Training internal {partition}",
            "strategy": "inherit deterministic recordId group membership from split manifest",
            "unit": "recordId group",
            "parameters": {
                "protocol_id": ARTIFACT_PROTOCOL_ID,
                "parent_split_protocol_id": SPLIT_PROTOCOL_ID,
                "partition": partition,
                "condition": condition,
                "membership_sha256": declared_partition["membership_sha256"],
                "clean_and_derived_share_partition": True,
                "used_for_tuning": True,
            },
            "seed": split_manifest["split"]["seed"],
        },
        "preprocessing": {
            "implementation": "chemicheck119_data.lora_artifacts",
            "version": IMPLEMENTATION_VERSION,
            "parameters": {
                "variant": variant,
                "source_manifest_sha256": source_manifest_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "priority_terms_sha256": priority_terms_sha256,
                "implementation_provenance": implementation_provenance,
                "per_record_seed_derivation": (
                    "sha256(base_seed:source_audio_sha256:variant_id)[:8] mod 2^32"
                ),
            },
            "seed": split_manifest["split"]["seed"],
        },
        "integrity_report": {
            "schema_version": "1.0.0",
            "generated_at": generated_at,
            "required_fields": {"status": "passed", "missing_count": 0},
            "duplicates": {"status": "passed", "count": 0},
            "schema_validation": {"status": "passed", "error_count": 0},
            "pairing": {
                "status": "passed",
                "strategy": "same source member stem in partition audio and label archives",
                "paired_count": declared_partition["record_count"],
            },
            "reference_timing": {
                "status": "not_evaluated",
                "reason": "text references are preserved; derived timing was not independently relabeled",
            },
            "split_integrity": {
                "entities": {
                    "speaker": not_applicable,
                    "source": {"status": "passed", "overlap_count": 0},
                    "event": {
                        "status": "not_evaluated",
                        "reason": "provider labels contain no stable cross-record incident ID",
                    },
                }
            },
            "source_drift": {
                "status": "not_applicable",
                "changes_detected": 0,
                "reason": "derived artifacts are bound to immutable parent hashes",
            },
        },
        "artifacts": [
            {
                "path": f"{prefix}/{audio_path.name}",
                "sha256": sha256_file(audio_path),
                "bytes": audio_path.stat().st_size,
                "access": "private",
            },
            {
                "path": f"{prefix}/{label_path.name}",
                "sha256": sha256_file(label_path),
                "bytes": label_path.stat().st_size,
                "access": "private",
            },
            {
                "path": f"{prefix}/{ledger_path.name}",
                "sha256": sha256_file(ledger_path),
                "bytes": ledger_path.stat().st_size,
                "access": "private",
            },
        ],
        "inventory": {
            "paired_count": declared_partition["record_count"],
            "utterance_count": utterance_count,
            "audio_seconds": round(audio_seconds, 6),
            "audio_hours": round(audio_seconds / 3600, 6),
        },
        "evidence_scope": (
            "AIHub emergency-call Training derivative with procedural wind; "
            "not field-radio validation"
        ),
        "limitations": [
            "procedural wind is not recorded fireground noise",
            "SNR uses whole-record RMS rather than calibrated active-speech level",
            "development partition is used for model selection and is not a locked test",
        ],
    }


def build_lora_artifacts(
    *,
    audio_archive: Path,
    label_archive: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    priority_terms_path: Path,
    output_dir: Path,
    artifact_prefix: str,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Create deterministic private artifacts and an aggregate run summary."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not artifact_prefix.strip().startswith("gs://"):
        raise ValueError("artifact prefix must be a private GCS URI")
    source_manifest, split_manifest, source_sha256, split_sha256 = (
        _validate_split_snapshot(
            audio_archive=audio_archive,
            label_archive=label_archive,
            source_manifest_path=source_manifest_path,
            split_manifest_path=split_manifest_path,
            priority_terms_path=priority_terms_path,
        )
    )
    created = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    terms_sha256 = sha256_file(priority_terms_path)
    implementation_provenance = _implementation_provenance()
    variants = {spec.id: spec for spec in profile_variants() if spec.id in CONDITIONS}
    if set(variants) != set(CONDITIONS):
        raise ValueError("radio-sim profile does not contain the pinned conditions")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        ledgers: list[dict[str, object]] = []
        audio_seconds = {
            partition: {condition: 0.0 for condition in CONDITIONS}
            for partition in PARTITIONS
        }
        utterance_counts = {partition: 0 for partition in PARTITIONS}
        audio_outputs = {
            (partition, condition): temporary / f"{partition}-{condition}.zip"
            for partition in PARTITIONS
            for condition in CONDITIONS
        }
        label_outputs = {
            partition: temporary / f"{partition}-labels.zip"
            for partition in PARTITIONS
        }
        ledger_path = temporary / "provenance.private.jsonl"

        with zipfile.ZipFile(audio_archive) as audio_zip, zipfile.ZipFile(
            label_archive
        ) as label_zip:
            audio_members = _safe_members(audio_zip, ".wav")
            label_members = _safe_members(label_zip, ".json")
            if set(audio_members) != set(label_members):
                raise ValueError("source audio and label archive stems do not match")
            partition_by_stem, labels, documents = _partition_members(
                label_zip, label_members, split_manifest
            )
            with (
                zipfile.ZipFile(label_outputs["train"], "w") as train_labels,
                zipfile.ZipFile(label_outputs["dev"], "w") as dev_labels,
            ):
                label_destinations = {"train": train_labels, "dev": dev_labels}
                for stem in sorted(labels):
                    partition = partition_by_stem[stem]
                    _zip_write(label_destinations[partition], f"{stem}.json", labels[stem])
                    utterances = documents[stem].get("utterances")
                    if not isinstance(utterances, list):
                        raise ValueError("label utterances must be an array")
                    utterance_counts[partition] += len(utterances)

            destinations = {
                key: zipfile.ZipFile(path, "w") for key, path in audio_outputs.items()
            }
            try:
                for stem in sorted(audio_members):
                    partition = partition_by_stem[stem]
                    source_content = _read_member(
                        audio_zip,
                        audio_members[stem],
                        MAX_AUDIO_MEMBER_BYTES,
                    )
                    source_digest = _sha256_bytes(source_content)
                    record_id = str(documents[stem]["recordId"])
                    record_key = _sha256_bytes(record_id.encode("utf-8"))[:16]
                    for condition in CONDITIONS:
                        spec = variants[condition]
                        record_seed = _record_seed(
                            int(split_manifest["split"]["seed"]),
                            source_digest,
                            condition,
                        )
                        transformed, signal = apply_variant(
                            source_content, spec, seed=record_seed
                        )
                        _zip_write(
                            destinations[(partition, condition)],
                            f"{stem}.wav",
                            transformed,
                        )
                        audio_seconds[partition][condition] += float(
                            signal["output_seconds"]
                        )
                        ledgers.append(
                            {
                                "record_key": record_key,
                                "partition": partition,
                                "condition": condition,
                                "source_audio_sha256": source_digest,
                                "derived_audio_sha256": _sha256_bytes(transformed),
                                "seed": record_seed,
                                "variant": asdict(spec),
                                "signal": signal,
                            }
                        )
            finally:
                for destination in destinations.values():
                    destination.close()

        with ledger_path.open("x", encoding="utf-8") as destination:
            for row in ledgers:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger_path.chmod(0o600)
        for path in [*audio_outputs.values(), *label_outputs.values()]:
            path.chmod(0o600)

        manifests: list[dict[str, object]] = []
        for partition in PARTITIONS:
            for condition in CONDITIONS:
                manifest = _manifest(
                    partition=partition,
                    condition=condition,
                    source_manifest=source_manifest,
                    source_manifest_sha256=source_sha256,
                    split_manifest=split_manifest,
                    split_manifest_sha256=split_sha256,
                    priority_terms_sha256=terms_sha256,
                    generated_at=created,
                    artifact_prefix=artifact_prefix,
                    audio_path=audio_outputs[(partition, condition)],
                    label_path=label_outputs[partition],
                    ledger_path=ledger_path,
                    audio_seconds=audio_seconds[partition][condition],
                    utterance_count=utterance_counts[partition],
                    variant=asdict(variants[condition]),
                    implementation_provenance=implementation_provenance,
                )
                manifest_path = temporary / f"{partition}-{condition}.manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                manifest_path.chmod(0o600)
                manifests.append(
                    {
                        "partition": partition,
                        "condition": condition,
                        "manifest": manifest_path.name,
                        "manifest_sha256": sha256_file(manifest_path),
                        "audio_sha256": sha256_file(audio_outputs[(partition, condition)]),
                        "labels_sha256": sha256_file(label_outputs[partition]),
                        "record_count": manifest["inventory"]["paired_count"],
                        "utterance_count": manifest["inventory"]["utterance_count"],
                        "audio_hours": manifest["inventory"]["audio_hours"],
                    }
                )

        run_summary = {
            "schema_version": "1.0.0",
            "protocol_id": ARTIFACT_PROTOCOL_ID,
            "status": "completed",
            "fact_status": "구현 완료",
            "created_at": created,
            "source_manifest_sha256": source_sha256,
            "split_manifest_sha256": split_sha256,
            "priority_terms_sha256": terms_sha256,
            "implementation_provenance": implementation_provenance,
            "private_ledger_sha256": sha256_file(ledger_path),
            "manifests": manifests,
            "privacy": {
                "git_commit_allowed": False,
                "private_storage_required": True,
                "console_contains_record_ids_or_transcripts": False,
            },
            "automatic_training_allowed": False,
            "claim_scope": (
                "immutable data artifacts only; no LoRA performance or field-radio claim"
            ),
        }
        summary_path = temporary / "run-summary.json"
        summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_path.chmod(0o600)
        temporary.rename(output_dir)
    return run_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--label-archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    summary = build_lora_artifacts(
        audio_archive=args.audio_archive,
        label_archive=args.label_archive,
        source_manifest_path=args.source_manifest,
        split_manifest_path=args.split_manifest,
        priority_terms_path=args.priority_terms,
        output_dir=args.output_dir,
        artifact_prefix=args.artifact_prefix,
        generated_at=args.generated_at,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "manifest_count": len(summary["manifests"]),
                "output": str(args.output_dir / "run-summary.json"),
                "automatic_training_allowed": summary["automatic_training_allowed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

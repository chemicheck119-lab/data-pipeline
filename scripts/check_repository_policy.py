"""Reject large artifacts, restricted data, and common secrets from Git history."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for


MAX_TRACKED_BYTES = 10 * 1024 * 1024
FORBIDDEN_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".ggml",
    ".gguf",
    ".h5",
    ".hdf5",
    ".keras",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}
FORBIDDEN_AUDIO_SUFFIXES = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
FORBIDDEN_DOCUMENT_ARCHIVE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".doc",
    ".docx",
    ".gz",
    ".odf",
    ".odp",
    ".ods",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rar",
    ".rtf",
    ".tar",
    ".tgz",
    ".xls",
    ".xlsx",
    ".xz",
    ".zip",
}
DATASET_CONTENT_SUFFIXES = {
    ".avro",
    ".arrow",
    ".csv",
    ".db",
    ".feather",
    ".ipc",
    ".json",
    ".jsonl",
    ".joblib",
    ".mat",
    ".ndjson",
    ".npy",
    ".npz",
    ".orc",
    ".parquet",
    ".pickle",
    ".pkl",
    ".rdata",
    ".rds",
    ".sav",
    ".sql",
    ".sqlite",
    ".tsv",
    ".xml",
}
PLAIN_TEXT_CORPUS_SUFFIXES = {".ctm", ".lab", ".stm", ".text", ".txt"}
CORPUS_DIRECTORY_NAMES = {
    "corpora",
    "corpus",
    "dataset",
    "datasets",
    "transcript",
    "transcripts",
}
FORBIDDEN_CREDENTIAL_NAMES = {
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "service-account.json",
}
FORBIDDEN_CREDENTIAL_SUFFIXES = {
    ".jks",
    ".kdbx",
    ".key",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
}
SECRET_CONTENT_PATTERNS = (
    (
        "private key",
        re.compile(
            br"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
        ),
    ),
    ("AWS access key", re.compile(br"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(br"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("GitLab token", re.compile(br"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("OpenAI API key", re.compile(br"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "signed download URL",
        re.compile(
            br"https?://[^\s\"'<>]{1,4096}[?&]"
            br"(?:X-Amz-Signature|X-Goog-Signature|Signature|sig)="
            br"[A-Za-z0-9%+/=_-]{16,}",
            re.IGNORECASE,
        ),
    ),
)
PERSONAL_DATA_PATTERNS = (
    (
        "email address",
        re.compile(br"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "Korean phone number",
        re.compile(br"\b(?:\+?82[- ]?)?0?1[016789][- ]?[0-9]{3,4}[- ]?[0-9]{4}\b"),
    ),
    (
        "Korean resident registration number",
        re.compile(br"\b[0-9]{6}-[1-8][0-9]{6}\b"),
    ),
)
FORBIDDEN_DATA_ROOTS = {
    Path("data/raw"),
    Path("data/interim"),
    Path("data/processed"),
}
FORBIDDEN_OUTPUT_ROOTS = FORBIDDEN_DATA_ROOTS | {
    Path("artifacts"),
    Path("models"),
    Path("outputs"),
}
FORBIDDEN_OUTPUT_DIRECTORY_NAMES = {"artifacts", "models", "outputs"}
FORBIDDEN_MODEL_DIRECTORY_NAMES = {"checkpoints", "models"}
ALLOWED_PLACEHOLDERS = {
    root / "README.md" for root in FORBIDDEN_OUTPUT_ROOTS
}
MANIFEST_ROOT = Path("data/manifests")
SCHEMA_ROOT = Path("schemas")
MANIFEST_CLASSIFICATIONS = {"approved_restricted", "derived", "public", "synthetic"}
MANIFEST_USAGE_ROLES = {"development", "evaluation", "fixture", "reference", "training"}
NAMED_EVALUATION_RECORD_COUNTS = {
    "parser_national_external_442": 442,
    "resolver_ulsan_locked_419": 419,
    "speech_aihub119_gwangju_fire_validation_77": 77,
}
CROSS_REGION_SPEECH_EVALUATION_PATTERN = re.compile(
    r"^speech_(aihub_71768_(?:seoul|incheon)_fire)_validation_([1-9][0-9]*)$"
)
RADIO_SIMULATION_EVALUATION_PATTERN = re.compile(
    r"^speech_(aihub_71768_(?:gwangju|seoul|incheon)_fire_"
    r"radio_sim_v1_[a-z0-9_]+)_([1-9][0-9]*)$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
FIXTURE_METADATA_SUFFIX = ".fixture.json"
FIXTURE_CLASSIFICATIONS = {"public_redistributable", "synthetic"}


@dataclass(frozen=True)
class TrackedObject:
    path: Path
    size: int
    object_id: Optional[str] = None
    object_type: str = "blob"
    object_mode: str = "100644"
    revision: str = "manual"


def current_tree_objects(repository: Path = Path(".")) -> list[TrackedObject]:
    """Read staged paths and blob sizes from the Git index."""
    result = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        check=True,
        capture_output=True,
        cwd=repository,
    )
    objects: list[TrackedObject] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3 or fields[2] != b"0":
            continue
        mode = fields[0]
        object_id = fields[1].decode("ascii")
        object_type = "commit" if mode == b"160000" else "blob"
        if object_type == "blob":
            size = int(
                subprocess.run(
                    ["git", "cat-file", "-s", object_id],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=repository,
                ).stdout.strip()
            )
        else:
            size = 0
        path = raw_path.decode("utf-8", errors="surrogateescape")
        objects.append(
            TrackedObject(
                path=Path(path),
                size=size,
                object_id=object_id,
                object_type=object_type,
                object_mode=mode.decode("ascii"),
                revision="INDEX",
            )
        )
    return objects


def revision_objects(
    revision: str, repository: Path = Path(".")
) -> list[TrackedObject]:
    """Return every file path present in every commit selected by ``revision``.

    ``git rev-list --objects`` cannot be used as a path inventory because its path
    field is only an indeterminate hint for an object.  Listing each commit tree
    preserves every path, including two paths that point at the same blob and a
    restricted file that is deleted by a later commit in the pull request.
    """
    commit_result = subprocess.run(
        ["git", "rev-list", "--reverse", revision],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository,
    )
    objects: list[TrackedObject] = []
    seen: set[tuple[str, str, str]] = set()
    for commit in commit_result.stdout.splitlines():
        tree_result = subprocess.run(
            ["git", "ls-tree", "-rlz", "--full-tree", commit],
            check=True,
            capture_output=True,
            cwd=repository,
        )
        for record in tree_result.stdout.split(b"\0"):
            if not record:
                continue
            metadata, separator, raw_path = record.partition(b"\t")
            if not separator:
                continue
            fields = metadata.split()
            if len(fields) != 4 or fields[1] == b"tree":
                continue
            object_id = fields[2].decode("ascii")
            path = raw_path.decode("utf-8", errors="surrogateescape")
            key = (commit, object_id, path)
            if key in seen:
                continue
            seen.add(key)
            object_type = fields[1].decode("ascii")
            size = int(fields[3]) if fields[3] != b"-" else 0
            objects.append(
                TrackedObject(
                    path=Path(path),
                    size=size,
                    object_id=object_id,
                    object_type=object_type,
                    object_mode=fields[0].decode("ascii"),
                    revision=commit,
                )
            )
    return objects


def commit_range_objects(
    base_ref: str, repository: Path = Path(".")
) -> list[TrackedObject]:
    return revision_objects(f"{base_ref}..HEAD", repository)


def reachable_commit_objects(repository: Path = Path(".")) -> list[TrackedObject]:
    return revision_objects("HEAD", repository)


def is_forbidden_output(path: Path) -> bool:
    below_data_root = any(
        root == path or root in path.parents for root in FORBIDDEN_DATA_ROOTS
    )
    below_named_output_directory = any(
        part in FORBIDDEN_OUTPUT_DIRECTORY_NAMES for part in path.parts[:-1]
    )
    return below_data_root or below_named_output_directory


def is_environment_secret(path: Path) -> bool:
    name = path.name
    return name != ".env.example" and (name == ".env" or name.startswith(".env."))


def is_credential_path(path: Path) -> bool:
    return (
        ".ssh" in path.parts
        or path.name.lower() in FORBIDDEN_CREDENTIAL_NAMES
        or path.suffix.lower() in FORBIDDEN_CREDENTIAL_SUFFIXES
    )


def is_model_weight(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
        or ".ckpt." in name
        or any(part.lower() in FORBIDDEN_MODEL_DIRECTORY_NAMES for part in path.parts[:-1])
    )


def blob_content(
    tracked: TrackedObject,
    repository: Path,
    blob_cache: dict[str, bytes],
) -> Optional[bytes]:
    if (
        tracked.object_type != "blob"
        or tracked.object_id is None
        or tracked.size > MAX_TRACKED_BYTES
    ):
        return None
    content = blob_cache.get(tracked.object_id)
    if content is None:
        content = subprocess.run(
            ["git", "cat-file", "blob", tracked.object_id],
            check=True,
            capture_output=True,
            cwd=repository,
        ).stdout
        blob_cache[tracked.object_id] = content
    return content


def secret_labels(
    tracked: TrackedObject,
    repository: Path,
    blob_cache: dict[str, bytes],
) -> list[str]:
    content = blob_content(tracked, repository, blob_cache)
    if content is None:
        return []
    return [label for label, pattern in SECRET_CONTENT_PATTERNS if pattern.search(content)]


def personal_data_labels(
    tracked: TrackedObject,
    repository: Path,
    blob_cache: dict[str, bytes],
) -> list[str]:
    content = blob_content(tracked, repository, blob_cache)
    if content is None:
        return []
    return [
        label for label, pattern in PERSONAL_DATA_PATTERNS if pattern.search(content)
    ]


def is_manifest_path(path: Path) -> bool:
    return MANIFEST_ROOT in path.parents and path.name != "README.md"


def parse_iso8601_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def is_iso8601_timestamp(value: object) -> bool:
    return parse_iso8601_timestamp(value) is not None


def is_zero_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def recipe_errors(recipe: object, prefix: str) -> list[str]:
    if not isinstance(recipe, dict):
        return [f"{prefix} must be an object"]
    errors: list[str] = []
    for field in ("implementation", "version"):
        if not isinstance(recipe.get(field), str) or not recipe[field].strip():
            errors.append(f"{prefix}.{field} must be a non-empty string")
    if not isinstance(recipe.get("parameters"), dict):
        errors.append(f"{prefix}.parameters must be an object")
    return errors


def integrity_report_errors(report: object, prefix: str) -> list[str]:
    if not isinstance(report, dict):
        return [f"{prefix} must be an object"]
    errors: list[str] = []
    if not isinstance(report.get("schema_version"), str) or not report[
        "schema_version"
    ].strip():
        errors.append(f"{prefix}.schema_version must be a non-empty string")
    if not is_iso8601_timestamp(report.get("generated_at")):
        errors.append(f"{prefix}.generated_at must be an ISO-8601 timestamp with timezone")

    required_fields = report.get("required_fields")
    if not isinstance(required_fields, dict):
        errors.append(f"{prefix}.required_fields must be an object")
    elif required_fields.get("status") != "passed" or not is_zero_integer(
        required_fields.get("missing_count")
    ):
        errors.append(
            f"{prefix}.required_fields must report passed with missing_count 0"
        )

    duplicates = report.get("duplicates")
    if not isinstance(duplicates, dict):
        errors.append(f"{prefix}.duplicates must be an object")
    elif duplicates.get("status") != "passed" or not is_zero_integer(
        duplicates.get("count")
    ):
        errors.append(f"{prefix}.duplicates must report passed with count 0")

    schema_validation = report.get("schema_validation")
    if not isinstance(schema_validation, dict):
        errors.append(f"{prefix}.schema_validation must be an object")
    elif schema_validation.get("status") != "passed" or not is_zero_integer(
        schema_validation.get("error_count")
    ):
        errors.append(
            f"{prefix}.schema_validation must report passed with error_count 0"
        )

    split_integrity = report.get("split_integrity")
    if not isinstance(split_integrity, dict):
        errors.append(f"{prefix}.split_integrity must be an object")
    else:
        entities = split_integrity.get("entities")
        if not isinstance(entities, dict):
            errors.append(f"{prefix}.split_integrity.entities must be an object")
        else:
            for entity in ("speaker", "source", "event"):
                result = entities.get(entity)
                entity_prefix = f"{prefix}.split_integrity.entities.{entity}"
                if not isinstance(result, dict):
                    errors.append(f"{entity_prefix} must be an object")
                    continue
                status = result.get("status")
                if status not in {"passed", "not_applicable"}:
                    errors.append(
                        f"{entity_prefix}.status must be passed or not_applicable"
                    )
                if status == "passed" and not is_zero_integer(
                    result.get("overlap_count")
                ):
                    errors.append(
                        f"{entity_prefix}.overlap_count must be 0 when passed"
                    )
                if status == "not_applicable" and (
                    not isinstance(result.get("reason"), str)
                    or not result["reason"].strip()
                ):
                    errors.append(
                        f"{entity_prefix}.reason is required when not_applicable"
                    )

    source_drift = report.get("source_drift")
    if not isinstance(source_drift, dict):
        errors.append(f"{prefix}.source_drift must be an object")
    else:
        drift_status = source_drift.get("status")
        if drift_status not in {"passed", "not_applicable"}:
            errors.append(
                f"{prefix}.source_drift.status must be passed or not_applicable"
            )
        changes_detected = source_drift.get("changes_detected")
        if not isinstance(changes_detected, int) or isinstance(
            changes_detected, bool
        ) or changes_detected < 0:
            errors.append(
                f"{prefix}.source_drift.changes_detected must be a non-negative integer"
            )
        elif changes_detected != 0:
            errors.append(
                f"{prefix}.source_drift.changes_detected must be 0"
            )
        if drift_status == "not_applicable" and (
            not isinstance(source_drift.get("reason"), str)
            or not source_drift["reason"].strip()
        ):
            errors.append(
                f"{prefix}.source_drift.reason is required when not_applicable"
            )
    return errors


def split_uses_randomness(split: dict) -> bool:
    """Return whether a split recipe declares stochastic behavior."""
    random_markers = re.compile(
        r"(?:^|[^a-z])(random|shuffle|stochastic|bootstrap|resample)"
    )
    strategy = split.get("strategy")
    parameters = split.get("parameters")
    searchable = " ".join(
        (
            strategy.lower() if isinstance(strategy, str) else "",
            json.dumps(parameters, sort_keys=True).lower()
            if isinstance(parameters, dict)
            else "",
        )
    )
    return bool(random_markers.search(searchable))


def split_seed_errors(split: dict, prefix: str) -> list[str]:
    seed = split.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        return [f"{prefix}.seed must be an integer or null"]
    if seed is None and split_uses_randomness(split):
        return [f"{prefix}.seed must be an integer for a stochastic strategy"]
    return []


def preprocessing_seed_errors(recipe: dict, prefix: str) -> list[str]:
    seed = recipe.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        return [f"{prefix}.seed must be an integer or null"]
    stochastic_recipe = {
        "strategy": recipe.get("implementation"),
        "parameters": recipe.get("parameters"),
    }
    if seed is None and split_uses_randomness(stochastic_recipe):
        return [f"{prefix}.seed must be an integer for stochastic preprocessing"]
    return []


def evaluation_usage_errors(manifest: dict, prefix: str) -> list[str]:
    usage_role = manifest.get("usage_role")
    if usage_role not in MANIFEST_USAGE_ROLES:
        allowed = ", ".join(sorted(MANIFEST_USAGE_ROLES))
        return [f"{prefix} usage_role must be one of: {allowed}"]

    errors: list[str] = []
    split = manifest.get("split")
    if isinstance(split, dict):
        split_name = split.get("name")
        parameters = split.get("parameters")
        evaluation_split = isinstance(split_name, str) and bool(
            re.search(r"(?:^|[^a-z])(test|eval|evaluation)(?:$|[^a-z])", split_name.lower())
        )
        if (
            (usage_role == "evaluation" or evaluation_split)
            and isinstance(parameters, dict)
            and parameters.get("used_for_tuning") is True
        ):
            errors.append(f"{prefix} evaluation split must not be used for tuning")

    evaluation = manifest.get("evaluation")
    if usage_role != "evaluation":
        if evaluation is not None:
            errors.append(
                f"{prefix} evaluation metadata is only allowed for usage_role evaluation"
            )
        return errors
    if not isinstance(evaluation, dict):
        errors.append(f"{prefix} evaluation must be an object for usage_role evaluation")
        return errors
    evaluation_id = evaluation.get("id")
    expected_count = NAMED_EVALUATION_RECORD_COUNTS.get(evaluation_id)
    dataset_id = manifest.get("dataset_id")
    if expected_count is None and isinstance(evaluation_id, str):
        dynamic_match = CROSS_REGION_SPEECH_EVALUATION_PATTERN.fullmatch(
            evaluation_id
        ) or RADIO_SIMULATION_EVALUATION_PATTERN.fullmatch(evaluation_id)
        if dynamic_match and dataset_id == dynamic_match.group(1):
            expected_count = int(dynamic_match.group(2))
    if expected_count is None:
        errors.append(
            f"{prefix} evaluation.id is not a registered fixed, cross-region, "
            "or radio-simulation evaluation"
        )
    record_count = evaluation.get("record_count")
    if not isinstance(record_count, int) or isinstance(record_count, bool):
        errors.append(f"{prefix} evaluation.record_count must be an integer")
    elif expected_count is not None and record_count != expected_count:
        errors.append(
            f"{prefix} evaluation {evaluation_id} must have record_count {expected_count}"
        )
    return errors


def manifest_timestamp_errors(manifest: dict, prefix: str) -> list[str]:
    created_at = parse_iso8601_timestamp(manifest.get("created_at"))
    source = manifest.get("source")
    collected_at = (
        parse_iso8601_timestamp(source.get("collected_at"))
        if isinstance(source, dict)
        else None
    )
    integrity_report = manifest.get("integrity_report")
    report_generated_at = (
        parse_iso8601_timestamp(integrity_report.get("generated_at"))
        if isinstance(integrity_report, dict)
        else None
    )
    if report_generated_at is None:
        return []
    errors: list[str] = []
    if collected_at is not None and report_generated_at < collected_at:
        errors.append(
            f"{prefix} integrity_report.generated_at must not predate source.collected_at"
        )
    if created_at is not None and report_generated_at < created_at:
        errors.append(
            f"{prefix} integrity_report.generated_at must not predate created_at"
        )
    return errors


def manifest_errors(
    tracked: TrackedObject,
    repository: Path,
    blob_cache: dict[str, bytes],
) -> list[str]:
    if not is_manifest_path(tracked.path):
        return []
    prefix = f"invalid dataset manifest {tracked.path}:"
    if tracked.path.suffix.lower() != ".json":
        return [f"{prefix} only JSON manifests are supported"]
    content = blob_content(tracked, repository, blob_cache)
    if content is None:
        return [f"{prefix} content is unavailable for validation"]
    try:
        manifest = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"{prefix} malformed UTF-8 JSON ({error})"]
    if not isinstance(manifest, dict):
        return [f"{prefix} root must be a JSON object"]

    errors: list[str] = []
    for field in ("schema_version", "dataset_id", "dataset_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            errors.append(f"{prefix} {field} must be a non-empty string")
    if not is_iso8601_timestamp(manifest.get("created_at")):
        errors.append(f"{prefix} created_at must be an ISO-8601 timestamp with timezone")

    classification = manifest.get("classification")
    if classification not in MANIFEST_CLASSIFICATIONS:
        allowed = ", ".join(sorted(MANIFEST_CLASSIFICATIONS))
        errors.append(f"{prefix} classification must be one of: {allowed}")
    errors.extend(evaluation_usage_errors(manifest, prefix))

    source = manifest.get("source")
    if not isinstance(source, dict):
        errors.append(f"{prefix} source must be an object")
    else:
        for field in ("name", "url", "license", "version"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                errors.append(f"{prefix} source.{field} must be a non-empty string")
        if not is_iso8601_timestamp(source.get("collected_at")):
            errors.append(
                f"{prefix} source.collected_at must be an ISO-8601 timestamp with timezone"
            )

    split = manifest.get("split")
    if not isinstance(split, dict):
        errors.append(f"{prefix} split must be an object")
    else:
        for field in ("name", "strategy", "unit"):
            if not isinstance(split.get(field), str) or not split[field].strip():
                errors.append(f"{prefix} split.{field} must be a non-empty string")
        if not isinstance(split.get("parameters"), dict):
            errors.append(f"{prefix} split.parameters must be an object")
        errors.extend(split_seed_errors(split, f"{prefix} split"))

    if classification == "derived":
        preprocessing = manifest.get("preprocessing")
        errors.extend(recipe_errors(preprocessing, f"{prefix} preprocessing"))
        if isinstance(preprocessing, dict):
            errors.extend(
                preprocessing_seed_errors(preprocessing, f"{prefix} preprocessing")
            )
    if classification == "synthetic":
        generation = manifest.get("generation")
        errors.extend(recipe_errors(generation, f"{prefix} generation"))
        if isinstance(generation, dict):
            seed = generation.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                errors.append(f"{prefix} generation.seed must be an integer")

    errors.extend(
        integrity_report_errors(
            manifest.get("integrity_report"), f"{prefix} integrity_report"
        )
    )
    errors.extend(manifest_timestamp_errors(manifest, prefix))

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{prefix} artifacts must be a non-empty array")
    else:
        for index, artifact in enumerate(artifacts):
            item_prefix = f"{prefix} artifacts[{index}]"
            if not isinstance(artifact, dict):
                errors.append(f"{item_prefix} must be an object")
                continue
            if not isinstance(artifact.get("path"), str) or not artifact["path"].strip():
                errors.append(f"{item_prefix}.path must be a non-empty string")
            else:
                artifact_path = Path(artifact["path"])
                if artifact_path.is_absolute() or ".." in artifact_path.parts:
                    errors.append(f"{item_prefix}.path must be repository-relative")
            sha256 = artifact.get("sha256")
            if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
                errors.append(f"{item_prefix}.sha256 must be 64 hexadecimal characters")
    return errors


def manifest_diff_errors(
    old_ref: str,
    new_ref: str,
    label: str,
    repository: Path = Path("."),
) -> list[str]:
    diff_result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            old_ref,
            new_ref,
            "--",
            str(MANIFEST_ROOT),
        ],
        check=True,
        capture_output=True,
        cwd=repository,
    )
    fields = diff_result.stdout.split(b"\0")
    errors: list[str] = []
    for index in range(0, len(fields) - 1, 2):
        status = fields[index].decode("ascii", errors="replace")
        raw_path = fields[index + 1]
        if not status or not raw_path:
            continue
        path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        if is_manifest_path(path) and status != "A":
            errors.append(
                f"published dataset manifests are append-only ({status} in {label}): {path}"
            )
    return errors


def append_only_manifest_errors(
    base_ref: Optional[str], repository: Path = Path(".")
) -> list[str]:
    revision = f"{base_ref}..HEAD" if base_ref else "HEAD"
    commit_result = subprocess.run(
        ["git", "rev-list", "--reverse", revision],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository,
    )
    errors: list[str] = []
    if base_ref:
        errors.extend(
            manifest_diff_errors(base_ref, "HEAD", f"{base_ref}..HEAD", repository)
        )
    for commit in commit_result.stdout.splitlines():
        parent_result = subprocess.run(
            ["git", "rev-list", "--parents", "-n", "1", commit],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository,
        )
        parents = parent_result.stdout.strip().split()[1:]
        for parent in parents:
            errors.extend(
                manifest_diff_errors(parent, commit, commit[:12], repository)
            )
    return errors


def staged_append_only_manifest_errors(repository: Path = Path(".")) -> list[str]:
    has_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        cwd=repository,
    )
    if has_head.returncode != 0:
        return []
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-status",
            "-z",
            "--no-renames",
            "HEAD",
            "--",
            str(MANIFEST_ROOT),
        ],
        check=True,
        capture_output=True,
        cwd=repository,
    )
    fields = result.stdout.split(b"\0")
    errors: list[str] = []
    for index in range(0, len(fields) - 1, 2):
        status = fields[index].decode("ascii", errors="replace")
        raw_path = fields[index + 1]
        if not status or not raw_path:
            continue
        path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        if is_manifest_path(path) and status != "A":
            errors.append(
                f"published dataset manifests are append-only (staged {status}): {path}"
            )
    return errors


def is_fixture_metadata(path: Path) -> bool:
    return (
        "fixtures" in path.parts[:-1]
        and path.name.endswith(FIXTURE_METADATA_SUFFIX)
    )


def is_fixture_payload(path: Path) -> bool:
    return (
        "fixtures" in path.parts[:-1]
        and path.name != "README.md"
        and not is_fixture_metadata(path)
    )


def fixture_metadata_path(path: Path) -> Path:
    return path.with_name(path.name + FIXTURE_METADATA_SUFFIX)


def fixture_payload_path(path: Path) -> Path:
    return path.with_name(path.name[: -len(FIXTURE_METADATA_SUFFIX)])


def fixture_errors(
    tracked: TrackedObject,
    object_lookup: dict[tuple[str, Path], TrackedObject],
    repository: Path,
    blob_cache: dict[str, bytes],
) -> list[str]:
    if not is_fixture_payload(tracked.path):
        return []
    prefix = f"invalid fixture {tracked.path}:"
    if tracked.object_mode == "120000":
        return [f"{prefix} symbolic-link payloads are not allowed"]
    errors: list[str] = []
    content = blob_content(tracked, repository, blob_cache)
    if content is None:
        return [f"{prefix} payload must be an available Git blob"]

    metadata_path = fixture_metadata_path(tracked.path)
    metadata_object = object_lookup.get((tracked.revision, metadata_path))
    if metadata_object is None:
        errors.append(f"{prefix} missing companion {metadata_path}")
        return errors
    metadata_content = blob_content(metadata_object, repository, blob_cache)
    if metadata_content is None:
        errors.append(f"{prefix} fixture metadata is unavailable")
        return errors
    try:
        metadata = json.loads(metadata_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"{prefix} malformed fixture metadata ({error})")
        return errors
    if not isinstance(metadata, dict):
        errors.append(f"{prefix} fixture metadata root must be an object")
        return errors

    if not isinstance(metadata.get("schema_version"), str) or not metadata[
        "schema_version"
    ].strip():
        errors.append(f"{prefix} metadata.schema_version must be a non-empty string")
    classification = metadata.get("classification")
    if classification not in FIXTURE_CLASSIFICATIONS:
        allowed = ", ".join(sorted(FIXTURE_CLASSIFICATIONS))
        errors.append(f"{prefix} metadata.classification must be one of: {allowed}")
    if metadata.get("contains_personal_data") is not False:
        errors.append(f"{prefix} metadata.contains_personal_data must be false")
    if not isinstance(metadata.get("license"), str) or not metadata["license"].strip():
        errors.append(f"{prefix} metadata.license must be a non-empty string")

    source = metadata.get("source")
    if not isinstance(source, dict):
        errors.append(f"{prefix} metadata.source must be an object")
    else:
        for field in ("name", "url", "version"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                errors.append(
                    f"{prefix} metadata.source.{field} must be a non-empty string"
                )
        if not is_iso8601_timestamp(source.get("collected_at")):
            errors.append(
                f"{prefix} metadata.source.collected_at must be an ISO-8601 timestamp with timezone"
            )

    declared_sha256 = metadata.get("sha256")
    if not isinstance(declared_sha256, str) or not SHA256_PATTERN.fullmatch(
        declared_sha256
    ):
        errors.append(f"{prefix} metadata.sha256 must be 64 hexadecimal characters")
    elif content is not None and hashlib.sha256(content).hexdigest() != declared_sha256.lower():
        errors.append(f"{prefix} metadata.sha256 does not match the fixture blob")

    if classification == "synthetic":
        generation = metadata.get("generation")
        errors.extend(recipe_errors(generation, f"{prefix} metadata.generation"))
        if isinstance(generation, dict):
            seed = generation.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                errors.append(f"{prefix} metadata.generation.seed must be an integer")
    return errors


def fixture_metadata_pair_errors(
    tracked: TrackedObject,
    object_lookup: dict[tuple[str, Path], TrackedObject],
    repository: Path,
    blob_cache: dict[str, bytes],
) -> list[str]:
    if not is_fixture_metadata(tracked.path):
        return []
    payload_path = fixture_payload_path(tracked.path)
    payload = object_lookup.get((tracked.revision, payload_path))
    if payload is None:
        return [f"orphan fixture metadata {tracked.path}: missing payload {payload_path}"]
    if fixture_errors(payload, object_lookup, repository, blob_cache):
        return [
            f"fixture metadata {tracked.path} does not validate its companion payload"
        ]
    return []


def is_approved_dataset_content_path(
    path: Path,
    is_approved_fixture: bool,
    is_approved_fixture_metadata: bool,
    is_approved_schema: bool,
) -> bool:
    return (
        is_approved_fixture
        or is_approved_fixture_metadata
        or is_manifest_path(path)
        or is_approved_schema
    )


def is_schema_path(path: Path) -> bool:
    return (
        (path.parent == SCHEMA_ROOT or SCHEMA_ROOT in path.parents)
        and path.name.lower().endswith(".schema.json")
    )


def schema_errors(
    tracked: TrackedObject,
    repository: Path,
    blob_cache: dict[str, bytes],
) -> list[str]:
    if not is_schema_path(tracked.path):
        return []
    prefix = f"invalid JSON schema {tracked.path}:"
    content = blob_content(tracked, repository, blob_cache)
    if content is None:
        return [f"{prefix} content is unavailable for validation"]
    try:
        schema = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"{prefix} malformed UTF-8 JSON ({error})"]
    if not isinstance(schema, dict):
        return [f"{prefix} root must be a JSON object"]
    dialect = schema.get("$schema")
    if not isinstance(dialect, str) or not re.fullmatch(
        r"https?://json-schema\.org/draft(?:/|-)[^\s#]+#?", dialect
    ):
        return [f"{prefix} $schema must declare a json-schema.org draft"]
    if not any(
        keyword in schema
        for keyword in ("$ref", "allOf", "anyOf", "oneOf", "properties", "type")
    ):
        return [f"{prefix} must contain a structural JSON Schema keyword"]
    validator_class = validator_for(schema, default=None)
    if validator_class is None:
        return [f"{prefix} $schema dialect is not supported by jsonschema"]
    try:
        validator_class.check_schema(schema)
    except SchemaError as error:
        return [f"{prefix} document does not satisfy its metaschema ({error.message})"]
    return []


def is_plain_text_corpus(path: Path) -> bool:
    """Identify common transcript/corpus text files without blocking general .txt files."""
    if path.suffix.lower() not in PLAIN_TEXT_CORPUS_SUFFIXES:
        return False
    parent_names = {part.lower() for part in path.parts[:-1]}
    stem_tokens = {token for token in re.split(r"[^a-z0-9]+", path.stem.lower()) if token}
    return bool(parent_names & CORPUS_DIRECTORY_NAMES) or bool(
        stem_tokens & CORPUS_DIRECTORY_NAMES
    )


def violations(
    objects: list[TrackedObject], repository: Path = Path(".")
) -> list[str]:
    errors: list[str] = []
    blob_cache: dict[str, bytes] = {}
    object_lookup = {(tracked.revision, tracked.path): tracked for tracked in objects}
    for tracked in objects:
        fixture_validation_errors = fixture_errors(
            tracked, object_lookup, repository, blob_cache
        )
        is_approved_fixture = (
            is_fixture_payload(tracked.path) and not fixture_validation_errors
        )
        metadata_pair_errors = fixture_metadata_pair_errors(
            tracked, object_lookup, repository, blob_cache
        )
        is_approved_fixture_metadata = (
            is_fixture_metadata(tracked.path) and not metadata_pair_errors
        )
        schema_validation_errors = schema_errors(tracked, repository, blob_cache)
        is_approved_schema = (
            is_schema_path(tracked.path) and not schema_validation_errors
        )
        if tracked.object_type == "blob" and tracked.size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 10 MiB: {tracked.path}")
        if is_model_weight(tracked.path):
            errors.append(f"model weight must not be tracked: {tracked.path}")
        if (
            tracked.path.suffix.lower() in FORBIDDEN_AUDIO_SUFFIXES
            and not is_approved_fixture
        ):
            errors.append(f"audio data must not be tracked: {tracked.path}")
        if tracked.path.suffix.lower() in FORBIDDEN_DOCUMENT_ARCHIVE_SUFFIXES:
            errors.append(f"source document or archive must not be tracked: {tracked.path}")
        if (
            (
                tracked.path.suffix.lower() in DATASET_CONTENT_SUFFIXES
                or is_plain_text_corpus(tracked.path)
            )
            and not is_approved_dataset_content_path(
                tracked.path,
                is_approved_fixture,
                is_approved_fixture_metadata,
                is_approved_schema,
            )
        ):
            errors.append(
                f"dataset content must be stored outside Git or as an approved fixture: {tracked.path}"
            )
        if is_environment_secret(tracked.path):
            errors.append(f"environment secret file must not be tracked: {tracked.path}")
        if is_credential_path(tracked.path):
            errors.append(f"credential file must not be tracked: {tracked.path}")
        for label in secret_labels(tracked, repository, blob_cache):
            errors.append(f"possible {label} detected in tracked blob: {tracked.path}")
        for label in personal_data_labels(tracked, repository, blob_cache):
            errors.append(f"possible {label} detected in tracked blob: {tracked.path}")
        errors.extend(manifest_errors(tracked, repository, blob_cache))
        errors.extend(fixture_validation_errors)
        errors.extend(metadata_pair_errors)
        errors.extend(schema_validation_errors)
        is_allowed_placeholder = (
            tracked.object_type == "blob" and tracked.path in ALLOWED_PLACEHOLDERS
        )
        if not is_allowed_placeholder and is_forbidden_output(tracked.path):
            errors.append(f"generated output must not be tracked: {tracked.path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    scan_mode = parser.add_mutually_exclusive_group()
    scan_mode.add_argument(
        "--base-ref",
        help="scan every file path in every commit after this Git revision",
    )
    scan_mode.add_argument(
        "--all-history",
        action="store_true",
        help="scan every file path in every commit reachable from HEAD",
    )
    args = parser.parse_args()
    if args.base_ref:
        objects = commit_range_objects(args.base_ref)
    elif args.all_history:
        objects = reachable_commit_objects()
    else:
        objects = current_tree_objects()
    errors = violations(objects)
    if args.base_ref:
        errors.extend(append_only_manifest_errors(args.base_ref))
    elif args.all_history:
        errors.extend(append_only_manifest_errors(None))
    else:
        errors.extend(staged_append_only_manifest_errors())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    manifest_count = sum(is_manifest_path(tracked.path) for tracked in objects)
    print(
        "repository data policy check passed; "
        f"validated dataset manifests: {manifest_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

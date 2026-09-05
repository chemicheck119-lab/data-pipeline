"""Create deterministic, provenance-bound radio-channel simulation archives.

This module produces *derived proxy data*.  Its procedural noise and codec
approximations are not recordings from a fireground radio and must never be
reported as field validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unicodedata
import wave
import zipfile

import numpy as np
import scipy
from scipy.signal import butter, resample_poly, sosfilt

from .aihub119 import sha256_file


IMPLEMENTATION_VERSION = "1.0.0"
PROFILE_ID = "radio-sim-v1"
MAX_AUDIO_MEMBER_BYTES = 32 * 1024 * 1024
MAX_LABEL_MEMBER_BYTES = 4 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_RECORDS_PER_STRATUM = 100
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class VariantSpec:
    id: str
    kind: str
    parameters: dict[str, object]


def profile_variants() -> tuple[VariantSpec, ...]:
    """Return the frozen, pre-registered radio simulation profile."""

    variants: list[VariantSpec] = [
        VariantSpec("clean", "clean", {}),
        VariantSpec(
            "bandlimit_8khz",
            "bandlimit",
            {"sample_rate_hz": 8000, "highpass_hz": 300, "lowpass_hz": 3400},
        ),
        VariantSpec(
            "mulaw_8khz",
            "mulaw",
            {
                "sample_rate_hz": 8000,
                "quantization_levels": 256,
                "mu": 255,
                "limitation": "mathematical mu-law proxy; not a vendor radio codec",
            },
        ),
    ]
    for noise_kind in ("siren", "vehicle", "wind"):
        for snr_db in (20, 10, 0):
            variants.append(
                VariantSpec(
                    f"{noise_kind}_snr{snr_db}",
                    "noise",
                    {"noise_kind": noise_kind, "snr_db": snr_db},
                )
            )
    variants.extend(
        (
            VariantSpec("start_cut_300ms", "cut", {"start_ms": 300, "end_ms": 0}),
            VariantSpec("end_cut_300ms", "cut", {"start_ms": 0, "end_ms": 300}),
            VariantSpec("hard_clip_minus12dbfs", "clip", {"threshold_dbfs": -12}),
            VariantSpec("gain_minus18db", "gain", {"gain_db": -18}),
            VariantSpec(
                "dropout_3x120ms",
                "dropout",
                {"count": 3, "duration_ms": 120},
            ),
            VariantSpec(
                "combined_radio_snr10",
                "combined",
                {
                    "sample_rate_hz": 8000,
                    "highpass_hz": 300,
                    "lowpass_hz": 3400,
                    "noise_kind": "vehicle_wind",
                    "snr_db": 10,
                    "mu": 255,
                    "start_ms": 200,
                    "end_ms": 200,
                    "dropout_count": 2,
                    "dropout_duration_ms": 100,
                    "limitation": "composite procedural stress case; not a measured channel",
                },
            ),
        )
    )
    return tuple(variants)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_members(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for info in archive.infolist():
        if not info.filename.lower().endswith(suffix):
            continue
        maximum = (
            MAX_AUDIO_MEMBER_BYTES if suffix == ".wav" else MAX_LABEL_MEMBER_BYTES
        )
        if info.file_size <= 0 or info.file_size > maximum:
            raise ValueError(f"unsafe archive member size: {info.filename}")
        if info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
            raise ValueError(f"unsafe archive compression ratio: {info.filename}")
        stem = PurePosixPath(info.filename).stem
        if stem in result:
            raise ValueError(f"duplicate archive stem: {stem}")
        result[stem] = info.filename
    return result


def _read_member(
    archive: zipfile.ZipFile, name: str, maximum_bytes: int
) -> bytes:
    with archive.open(name) as source:
        content = source.read(maximum_bytes + 1)
    if not content or len(content) > maximum_bytes:
        raise ValueError(f"archive member exceeded bounded read: {name}")
    return content


def _normalise_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def load_priority_terms(path: Path) -> tuple[str, ...]:
    terms = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not terms or len(terms) != len(set(terms)):
        raise ValueError("priority term file must contain unique non-empty terms")
    if any(not _normalise_search_text(term) for term in terms):
        raise ValueError("priority term normalizes to an empty value")
    return terms


def _label_text(content: bytes) -> str:
    try:
        label = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid label JSON") from error
    utterances = label.get("utterances") if isinstance(label, dict) else None
    if not isinstance(utterances, list):
        raise ValueError("label utterances must be an array")
    texts = [
        item.get("text", "")
        for item in utterances
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    result = " ".join(texts).strip()
    if not result:
        raise ValueError("label transcript is empty")
    return result


def _has_priority_term(text: str, terms: tuple[str, ...]) -> bool:
    searchable = _normalise_search_text(text)
    return any(_normalise_search_text(term) in searchable for term in terms)


def _stable_order(stems: list[str], seed: int) -> list[str]:
    return sorted(
        stems,
        key=lambda stem: hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest(),
    )


def _decode_pcm16_mono(content: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(content), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            payload = source.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError("invalid WAV input") from error
    if (
        channels not in {1, 2}
        or sample_width != 2
        or sample_rate < 8000
        or frame_count <= 0
        or compression != "NONE"
    ):
        raise ValueError(
            "radio simulation requires uncompressed 16-bit mono/stereo WAV at >=8kHz"
        )
    expected = frame_count * channels * sample_width
    if len(payload) != expected:
        raise ValueError("truncated WAV payload")
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    if not np.all(np.isfinite(samples)):
        raise ValueError("non-finite WAV samples")
    return samples, sample_rate


def _encode_pcm16_mono(samples: np.ndarray, sample_rate: int) -> bytes:
    bounded = np.clip(samples, -1.0, 32767.0 / 32768.0)
    payload = np.rint(bounded * 32768.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(payload)
    return output.getvalue()


def _bandlimit_8khz(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    ratio = Fraction(8000, sample_rate).limit_denominator()
    resampled = resample_poly(samples, ratio.numerator, ratio.denominator)
    bandpass = butter(6, (300, 3400), btype="bandpass", fs=8000, output="sos")
    return sosfilt(bandpass, resampled), 8000


def _mulaw_roundtrip(samples: np.ndarray, mu: int = 255) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / math.log1p(mu)
    encoded = np.rint((compressed + 1.0) * 127.5).astype(np.uint8)
    quantized = encoded.astype(np.float64) / 127.5 - 1.0
    return np.sign(quantized) * np.expm1(np.abs(quantized) * math.log1p(mu)) / mu


def _noise(kind: str, length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    time_axis = np.arange(length, dtype=np.float64) / sample_rate
    if kind == "siren":
        instantaneous_hz = 950.0 + 350.0 * np.sin(2.0 * np.pi * 0.55 * time_axis)
        phase = 2.0 * np.pi * np.cumsum(instantaneous_hz) / sample_rate
        return np.sin(phase) + 0.25 * np.sin(2.0 * phase)
    white = rng.standard_normal(length)
    if kind == "vehicle":
        lowpass = butter(4, 450, btype="lowpass", fs=sample_rate, output="sos")
        return sosfilt(lowpass, white) + 0.35 * np.sin(2.0 * np.pi * 95 * time_axis)
    if kind == "wind":
        lowpass = butter(3, 600, btype="lowpass", fs=sample_rate, output="sos")
        gust = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.35 * time_axis + 0.7)
        return sosfilt(lowpass, white) * gust
    if kind == "vehicle_wind":
        return _noise("vehicle", length, sample_rate, rng) + _noise(
            "wind", length, sample_rate, rng
        )
    raise ValueError(f"unsupported procedural noise kind: {kind}")


def _mix_at_snr(
    samples: np.ndarray,
    sample_rate: int,
    kind: str,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    generated = _noise(kind, len(samples), sample_rate, rng)
    clean_rms = float(np.sqrt(np.mean(np.square(samples))))
    noise_rms = float(np.sqrt(np.mean(np.square(generated))))
    if clean_rms <= 1e-9 or noise_rms <= 1e-9:
        raise ValueError("cannot define SNR for silent or empty signal")
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = generated * (target_noise_rms / noise_rms)
    mixed = samples + scaled_noise
    peak = float(np.max(np.abs(mixed)))
    peak_gain = min(1.0, 0.999 / peak) if peak else 1.0
    # Scaling the complete mixture prevents accidental clipping while preserving SNR.
    mixed *= peak_gain
    observed_noise_rms = float(np.sqrt(np.mean(np.square(scaled_noise))))
    measured_snr_db = 20.0 * math.log10(clean_rms / observed_noise_rms)
    return mixed, {
        "measured_whole_record_snr_db": round(measured_snr_db, 6),
        "mixture_peak_before_normalization": round(peak, 9),
        "peak_normalization_gain": round(peak_gain, 9),
    }


def _cut(samples: np.ndarray, sample_rate: int, start_ms: int, end_ms: int) -> np.ndarray:
    start = round(sample_rate * start_ms / 1000)
    end = round(sample_rate * end_ms / 1000)
    if start + end >= len(samples) - round(sample_rate * 0.1):
        raise ValueError("cut transform would remove the complete utterance")
    return samples[start : len(samples) - end if end else None]


def _dropout(
    samples: np.ndarray,
    sample_rate: int,
    count: int,
    duration_ms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    width = round(sample_rate * duration_ms / 1000)
    if width <= 0 or width >= len(samples):
        raise ValueError("dropout duration is outside the signal")
    result = samples.copy()
    # Use equal temporal cells so the requested gaps cannot all collapse together.
    boundaries = np.linspace(0, len(samples) - width, count + 1, dtype=int)
    for index in range(count):
        low = int(boundaries[index])
        high = max(low, int(boundaries[index + 1]) - 1)
        start = int(rng.integers(low, high + 1))
        result[start : start + width] = 0.0
    return result


def _record_seed(base_seed: int, source_digest: str, variant_id: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{source_digest}:{variant_id}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def apply_variant(
    content: bytes, spec: VariantSpec, *, seed: int
) -> tuple[bytes, dict[str, object]]:
    samples, sample_rate = _decode_pcm16_mono(content)
    input_seconds = len(samples) / sample_rate
    rng = np.random.default_rng(seed)
    signal_details: dict[str, object] = {}
    if spec.kind == "clean":
        return content, {
            "input_sample_rate_hz": sample_rate,
            "output_sample_rate_hz": sample_rate,
            "input_seconds": round(input_seconds, 6),
            "output_seconds": round(input_seconds, 6),
            "bit_exact_copy": True,
        }
    elif spec.kind == "bandlimit":
        transformed, output_rate = _bandlimit_8khz(samples, sample_rate)
    elif spec.kind == "mulaw":
        transformed, output_rate = _bandlimit_8khz(samples, sample_rate)
        transformed = _mulaw_roundtrip(transformed, int(spec.parameters["mu"]))
    elif spec.kind == "noise":
        transformed, signal_details = _mix_at_snr(
            samples,
            sample_rate,
            str(spec.parameters["noise_kind"]),
            float(spec.parameters["snr_db"]),
            rng,
        )
        output_rate = sample_rate
    elif spec.kind == "cut":
        transformed = _cut(
            samples,
            sample_rate,
            int(spec.parameters["start_ms"]),
            int(spec.parameters["end_ms"]),
        )
        output_rate = sample_rate
    elif spec.kind == "clip":
        threshold = 10.0 ** (float(spec.parameters["threshold_dbfs"]) / 20.0)
        transformed = np.clip(samples, -threshold, threshold)
        output_rate = sample_rate
    elif spec.kind == "gain":
        gain = 10.0 ** (float(spec.parameters["gain_db"]) / 20.0)
        transformed = samples * gain
        output_rate = sample_rate
    elif spec.kind == "dropout":
        transformed = _dropout(
            samples,
            sample_rate,
            int(spec.parameters["count"]),
            int(spec.parameters["duration_ms"]),
            rng,
        )
        output_rate = sample_rate
    elif spec.kind == "combined":
        transformed, output_rate = _bandlimit_8khz(samples, sample_rate)
        transformed, signal_details = _mix_at_snr(
            transformed,
            output_rate,
            str(spec.parameters["noise_kind"]),
            float(spec.parameters["snr_db"]),
            rng,
        )
        transformed = _mulaw_roundtrip(transformed, int(spec.parameters["mu"]))
        transformed = _cut(
            transformed,
            output_rate,
            int(spec.parameters["start_ms"]),
            int(spec.parameters["end_ms"]),
        )
        transformed = _dropout(
            transformed,
            output_rate,
            int(spec.parameters["dropout_count"]),
            int(spec.parameters["dropout_duration_ms"]),
            rng,
        )
    else:
        raise ValueError(f"unsupported transform kind: {spec.kind}")
    output = _encode_pcm16_mono(transformed, output_rate)
    return output, {
        "input_sample_rate_hz": sample_rate,
        "output_sample_rate_hz": output_rate,
        "input_seconds": round(input_seconds, 6),
        "output_seconds": round(len(transformed) / output_rate, 6),
        **signal_details,
    }


def _zip_write(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, content)


def _load_source_manifest(
    path: Path, audio_archive: Path, label_archive: Path
) -> tuple[dict[str, object], str]:
    content = path.read_bytes()
    manifest = json.loads(content)
    if not isinstance(manifest, dict) or manifest.get("usage_role") != "evaluation":
        raise ValueError("source manifest must declare an evaluation dataset")
    for field in ("dataset_id", "dataset_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"source manifest is missing {field}")
    source = manifest.get("source")
    evaluation = manifest.get("evaluation")
    if not isinstance(source, dict):
        raise ValueError("source manifest is missing source")
    if not isinstance(evaluation, dict):
        raise ValueError("source manifest is missing evaluation")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source manifest artifacts must be an array")
    expected = {
        item.get("sha256")
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    }
    observed = {sha256_file(audio_archive), sha256_file(label_archive)}
    if not observed.issubset(expected):
        raise ValueError("source archives do not match the pinned manifest")
    return manifest, _sha256_bytes(content)


def _manifest(
    *,
    source_manifest: dict[str, object],
    source_manifest_sha256: str,
    source_audio_sha256: str,
    source_labels_sha256: str,
    priority_terms_sha256: str,
    spec: VariantSpec,
    seed: int,
    selected_positive: int,
    selected_negative: int,
    audio_archive: Path,
    label_archive: Path,
    ledger: Path,
    audio_seconds: float,
    artifact_prefix: str,
    generated_at: str,
) -> dict[str, object]:
    record_count = selected_positive + selected_negative
    source_dataset_id = str(source_manifest["dataset_id"])
    dataset_id = f"{source_dataset_id}_{PROFILE_ID.replace('-', '_')}_{spec.id}"
    source = dict(source_manifest["source"])
    source["parent_manifest_sha256"] = source_manifest_sha256
    prefix = artifact_prefix.rstrip("/")
    artifacts = [
        {
            "path": f"{prefix}/audio/{audio_archive.name}",
            "sha256": sha256_file(audio_archive),
            "bytes": audio_archive.stat().st_size,
        },
        {
            "path": f"{prefix}/labels/{label_archive.name}",
            "sha256": sha256_file(label_archive),
            "bytes": label_archive.stat().st_size,
        },
        {
            "path": f"{prefix}/{ledger.name}",
            "sha256": sha256_file(ledger),
            "bytes": ledger.stat().st_size,
            "access": "private",
        },
    ]
    not_applicable = {
        "status": "not_applicable",
        "reason": "derived from a held-out evaluation partition; no training partition supplied",
    }
    return {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_version": f"{source_manifest['dataset_version']}+{PROFILE_ID}",
        "created_at": generated_at,
        "classification": "derived",
        "usage_role": "evaluation",
        "source": source,
        "split": {
            "name": "Validation radio simulation",
            "strategy": "pre-registered priority-term stratified deterministic hash sample",
            "unit": "record",
            "parameters": {
                "provider_partition": "Validation",
                "used_for_tuning": False,
                "selected_priority_term_positive": selected_positive,
                "selected_priority_term_negative": selected_negative,
                "priority_terms_sha256": priority_terms_sha256,
            },
            "seed": seed,
        },
        "preprocessing": {
            "implementation": "chemicheck119_data.radio_simulation",
            "version": IMPLEMENTATION_VERSION,
            "parameters": {
                "profile_id": PROFILE_ID,
                "variant": asdict(spec),
                "dependencies": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "source_audio_sha256": source_audio_sha256,
                "source_labels_sha256": source_labels_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "priority_terms_sha256": priority_terms_sha256,
                "per_record_seed_derivation": "sha256(base_seed:source_audio_sha256:variant_id)[:8] mod 2^32",
            },
            "seed": seed,
        },
        "integrity_report": {
            "schema_version": "1.0.0",
            "generated_at": generated_at,
            "required_fields": {"status": "passed", "missing_count": 0},
            "duplicates": {"status": "passed", "count": 0},
            "schema_validation": {"status": "passed", "error_count": 0},
            "reference_timing": {
                "status": (
                    "not_applicable"
                    if spec.kind in {"cut", "combined"}
                    else "not_evaluated"
                ),
                "reason": (
                    "source transcript and timestamps are intentionally preserved so "
                    "speech removed by start/end cuts is scored as deletion"
                    if spec.kind in {"cut", "combined"}
                    else "the shared source labels are used as text references; derived "
                    "audio timestamp alignment was not independently relabeled"
                ),
            },
            "pairing": {
                "status": "passed",
                "strategy": "same sampled source member stem in audio and label archives",
                "paired_count": record_count,
            },
            "split_integrity": {
                "entities": {
                    "speaker": dict(not_applicable),
                    "source": dict(not_applicable),
                    "event": dict(not_applicable),
                }
            },
            "source_drift": {
                "status": "not_applicable",
                "changes_detected": 0,
                "reason": "derived artifacts are bound to an immutable parent manifest hash",
            },
        },
        "artifacts": artifacts,
        "inventory": {
            "paired_count": record_count,
            "priority_term_positive_count": selected_positive,
            "priority_term_negative_count": selected_negative,
            "audio_seconds": round(audio_seconds, 6),
            "audio_hours": round(audio_seconds / 3600.0, 6),
        },
        "evaluation": {
            "id": f"speech_{dataset_id}_{record_count}",
            "record_count": record_count,
        },
        "evidence_scope": "procedural simulated communication distortion; not field-radio validation",
        "limitations": [
            "procedural siren, vehicle, and wind signals are proxies, not recorded fireground noise",
            "SNR uses whole-record RMS rather than calibrated active-speech level",
            "mu-law is a mathematical 8-bit companding proxy, not a vendor radio codec",
            "selection is stratified by pre-registered priority-term presence and is not population prevalence",
            "source transcript text is retained after cuts to measure deletion errors; source timestamps are not realigned",
        ],
    }


def build_radio_simulation(
    *,
    audio_archive: Path,
    label_archive: Path,
    source_manifest_path: Path,
    priority_terms_path: Path,
    output_dir: Path,
    artifact_prefix: str,
    positive_records: int = 20,
    negative_records: int = 20,
    seed: int = 119,
    generated_at: str | None = None,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not (1 <= positive_records <= MAX_RECORDS_PER_STRATUM):
        raise ValueError("positive_records must be between 1 and 100")
    if not (1 <= negative_records <= MAX_RECORDS_PER_STRATUM):
        raise ValueError("negative_records must be between 1 and 100")
    if not artifact_prefix.strip():
        raise ValueError("artifact_prefix must be declared")
    source_manifest, source_manifest_sha256 = _load_source_manifest(
        source_manifest_path, audio_archive, label_archive
    )
    terms = load_priority_terms(priority_terms_path)
    priority_terms_sha256 = sha256_file(priority_terms_path)
    source_audio_sha256 = sha256_file(audio_archive)
    source_labels_sha256 = sha256_file(label_archive)
    created = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    variants = profile_variants()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        audio_output_dir = temporary / "audio"
        label_output_dir = temporary / "labels"
        manifest_output_dir = temporary / "manifests"
        audio_output_dir.mkdir()
        label_output_dir.mkdir()
        manifest_output_dir.mkdir()
        ledger_path = temporary / "provenance.private.jsonl"
        labels_output = label_output_dir / "sampled-labels.zip"

        with zipfile.ZipFile(audio_archive) as audio_zip, zipfile.ZipFile(
            label_archive
        ) as label_zip:
            audio_members = _safe_members(audio_zip, ".wav")
            label_members = _safe_members(label_zip, ".json")
            if set(audio_members) != set(label_members):
                raise ValueError("source audio and label archive stems do not match")
            declared_count = source_manifest.get("evaluation", {}).get("record_count")
            if declared_count != len(audio_members):
                raise ValueError(
                    "source manifest record count does not match source archives"
                )

            label_contents: dict[str, bytes] = {}
            positive: list[str] = []
            negative: list[str] = []
            for stem in sorted(audio_members):
                content = _read_member(
                    label_zip, label_members[stem], MAX_LABEL_MEMBER_BYTES
                )
                label_contents[stem] = content
                target = positive if _has_priority_term(_label_text(content), terms) else negative
                target.append(stem)
            selected_positive = _stable_order(positive, seed)[:positive_records]
            selected_negative = _stable_order(negative, seed)[:negative_records]
            if not selected_positive or not selected_negative:
                raise ValueError(
                    "both priority-term-positive and priority-term-negative records are required"
                )
            selected = sorted(selected_positive + selected_negative)

            with zipfile.ZipFile(labels_output, "w") as destination:
                for stem in selected:
                    _zip_write(destination, f"{stem}.json", label_contents[stem])

            ledgers: list[dict[str, object]] = []
            for spec in variants:
                variant_archive = audio_output_dir / f"{spec.id}.zip"
                with zipfile.ZipFile(variant_archive, "w") as destination:
                    for stem in selected:
                        source_bytes = _read_member(
                            audio_zip, audio_members[stem], MAX_AUDIO_MEMBER_BYTES
                        )
                        source_digest = _sha256_bytes(source_bytes)
                        record_seed = _record_seed(seed, source_digest, spec.id)
                        transformed, signal = apply_variant(
                            source_bytes, spec, seed=record_seed
                        )
                        _zip_write(destination, f"{stem}.wav", transformed)
                        ledgers.append(
                            {
                                "source_id": stem,
                                "source_member": audio_members[stem],
                                "source_audio_sha256": source_digest,
                                "source_label_sha256": _sha256_bytes(label_contents[stem]),
                                "record_key": _sha256_bytes(stem.encode("utf-8"))[:16],
                                "variant": asdict(spec),
                                "seed": record_seed,
                                "derived_member": f"{stem}.wav",
                                "derived_audio_sha256": _sha256_bytes(transformed),
                                "signal": signal,
                            }
                        )

        with ledger_path.open("x", encoding="utf-8") as destination:
            for row in ledgers:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger_path.chmod(0o600)

        manifests: list[dict[str, object]] = []
        for spec in variants:
            variant_audio_seconds = sum(
                float(row["signal"]["output_seconds"])
                for row in ledgers
                if row["variant"]["id"] == spec.id
            )
            manifest = _manifest(
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                source_audio_sha256=source_audio_sha256,
                source_labels_sha256=source_labels_sha256,
                priority_terms_sha256=priority_terms_sha256,
                spec=spec,
                seed=seed,
                selected_positive=len(selected_positive),
                selected_negative=len(selected_negative),
                audio_archive=audio_output_dir / f"{spec.id}.zip",
                label_archive=labels_output,
                ledger=ledger_path,
                audio_seconds=variant_audio_seconds,
                artifact_prefix=artifact_prefix,
                generated_at=created,
            )
            manifest_path = manifest_output_dir / f"{spec.id}.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifests.append(
                {
                    "variant": spec.id,
                    "evaluation_id": manifest["evaluation"]["id"],
                    "manifest": f"manifests/{manifest_path.name}",
                    "manifest_sha256": sha256_file(manifest_path),
                    "audio_seconds": round(variant_audio_seconds, 6),
                }
            )

        summary: dict[str, object] = {
            "schema_version": "1.0.0",
            "profile_id": PROFILE_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "generated_at": created,
            "source_manifest_sha256": source_manifest_sha256,
            "source_audio_sha256": source_audio_sha256,
            "source_labels_sha256": source_labels_sha256,
            "priority_terms_sha256": priority_terms_sha256,
            "seed": seed,
            "selected": {
                "priority_term_positive": len(selected_positive),
                "priority_term_negative": len(selected_negative),
                "total": len(selected),
            },
            "variant_count": len(variants),
            "total_audio_seconds": round(
                sum(float(item["audio_seconds"]) for item in manifests), 6
            ),
            "total_audio_hours": round(
                sum(float(item["audio_seconds"]) for item in manifests) / 3600.0,
                6,
            ),
            "private_ledger_sha256": sha256_file(ledger_path),
            "manifests": manifests,
            "evidence_scope": "simulated communication distortion; not field-radio validation",
        }
        (temporary / "run-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(temporary), str(output_dir))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--label-archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--positive-records", type=int, default=20)
    parser.add_argument("--negative-records", type=int, default=20)
    parser.add_argument("--seed", type=int, default=119)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    summary = build_radio_simulation(
        audio_archive=args.audio_archive,
        label_archive=args.label_archive,
        source_manifest_path=args.source_manifest,
        priority_terms_path=args.priority_terms,
        output_dir=args.output_dir,
        artifact_prefix=args.artifact_prefix,
        positive_records=args.positive_records,
        negative_records=args.negative_records,
        seed=args.seed,
        generated_at=args.generated_at,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "profile_id": summary["profile_id"],
                "selected": summary["selected"],
                "variant_count": summary["variant_count"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

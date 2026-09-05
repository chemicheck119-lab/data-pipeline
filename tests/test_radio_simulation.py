import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

import numpy as np

from chemicheck119_data.aihub119 import sha256_file
from chemicheck119_data.radio_simulation import (
    PROFILE_ID,
    apply_variant,
    build_radio_simulation,
    profile_variants,
)


def wav_bytes(marker: int, seconds: float = 1.2, sample_rate: int = 16000) -> bytes:
    time_axis = np.arange(round(seconds * sample_rate)) / sample_rate
    samples = 0.3 * np.sin(2 * np.pi * (300 + marker * 10) * time_axis)
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(np.rint(samples * 32768).astype("<i2").tobytes())
    return output.getvalue()


def write_source(root: Path) -> tuple[Path, Path, Path, Path]:
    audio_path = root / "validation-audio.zip"
    labels_path = root / "validation-labels.zip"
    records = (
        ("positive-a", "공장에서 염산이 누출됐습니다"),
        ("positive-b", "탱크에 염산 표기가 있습니다"),
        ("negative-a", "창고에서 연기가 납니다"),
        ("negative-b", "차량 엔진에 불이 났습니다"),
    )
    with zipfile.ZipFile(audio_path, "w") as archive:
        for index, (stem, _) in enumerate(records, start=1):
            archive.writestr(f"source/{stem}.wav", wav_bytes(index))
    with zipfile.ZipFile(labels_path, "w") as archive:
        for stem, text in records:
            label = {
                "recordId": f"event-{stem}",
                "utterances": [
                    {
                        "id": "u1",
                        "startAt": 0,
                        "endAt": 1000,
                        "text": text,
                        "speaker": 1,
                    }
                ],
            }
            archive.writestr(
                f"source/{stem}.json", json.dumps(label, ensure_ascii=False)
            )
    manifest_path = root / "source-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_id": "aihub_71768_seoul_fire",
                "dataset_version": "fixture-v1",
                "created_at": "2026-09-05T00:00:00Z",
                "classification": "approved_restricted",
                "usage_role": "evaluation",
                "source": {
                    "name": "AIHub fixture",
                    "url": "https://example.invalid/aihub",
                    "license": "test-only",
                    "version": "fixture-v1",
                    "collected_at": "2026-09-05T00:00:00Z",
                },
                "evaluation": {"id": "fixture-evaluation", "record_count": 4},
                "artifacts": [
                    {"path": "private/audio.zip", "sha256": sha256_file(audio_path)},
                    {"path": "private/labels.zip", "sha256": sha256_file(labels_path)},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    terms_path = root / "terms.txt"
    terms_path.write_text("염산\n", encoding="utf-8")
    return audio_path, labels_path, manifest_path, terms_path


class RadioSimulationTest(unittest.TestCase):
    def test_profile_covers_required_distortions(self) -> None:
        identifiers = {spec.id for spec in profile_variants()}
        self.assertEqual(18, len(identifiers))
        self.assertTrue(
            {
                "clean",
                "bandlimit_8khz",
                "mulaw_8khz",
                "siren_snr20",
                "siren_snr10",
                "siren_snr0",
                "vehicle_snr10",
                "wind_snr10",
                "start_cut_300ms",
                "end_cut_300ms",
                "hard_clip_minus12dbfs",
                "gain_minus18db",
                "dropout_3x120ms",
                "combined_radio_snr10",
            }.issubset(identifiers)
        )

    def test_build_is_deterministic_and_records_private_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels, source_manifest, terms = write_source(root)
            outputs = []
            for name in ("first", "second"):
                output = root / name
                summary = build_radio_simulation(
                    audio_archive=audio,
                    label_archive=labels,
                    source_manifest_path=source_manifest,
                    priority_terms_path=terms,
                    output_dir=output,
                    artifact_prefix="gs://private/derived/radio-sim-v1",
                    positive_records=1,
                    negative_records=1,
                    seed=119,
                    generated_at="2026-09-05T01:00:00Z",
                )
                self.assertEqual(PROFILE_ID, summary["profile_id"])
                self.assertEqual(2, summary["selected"]["total"])
                self.assertEqual(18, summary["variant_count"])
                self.assertGreater(summary["total_audio_seconds"], 0)
                self.assertGreater(summary["total_audio_hours"], 0)
                outputs.append(output)

            first_hashes = {
                path.name: sha256_file(path)
                for path in (outputs[0] / "audio").glob("*.zip")
            }
            second_hashes = {
                path.name: sha256_file(path)
                for path in (outputs[1] / "audio").glob("*.zip")
            }
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(
                sha256_file(outputs[0] / "labels" / "sampled-labels.zip"),
                sha256_file(outputs[1] / "labels" / "sampled-labels.zip"),
            )

            ledger_path = outputs[0] / "provenance.private.jsonl"
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(36, len(ledger_rows))
            self.assertTrue(all(row["source_id"] for row in ledger_rows))
            self.assertTrue(all(len(row["source_audio_sha256"]) == 64 for row in ledger_rows))
            self.assertTrue(all(len(row["derived_audio_sha256"]) == 64 for row in ledger_rows))
            self.assertTrue(all(isinstance(row["seed"], int) for row in ledger_rows))
            clean_rows = [row for row in ledger_rows if row["variant"]["id"] == "clean"]
            self.assertTrue(all(row["signal"]["bit_exact_copy"] for row in clean_rows))
            with zipfile.ZipFile(audio) as source_zip, zipfile.ZipFile(
                outputs[0] / "audio" / "clean.zip"
            ) as clean_zip:
                for clean_name in clean_zip.namelist():
                    stem = Path(clean_name).stem
                    source_name = next(
                        name for name in source_zip.namelist() if Path(name).stem == stem
                    )
                    self.assertEqual(source_zip.read(source_name), clean_zip.read(clean_name))

            manifest = json.loads(
                (outputs[0] / "manifests" / "siren_snr10.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("derived", manifest["classification"])
            self.assertEqual("evaluation", manifest["usage_role"])
            self.assertEqual(2, manifest["evaluation"]["record_count"])
            self.assertFalse(manifest["split"]["parameters"]["used_for_tuning"])
            self.assertIn("not field-radio", manifest["evidence_scope"])
            self.assertGreater(manifest["inventory"]["audio_seconds"], 0)
            self.assertEqual(
                "not_evaluated",
                manifest["integrity_report"]["reference_timing"]["status"],
            )
            cut_manifest = json.loads(
                (outputs[0] / "manifests" / "start_cut_300ms.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "not_applicable",
                cut_manifest["integrity_report"]["reference_timing"]["status"],
            )

    def test_rejects_unpinned_source_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels, source_manifest, terms = write_source(root)
            source = json.loads(source_manifest.read_text(encoding="utf-8"))
            source["artifacts"][0]["sha256"] = "0" * 64
            source_manifest.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned manifest"):
                build_radio_simulation(
                    audio_archive=audio,
                    label_archive=labels,
                    source_manifest_path=source_manifest,
                    priority_terms_path=terms,
                    output_dir=root / "output",
                    artifact_prefix="gs://private/derived",
                    positive_records=1,
                    negative_records=1,
                )

            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                build_radio_simulation(
                    audio_archive=audio,
                    label_archive=labels,
                    source_manifest_path=source_manifest,
                    priority_terms_path=terms,
                    output_dir=output,
                    artifact_prefix="gs://private/derived",
                    positive_records=1,
                    negative_records=1,
                )

    def test_variant_is_seed_reproducible(self) -> None:
        source = wav_bytes(3)
        spec = next(item for item in profile_variants() if item.id == "wind_snr10")
        first, first_signal = apply_variant(source, spec, seed=42)
        second, second_signal = apply_variant(source, spec, seed=42)
        third, _ = apply_variant(source, spec, seed=43)
        self.assertEqual(first, second)
        self.assertEqual(first_signal, second_signal)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()

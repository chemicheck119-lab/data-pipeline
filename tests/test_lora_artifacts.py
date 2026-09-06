from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_data.aihub119 import DATASET_ID, DATASET_VERSION
from chemicheck119_data.lora_artifacts import build_lora_artifacts
from chemicheck119_data.training_split import build_training_split_manifest


def _wav_bytes(index: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        frames = bytearray()
        for frame in range(8000):
            value = round(6000 * math.sin(2 * math.pi * (180 + index) * frame / 8000))
            frames.extend(int(value).to_bytes(2, "little", signed=True))
        audio.writeframes(bytes(frames))
    return output.getvalue()


def _inputs(root: Path) -> dict[str, Path]:
    audio = root / "TS_fixture.zip"
    labels = root / "TL_fixture.zip"
    with zipfile.ZipFile(audio, "w") as audio_zip, zipfile.ZipFile(
        labels, "w"
    ) as label_zip:
        for index in range(10):
            stem = f"record-{index}"
            audio_zip.writestr(f"/{stem}.wav", _wav_bytes(index))
            document = {
                "_id": f"fixture-{index}",
                "audioPath": f"/{stem}.wav",
                "recordId": f"event-{index}",
                "startAt": 0,
                "endAt": 1000,
                "disasterLarge": "화재",
                "disasterMedium": "건물화재",
                "utterances": [
                    {
                        "id": f"utterance-{index}",
                        "startAt": 0,
                        "endAt": 900,
                        "text": f"연기 합성 문장 {index}",
                        "speaker": 1,
                    }
                ],
            }
            label_zip.writestr(
                f"/{stem}.json", json.dumps(document, ensure_ascii=False)
            )
    source = root / "aihub-71768-gwangju-fire-training.json"
    source_payload = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "created_at": "2026-09-05T00:00:00Z",
        "classification": "approved_restricted",
        "usage_role": "training",
        "source": {
            "name": "fixture",
            "url": "https://example.com",
            "license": "synthetic fixture",
            "version": DATASET_VERSION,
            "collected_at": "2026-09-05T00:00:00Z",
        },
        "split": {"name": "training"},
        "integrity_report": {
            "required_fields": {"status": "passed", "missing_count": 0},
            "duplicates": {"status": "passed", "count": 0},
            "schema_validation": {"status": "passed", "error_count": 0},
        },
        "artifacts": [
            {"path": f"gs://private/{audio.name}", "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()},
            {"path": f"gs://private/{labels.name}", "sha256": hashlib.sha256(labels.read_bytes()).hexdigest()},
        ],
        "inventory": {"paired_count": 10, "audio_seconds": 10.0},
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    terms = root / "speech_priority_terms_v1.txt"
    terms.write_text("# fixture\n연기\n가스\n", encoding="utf-8")
    split = root / "split-v2.json"
    split_payload = build_training_split_manifest(
        audio_archive=audio,
        label_archive=labels,
        source_manifest_path=source,
        priority_terms_path=terms,
        generated_at="2026-09-06T00:00:00Z",
    )
    split.write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "audio_archive": audio,
        "label_archive": labels,
        "source_manifest_path": source,
        "split_manifest_path": split,
        "priority_terms_path": terms,
    }


class LoraArtifactsTest(unittest.TestCase):
    def test_builds_deterministic_private_partition_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _inputs(root)
            first_dir = root / "first"
            second_dir = root / "second"
            options = {
                **inputs,
                "artifact_prefix": "gs://private/derived/lora-v1",
                "generated_at": "2026-09-06T01:00:00Z",
            }
            first = build_lora_artifacts(output_dir=first_dir, **options)
            second = build_lora_artifacts(output_dir=second_dir, **options)

            self.assertEqual(first, second)
            self.assertIs(first["automatic_training_allowed"], False)
            self.assertEqual(4, len(first["manifests"]))
            provenance = first["implementation_provenance"]
            self.assertEqual(
                {
                    "python",
                    "platform_system",
                    "zipinfo_create_system",
                    "zlib_compile",
                    "zlib_runtime",
                    "numpy",
                    "scipy",
                },
                set(provenance["dependencies"]),
            )
            self.assertEqual(
                {
                    "src/chemicheck119_data/lora_artifacts.py",
                    "src/chemicheck119_data/radio_simulation.py",
                    "src/chemicheck119_data/training_split.py",
                },
                {item["path"] for item in provenance["sources"]},
            )
            self.assertTrue(
                all(len(item["sha256"]) == 64 for item in provenance["sources"])
            )
            by_key = {
                (item["partition"], item["condition"]): item
                for item in first["manifests"]
            }
            self.assertEqual(8, by_key[("train", "clean")]["record_count"])
            self.assertEqual(2, by_key[("dev", "wind_snr0")]["record_count"])
            self.assertEqual(
                by_key[("train", "clean")]["labels_sha256"],
                by_key[("train", "wind_snr0")]["labels_sha256"],
            )
            self.assertNotEqual(
                by_key[("train", "clean")]["audio_sha256"],
                by_key[("train", "wind_snr0")]["audio_sha256"],
            )
            serialized = json.dumps(first, ensure_ascii=False)
            self.assertNotIn("합성 문장", serialized)
            self.assertNotIn("record-", serialized)

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _inputs(root)
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                build_lora_artifacts(
                    **inputs,
                    output_dir=output,
                    artifact_prefix="gs://private/derived/lora-v1",
                )

    def test_rejects_split_manifest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _inputs(root)
            payload = json.loads(
                inputs["split_manifest_path"].read_text(encoding="utf-8")
            )
            payload["inventory"]["dev"]["record_count"] = 3
            inputs["split_manifest_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
                build_lora_artifacts(
                    **inputs,
                    output_dir=root / "output",
                    artifact_prefix="gs://private/derived/lora-v1",
                )

    def test_rejects_invalid_or_predating_generated_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _inputs(root)
            for index, generated_at in enumerate(
                ("2026-09-06T01:00:00", "2026-09-04T23:59:59Z")
            ):
                with self.subTest(generated_at=generated_at), self.assertRaisesRegex(
                    ValueError, "generated_at"
                ):
                    build_lora_artifacts(
                        **inputs,
                        output_dir=root / f"output-{index}",
                        artifact_prefix="gs://private/derived/lora-v1",
                        generated_at=generated_at,
                    )


if __name__ == "__main__":
    unittest.main()

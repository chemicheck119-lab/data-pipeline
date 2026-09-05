import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_data.aihub119 import build_manifests, inspect_archive_pair


def wav_bytes(seconds: float = 0.1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * int(8000 * seconds))
    return output.getvalue()


def write_pair(root: Path, split: str, stem: str, record_id: str) -> tuple[Path, Path]:
    audio_path = root / f"{split}-audio.zip"
    label_path = root / f"{split}-labels.zip"
    with zipfile.ZipFile(audio_path, "w") as archive:
        archive.writestr(f"/{stem}.wav", wav_bytes())
    label = {
        "_id": "fixture-id",
        "audioPath": f"/{stem}.wav",
        "recordId": record_id,
        "disasterLarge": "화재",
        "disasterMedium": "건물화재",
        "utterances": [
            {
                "id": "u1",
                "startAt": 0,
                "endAt": 100,
                "text": "합성 테스트 문장",
                "speaker": 1,
            }
        ],
    }
    with zipfile.ZipFile(label_path, "w") as archive:
        archive.writestr(f"/{stem}.json", json.dumps(label, ensure_ascii=False))
    return audio_path, label_path


class AIHub119Test(unittest.TestCase):
    def test_inspects_paired_archives_without_exposing_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, labels = write_pair(Path(directory), "train", "record-a", "event-a")
            stats = inspect_archive_pair(audio, labels, "training")
            self.assertEqual(1, stats.paired_count)
            self.assertEqual(0, stats.schema_error_count)
            self.assertNotIn("record_ids", stats.public_dict())

    def test_builds_distinct_training_and_evaluation_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_audio, train_labels = write_pair(
                root, "train", "record-a", "event-a"
            )
            validation_audio, validation_labels = write_pair(
                root, "validation", "record-b", "event-b"
            )
            training, validation = build_manifests(
                training_audio=train_audio,
                training_labels=train_labels,
                validation_audio=validation_audio,
                validation_labels=validation_labels,
                artifact_prefix="gs://private-bucket/raw",
                generated_at="2026-09-05T00:00:00Z",
            )
            self.assertEqual("training", training["usage_role"])
            self.assertNotIn("evaluation", training)
            self.assertEqual("evaluation", validation["usage_role"])
            self.assertEqual(
                "speech_aihub119_gwangju_fire_validation_77",
                validation["evaluation"]["id"],
            )

    def test_rejects_cross_split_record_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_audio, train_labels = write_pair(
                root, "train", "record-a", "shared-event"
            )
            validation_audio, validation_labels = write_pair(
                root, "validation", "record-b", "shared-event"
            )
            with self.assertRaisesRegex(ValueError, "leakage"):
                build_manifests(
                    training_audio=train_audio,
                    training_labels=train_labels,
                    validation_audio=validation_audio,
                    validation_labels=validation_labels,
                    artifact_prefix="gs://private-bucket/raw",
                )

    def test_counts_label_schema_type_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels = write_pair(root, "train", "record-a", "event-a")
            invalid = {
                "_id": "fixture-id",
                "audioPath": "/record-a.wav",
                "recordId": "event-a",
                "disasterLarge": "화재",
                "disasterMedium": "건물화재",
                "utterances": [
                    {
                        "id": "u1",
                        "startAt": 0,
                        "endAt": 100,
                        "text": "합성 테스트 문장",
                        "speaker": "caller",
                    }
                ],
            }
            with zipfile.ZipFile(labels, "w") as archive:
                archive.writestr(
                    "/record-a.json", json.dumps(invalid, ensure_ascii=False)
                )
            stats = inspect_archive_pair(audio, labels, "training")
            self.assertEqual(1, stats.schema_error_count)


if __name__ == "__main__":
    unittest.main()

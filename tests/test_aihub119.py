import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_data.aihub119 import (
    _validate_stats,
    build_manifests,
    inspect_archive_pair,
)


def wav_bytes(seconds: float = 0.1, marker: str = "fixture") -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        sample = (sum(marker.encode("utf-8")) % 32767).to_bytes(2, "little")
        audio.writeframes(sample * int(8000 * seconds))
    return output.getvalue()


def write_pair(root: Path, split: str, stem: str, record_id: str) -> tuple[Path, Path]:
    audio_path = root / f"{split}-audio.zip"
    label_path = root / f"{split}-labels.zip"
    with zipfile.ZipFile(audio_path, "w") as archive:
        archive.writestr(f"/{stem}.wav", wav_bytes(marker=stem))
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
                "text": f"합성 테스트 문장 {stem}",
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
                collected_at="2026-09-04T23:59:00Z",
                generated_at="2026-09-05T00:00:00Z",
                expected_training_records=1,
                expected_validation_records=1,
            )
            self.assertEqual("training", training["usage_role"])
            self.assertNotIn("evaluation", training)
            self.assertEqual("evaluation", validation["usage_role"])
            self.assertEqual(
                "speech_aihub119_gwangju_fire_validation_77",
                validation["evaluation"]["id"],
            )

    def test_rejects_truncated_fixed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_audio, train_labels = write_pair(
                root, "train", "record-a", "event-a"
            )
            validation_audio, validation_labels = write_pair(
                root, "validation", "record-b", "event-b"
            )
            with self.assertRaisesRegex(ValueError, "training record count"):
                build_manifests(
                    training_audio=train_audio,
                    training_labels=train_labels,
                    validation_audio=validation_audio,
                    validation_labels=validation_labels,
                    artifact_prefix="gs://private-bucket/raw",
                    collected_at="2026-09-04T23:59:00Z",
                )

    def test_unchanged_baseline_reports_source_drift_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_audio, train_labels = write_pair(
                root, "train", "record-a", "event-a"
            )
            validation_audio, validation_labels = write_pair(
                root, "validation", "record-b", "event-b"
            )
            parameters = {
                "training_audio": train_audio,
                "training_labels": train_labels,
                "validation_audio": validation_audio,
                "validation_labels": validation_labels,
                "artifact_prefix": "gs://private-bucket/raw",
                "collected_at": "2026-09-04T23:59:00Z",
                "expected_training_records": 1,
                "expected_validation_records": 1,
            }
            first_training, first_validation = build_manifests(**parameters)
            second_training, second_validation = build_manifests(
                **parameters,
                baseline_training_manifest=first_training,
                baseline_validation_manifest=first_validation,
            )
            self.assertEqual(
                "passed",
                second_training["integrity_report"]["source_drift"]["status"],
            )
            self.assertEqual(
                "passed",
                second_validation["integrity_report"]["source_drift"]["status"],
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
                    collected_at="2026-09-04T23:59:00Z",
                    expected_training_records=1,
                    expected_validation_records=1,
                )

    def test_rejects_label_that_declares_another_audio_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels = write_pair(root, "train", "record-a", "event-a")
            with zipfile.ZipFile(labels, "r") as source:
                label = json.loads(source.read("/record-a.json"))
            label["audioPath"] = "/different.wav"
            with zipfile.ZipFile(labels, "w") as destination:
                destination.writestr(
                    "/record-a.json", json.dumps(label, ensure_ascii=False)
                )
            stats = inspect_archive_pair(audio, labels, "training")
            self.assertEqual(1, stats.label_audio_path_mismatches)

    def test_rejects_renamed_audio_content_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_audio, train_labels = write_pair(
                root, "train", "record-a", "event-a"
            )
            validation_audio, validation_labels = write_pair(
                root, "validation", "record-b", "event-b"
            )
            with zipfile.ZipFile(train_audio) as source:
                copied_audio = source.read("/record-a.wav")
            with zipfile.ZipFile(validation_audio, "w") as destination:
                destination.writestr("/record-b.wav", copied_audio)
            with self.assertRaisesRegex(ValueError, "audio_content_overlap"):
                build_manifests(
                    training_audio=train_audio,
                    training_labels=train_labels,
                    validation_audio=validation_audio,
                    validation_labels=validation_labels,
                    artifact_prefix="gs://private-bucket/raw",
                    collected_at="2026-09-04T23:59:00Z",
                    expected_training_records=1,
                    expected_validation_records=1,
                )

    def test_rejects_duplicate_audio_content_with_distinct_stems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio.zip"
            label_path = root / "labels.zip"
            shared_audio = wav_bytes(marker="shared")
            with zipfile.ZipFile(audio_path, "w") as archive:
                archive.writestr("/record-a.wav", shared_audio)
                # RIFF container bytes differ, but decoded PCM input is identical.
                archive.writestr("/record-b.wav", shared_audio + b"trailing-metadata")
            with zipfile.ZipFile(label_path, "w") as archive:
                for stem in ("record-a", "record-b"):
                    label = {
                        "_id": f"fixture-{stem}",
                        "audioPath": f"/{stem}.wav",
                        "recordId": f"event-{stem}",
                        "disasterLarge": "화재",
                        "disasterMedium": "건물화재",
                        "utterances": [
                            {
                                "id": "u1",
                                "startAt": 0,
                                "endAt": 100,
                                "text": f"서로 다른 문장 {stem}",
                                "speaker": 1,
                            }
                        ],
                    }
                    archive.writestr(
                        f"/{stem}.json", json.dumps(label, ensure_ascii=False)
                    )
            stats = inspect_archive_pair(audio_path, label_path, "validation")
            self.assertEqual(1, stats.duplicate_audio_contents)
            with self.assertRaisesRegex(ValueError, "duplicate_audio_contents"):
                _validate_stats(stats)

    def test_rejects_label_outside_fire_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels = write_pair(root, "train", "record-a", "event-a")
            with zipfile.ZipFile(labels, "r") as source:
                label = json.loads(source.read("/record-a.json"))
            label["disasterLarge"] = "구급"
            with zipfile.ZipFile(labels, "w") as destination:
                destination.writestr(
                    "/record-a.json", json.dumps(label, ensure_ascii=False)
                )
            stats = inspect_archive_pair(audio, labels, "training")
            self.assertEqual(1, stats.schema_error_count)

    def test_detects_truncated_wav_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels = write_pair(root, "train", "record-a", "event-a")
            truncated = wav_bytes(marker="record-a")[:-20]
            with zipfile.ZipFile(audio, "w") as archive:
                archive.writestr("/record-a.wav", truncated)
            stats = inspect_archive_pair(audio, labels, "training")
            self.assertEqual(1, stats.schema_error_count)

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

    def test_packaged_and_repository_schemas_match(self) -> None:
        repository_schema = (
            Path(__file__).parents[1] / "schemas" / "aihub119-label.schema.json"
        )
        packaged_schema = (
            Path(__file__).parents[1]
            / "src"
            / "chemicheck119_data"
            / "schemas"
            / "aihub119-label.schema"
        )
        self.assertEqual(repository_schema.read_bytes(), packaged_schema.read_bytes())


if __name__ == "__main__":
    unittest.main()

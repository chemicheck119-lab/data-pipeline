from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_data.aihub119 import DATASET_ID, DATASET_VERSION
import chemicheck119_data.training_split as training_split_module
from chemicheck119_data.training_split import build_training_split_manifest, main


def _wav_bytes(seconds: float) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * int(round(8000 * seconds)))
    return output.getvalue()


def _write_inputs(
    root: Path,
    *,
    duplicate_record: bool = False,
    split_term_across_utterances: bool = False,
) -> tuple[Path, Path, Path, Path]:
    audio = root / "TS_fixture.zip"
    labels = root / "TL_fixture.zip"
    with zipfile.ZipFile(audio, "w") as audio_zip, zipfile.ZipFile(
        labels, "w"
    ) as label_zip:
        for index in range(10):
            record_id = "record-0" if duplicate_record and index == 1 else f"record-{index}"
            utterances = [
                {
                    "id": f"utterance-{index}",
                    "startAt": 0,
                    "endAt": 900,
                    "text": f"연기 fixture 문장 {index}",
                    "speaker": 1,
                }
            ]
            if split_term_across_utterances and index == 0:
                utterances = [
                    {
                        "id": "utterance-0-a",
                        "startAt": 0,
                        "endAt": 400,
                        "text": "가",
                        "speaker": 1,
                    },
                    {
                        "id": "utterance-0-b",
                        "startAt": 500,
                        "endAt": 900,
                        "text": "스",
                        "speaker": 2,
                    },
                ]
            document = {
                "_id": f"fixture-{index}",
                "audioPath": f"/record-{index}.wav",
                "recordId": record_id,
                "status": 1,
                "startAt": 0,
                "endAt": 1000 + index,
                "disasterLarge": "화재",
                "disasterMedium": "건물화재",
                "utterances": utterances,
            }
            audio_zip.writestr(f"/record-{index}.wav", _wav_bytes(1 + index / 1000))
            label_zip.writestr(
                f"/record-{index}.json", json.dumps(document, ensure_ascii=False)
            )
    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    label_sha = hashlib.sha256(labels.read_bytes()).hexdigest()
    source = root / "training.json"
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
            {"path": f"gs://private/{audio.name}", "sha256": audio_sha},
            {"path": f"gs://private/{labels.name}", "sha256": label_sha},
        ],
        "inventory": {
            "paired_count": 10,
            "audio_seconds": round(sum(1 + index / 1000 for index in range(10)), 3),
        },
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    terms = root / "speech_priority_terms_v1.txt"
    terms.write_text("# public terms\n연기\nLPG\n가스\n", encoding="utf-8")
    return audio, labels, source, terms


class TrainingSplitTest(unittest.TestCase):
    def test_builds_deterministic_aggregate_only_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, labels, source, terms = _write_inputs(Path(directory))
            options = {
                "audio_archive": audio,
                "label_archive": labels,
                "source_manifest_path": source,
                "priority_terms_path": terms,
                "generated_at": "2026-09-06T00:00:00Z",
            }
            first = build_training_split_manifest(**options)
            second = build_training_split_manifest(**options)

            self.assertEqual(first, second)
            self.assertEqual(8, first["inventory"]["train"]["record_count"])
            self.assertEqual(2, first["inventory"]["dev"]["record_count"])
            self.assertEqual(
                2,
                first["inventory"]["dev"]["priority_term_support"]["연기"]["record_support"],
            )
            serialized = json.dumps(first, ensure_ascii=False)
            self.assertNotIn("record-", serialized)
            self.assertNotIn("fixture 문장", serialized)
            self.assertEqual(
                hashlib.sha256(Path(training_split_module.__file__).read_bytes()).hexdigest(),
                first["provenance"]["generator_source_sha256"],
            )
            self.assertEqual(
                "not_evaluated",
                first["integrity_report"]["split_integrity"]["entities"]["event"]["status"],
            )

    def test_rejects_label_archive_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, labels, source, terms = _write_inputs(Path(directory))
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["artifacts"][0]["sha256"] = "0" * 64
            source.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_training_split_manifest(
                    audio_archive=audio,
                    label_archive=labels,
                    source_manifest_path=source,
                    priority_terms_path=terms,
                )

    def test_rejects_duplicate_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, labels, source, terms = _write_inputs(
                Path(directory), duplicate_record=True
            )
            with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
                build_training_split_manifest(
                    audio_archive=audio,
                    label_archive=labels,
                    source_manifest_path=source,
                    priority_terms_path=terms,
                )

    def test_does_not_form_term_across_utterance_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, labels, source, terms = _write_inputs(
                Path(directory), split_term_across_utterances=True
            )
            result = build_training_split_manifest(
                audio_archive=audio,
                label_archive=labels,
                source_manifest_path=source,
                priority_terms_path=terms,
            )
            support = sum(
                result["inventory"][partition]["priority_term_support"]["가스"]["record_support"]
                for partition in ("train", "dev")
            )
            self.assertEqual(0, support)

    def test_cli_refuses_to_overwrite_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, labels, source, terms = _write_inputs(root)
            output = root / "split.json"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                main(
                    [
                        "--label-archive",
                        str(labels),
                        "--audio-archive",
                        str(audio),
                        "--source-manifest",
                        str(source),
                        "--priority-terms",
                        str(terms),
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()

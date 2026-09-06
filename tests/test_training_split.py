from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from chemicheck119_data.aihub119 import DATASET_ID, DATASET_VERSION
from chemicheck119_data.training_split import build_training_split_manifest, main


def _write_inputs(root: Path, *, duplicate_record: bool = False) -> tuple[Path, Path, Path]:
    labels = root / "TL_fixture.zip"
    with zipfile.ZipFile(labels, "w") as archive:
        for index in range(10):
            record_id = "record-0" if duplicate_record and index == 1 else f"record-{index}"
            document = {
                "_id": f"fixture-{index}",
                "audioPath": f"/record-{index}.wav",
                "recordId": record_id,
                "status": 1,
                "startAt": 0,
                "endAt": 1000 + index,
                "disasterLarge": "화재",
                "disasterMedium": "건물화재",
                "utterances": [
                    {
                        "id": f"utterance-{index}",
                        "startAt": 0,
                        "endAt": 900,
                        "text": f"연기 fixture 문장 {index}",
                        "speaker": 1,
                    }
                ],
            }
            archive.writestr(
                f"/record-{index}.json", json.dumps(document, ensure_ascii=False)
            )
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
        "artifacts": [{"path": f"gs://private/{labels.name}", "sha256": label_sha}],
        "inventory": {
            "paired_count": 10,
            "audio_seconds": round(sum(1 + index / 1000 for index in range(10)), 3),
        },
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    terms = root / "terms.txt"
    terms.write_text("# public terms\n연기\nLPG\n", encoding="utf-8")
    return labels, source, terms


class TrainingSplitTest(unittest.TestCase):
    def test_builds_deterministic_aggregate_only_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels, source, terms = _write_inputs(Path(directory))
            options = {
                "label_archive": labels,
                "source_manifest_path": source,
                "priority_terms_path": terms,
                "generator_revision": "a" * 40,
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
                "not_evaluated",
                first["integrity_report"]["split_integrity"]["entities"]["event"]["status"],
            )

    def test_rejects_label_archive_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels, source, terms = _write_inputs(Path(directory))
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["artifacts"][0]["sha256"] = "0" * 64
            source.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_training_split_manifest(
                    label_archive=labels,
                    source_manifest_path=source,
                    priority_terms_path=terms,
                    generator_revision="a" * 40,
                )

    def test_rejects_duplicate_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels, source, terms = _write_inputs(
                Path(directory), duplicate_record=True
            )
            with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
                build_training_split_manifest(
                    label_archive=labels,
                    source_manifest_path=source,
                    priority_terms_path=terms,
                    generator_revision="a" * 40,
                )

    def test_cli_refuses_to_overwrite_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, source, terms = _write_inputs(root)
            output = root / "split.json"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                main(
                    [
                        "--label-archive",
                        str(labels),
                        "--source-manifest",
                        str(source),
                        "--priority-terms",
                        str(terms),
                        "--generator-revision",
                        "a" * 40,
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()

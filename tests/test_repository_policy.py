import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "check_repository_policy.py"
SPEC = importlib.util.spec_from_file_location("check_repository_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class RepositoryPolicyTest(unittest.TestCase):
    def test_rejects_generated_output_even_when_small(self) -> None:
        tracked = POLICY.TrackedObject(Path("outputs/export.json"), 10)
        self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_additional_weight_formats(self) -> None:
        tracked = POLICY.TrackedObject(Path("fixture/model.gguf"), 10)
        self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_multipart_tensorflow_checkpoints(self) -> None:
        for filename in ("model.ckpt.index", "model.ckpt.data-00000-of-00001"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path("exports") / filename, 10)
                self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_keras_weight_formats(self) -> None:
        for filename in ("model.weights.h5", "model.hdf5", "model.keras"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path("checkpoints") / filename, 10)
                self.assertTrue(POLICY.violations([tracked]))

    def test_allows_generated_root_readmes(self) -> None:
        tracked = POLICY.TrackedObject(Path("data/raw/README.md"), 10)
        self.assertEqual([], POLICY.violations([tracked]))

    def test_rejects_nested_readme_under_generated_root(self) -> None:
        tracked = POLICY.TrackedObject(Path("data/raw/private/README.md"), 10)
        self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_nested_output_directory(self) -> None:
        tracked = POLICY.TrackedObject(Path("experiments/outputs/export.json"), 10)
        self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_environment_secret_but_allows_example(self) -> None:
        for filename in (".env", ".env.production", "config/.env.local"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path(filename), 10)
                self.assertTrue(POLICY.violations([tracked]))
        example = POLICY.TrackedObject(Path(".env.example"), 10)
        self.assertEqual([], POLICY.violations([example]))

    def test_rejects_common_credential_paths(self) -> None:
        for filename in ("config/id_rsa", "credentials.yml", "keys/service.p12"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path(filename), 10)
                self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_raw_audio_outside_data_roots(self) -> None:
        tracked = POLICY.TrackedObject(Path("fixtures/call.wav"), 10)
        self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_source_documents_and_archives(self) -> None:
        for filename in ("fixtures/kosha_dump.pdf", "exports/source.zip"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path(filename), 10)
                self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_json_corpus_outside_allowlisted_paths(self) -> None:
        for filename in ("corpora/export.json", "corpora/export.fixture.json"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path(filename), 10)
                self.assertTrue(POLICY.violations([tracked]))

    def test_rejects_common_binary_corpus_formats(self) -> None:
        for filename in ("data.npy", "data.npz", "data.pkl", "data.arrow"):
            with self.subTest(filename=filename):
                tracked = POLICY.TrackedObject(Path("corpora") / filename, 10)
                self.assertTrue(POLICY.violations([tracked]))

    def test_commit_scan_keeps_every_path_for_identical_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            self._git(repository, "config", "user.name", "Policy Test")
            self._git(
                repository,
                "config",
                "user.email",
                "policy" + "@example.invalid",
            )

            (repository / "README.md").write_text("base\n")
            self._git(repository, "add", "README.md")
            self._git(repository, "commit", "-m", "base")
            base = self._git(repository, "rev-parse", "HEAD").stdout.strip()

            (repository / "allowed.txt").write_text("same bytes\n")
            restricted = repository / "data" / "raw" / "leak.dat"
            restricted.parent.mkdir(parents=True)
            restricted.write_text("same bytes\n")
            self._git(repository, "add", "allowed.txt", "data/raw/leak.dat")
            self._git(repository, "commit", "-m", "add duplicate blobs")
            os.remove(restricted)
            self._git(repository, "add", "-u")
            self._git(repository, "commit", "-m", "remove restricted file")
            self._git(
                repository,
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{base},data/raw/corpus",
            )
            self._git(repository, "commit", "-m", "add forbidden gitlink")

            objects = POLICY.commit_range_objects(base, repository)
            paths = {tracked.path for tracked in objects}
            self.assertIn(Path("allowed.txt"), paths)
            self.assertIn(Path("data/raw/leak.dat"), paths)
            self.assertIn(Path("data/raw/corpus"), paths)
            self.assertTrue(POLICY.violations(objects, repository))

            all_history_objects = POLICY.reachable_commit_objects(repository)
            self.assertIn(
                Path("data/raw/leak.dat"),
                {tracked.path for tracked in all_history_objects},
            )
            self.assertTrue(POLICY.violations(all_history_objects, repository))

    def test_index_scan_keeps_staged_file_missing_from_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            restricted = repository / "data" / "raw" / "leak.dat"
            restricted.parent.mkdir(parents=True)
            restricted.write_text("restricted\n")
            self._git(repository, "add", "-f", "data/raw/leak.dat")
            os.remove(restricted)

            objects = POLICY.current_tree_objects(repository)
            self.assertEqual([Path("data/raw/leak.dat")], [item.path for item in objects])
            self.assertTrue(POLICY.violations(objects, repository))

    def test_scans_blob_contents_for_private_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            secret = repository / "config.txt"
            begin_marker = "-----BEGIN " + "PRIVATE KEY-----\n"
            end_marker = "-----END " + "PRIVATE KEY-----\n"
            secret.write_text(
                begin_marker + "not-a-real-key\n" + end_marker
            )
            self._git(repository, "add", "config.txt")

            objects = POLICY.current_tree_objects(repository)
            errors = POLICY.violations(objects, repository)
            self.assertTrue(any("private key" in error for error in errors))

    def test_scans_blob_contents_for_pgp_private_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            begin_marker = "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----\n"
            secret = repository / "key.asc"
            secret.write_text(begin_marker + "not-a-real-key\n")
            self._git(repository, "add", "key.asc")

            objects = POLICY.current_tree_objects(repository)
            errors = POLICY.violations(objects, repository)
            self.assertTrue(any("private key" in error for error in errors))

    def test_scans_blob_contents_for_signed_download_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            signed_url = (
                "https://example.invalid/object?X-Amz-"
                + "Signature="
                + ("a" * 64)
            )
            record = repository / "download.txt"
            record.write_text(signed_url)
            self._git(repository, "add", "download.txt")

            objects = POLICY.current_tree_objects(repository)
            errors = POLICY.violations(objects, repository)
            self.assertTrue(any("signed download URL" in error for error in errors))

    def test_rejects_manifest_without_required_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            manifest = repository / "data" / "manifests" / "bad.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"records": []}))
            self._git(repository, "add", "data/manifests/bad.json")

            objects = POLICY.current_tree_objects(repository)
            errors = POLICY.violations(objects, repository)
            self.assertTrue(any("schema_version" in error for error in errors))
            self.assertTrue(any("source" in error for error in errors))
            self.assertTrue(any("artifacts" in error for error in errors))

    def test_accepts_manifest_with_required_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            manifest = repository / "data" / "manifests" / "valid.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "dataset_id": "example",
                        "dataset_version": "2026-09-05",
                        "created_at": "2026-09-05T00:00:00Z",
                        "classification": "synthetic",
                        "source": {
                            "name": "generated fixture",
                            "url": "https://example.invalid/dataset",
                            "license": "CC0-1.0",
                            "version": "fixture-v1",
                            "collected_at": "2026-09-05T00:00:00Z",
                        },
                        "split": {
                            "name": "test",
                            "strategy": "fixed fixture",
                            "unit": "record",
                            "parameters": {},
                            "seed": None,
                        },
                        "generation": {
                            "implementation": "tests.generate_fixture",
                            "version": "1.0.0",
                            "parameters": {},
                            "seed": 119,
                        },
                        "integrity_report": {
                            "schema_version": "1.0.0",
                            "generated_at": "2026-09-05T00:00:00Z",
                            "required_fields": {
                                "status": "passed",
                                "missing_count": 0,
                            },
                            "duplicates": {"status": "passed", "count": 0},
                            "split_integrity": {
                                "entities": {
                                    "speaker": {
                                        "status": "not_applicable",
                                        "reason": "non-speech fixture",
                                    },
                                    "source": {
                                        "status": "not_applicable",
                                        "reason": "single generated source",
                                    },
                                    "event": {
                                        "status": "not_applicable",
                                        "reason": "no incident events",
                                    },
                                },
                            },
                            "source_drift": {
                                "status": "not_applicable",
                                "changes_detected": 0,
                            },
                        },
                        "artifacts": [
                            {
                                "path": "fixtures/example.json",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                )
            )
            self._git(repository, "add", "data/manifests/valid.json")

            objects = POLICY.current_tree_objects(repository)
            self.assertEqual([], POLICY.violations(objects, repository))

    def test_derived_manifest_requires_preprocessing_recipe(self) -> None:
        errors = POLICY.recipe_errors(None, "manifest preprocessing")
        self.assertTrue(any("must be an object" in error for error in errors))

    def test_rejects_overwriting_published_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            self._git(repository, "config", "user.name", "Policy Test")
            self._git(
                repository,
                "config",
                "user.email",
                "policy" + "@example.invalid",
            )
            manifest = repository / "data" / "manifests" / "published.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n")
            self._git(repository, "add", "data/manifests/published.json")
            self._git(repository, "commit", "-m", "publish manifest")
            base = self._git(repository, "rev-parse", "HEAD").stdout.strip()
            manifest.write_text('{"changed": true}\n')
            self._git(repository, "add", "data/manifests/published.json")
            self._git(repository, "commit", "-m", "overwrite manifest")

            errors = POLICY.append_only_manifest_errors(base, repository)
            self.assertTrue(any("append-only" in error for error in errors))
            all_history_errors = POLICY.append_only_manifest_errors(None, repository)
            self.assertTrue(
                any("append-only" in error for error in all_history_errors)
            )

    def test_rejects_modifying_manifest_added_earlier_in_same_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            self._git(repository, "config", "user.name", "Policy Test")
            self._git(
                repository,
                "config",
                "user.email",
                "policy" + "@example.invalid",
            )
            (repository / "README.md").write_text("base\n")
            self._git(repository, "add", "README.md")
            self._git(repository, "commit", "-m", "base")
            base = self._git(repository, "rev-parse", "HEAD").stdout.strip()

            manifest = repository / "data" / "manifests" / "new.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n")
            self._git(repository, "add", "data/manifests/new.json")
            self._git(repository, "commit", "-m", "add manifest")
            manifest.write_text('{"changed": true}\n')
            self._git(repository, "add", "data/manifests/new.json")
            self._git(repository, "commit", "-m", "modify new manifest")

            errors = POLICY.append_only_manifest_errors(base, repository)
            self.assertTrue(any("append-only" in error for error in errors))

    def test_rejects_fixture_without_metadata(self) -> None:
        tracked = POLICY.TrackedObject(Path("fixtures/users.csv"), 10)
        errors = POLICY.violations([tracked])
        self.assertTrue(any("missing companion" in error for error in errors))

    def test_rejects_orphan_fixture_metadata(self) -> None:
        tracked = POLICY.TrackedObject(Path("fixtures/export.fixture.json"), 10)
        errors = POLICY.violations([tracked])
        self.assertTrue(any("orphan fixture metadata" in error for error in errors))

    def test_accepts_synthetic_fixture_with_matching_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            fixture = repository / "fixtures" / "sample.csv"
            fixture.parent.mkdir(parents=True)
            fixture.write_text("chemical,count\nwater,1\n")
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            metadata = fixture.with_name(fixture.name + ".fixture.json")
            metadata.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "classification": "synthetic",
                        "contains_personal_data": False,
                        "license": "CC0-1.0",
                        "source": {
                            "name": "unit-test generator",
                            "url": "https://example.invalid/generator",
                            "version": "1.0.0",
                            "collected_at": "2026-09-05T00:00:00Z",
                        },
                        "sha256": digest,
                        "generation": {
                            "implementation": "tests.generate_fixture",
                            "version": "1.0.0",
                            "parameters": {},
                            "seed": 119,
                        },
                    }
                )
            )
            self._git(repository, "add", "fixtures")

            objects = POLICY.current_tree_objects(repository)
            self.assertEqual([], POLICY.violations(objects, repository))

    def test_accepts_synthetic_audio_fixture_with_matching_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            fixture = repository / "fixtures" / "tone.wav"
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(b"RIFF-synthetic-test-audio")
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            metadata = fixture.with_name(fixture.name + ".fixture.json")
            metadata.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "classification": "synthetic",
                        "contains_personal_data": False,
                        "license": "CC0-1.0",
                        "source": {
                            "name": "unit-test tone generator",
                            "url": "https://example.invalid/generator",
                            "version": "1.0.0",
                            "collected_at": "2026-09-05T00:00:00Z",
                        },
                        "sha256": digest,
                        "generation": {
                            "implementation": "tests.generate_tone",
                            "version": "1.0.0",
                            "parameters": {"frequency_hz": 440},
                            "seed": 119,
                        },
                    }
                )
            )
            self._git(repository, "add", "fixtures")

            objects = POLICY.current_tree_objects(repository)
            self.assertEqual([], POLICY.violations(objects, repository))

    def test_rejects_personal_data_pattern_in_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init", "-b", "main")
            fixture = repository / "fixtures" / "users.csv"
            fixture.parent.mkdir(parents=True)
            email = "person" + "@example.org"
            fixture.write_text("email\n" + email + "\n")
            self._git(repository, "add", "fixtures/users.csv")

            objects = POLICY.current_tree_objects(repository)
            errors = POLICY.violations(objects, repository)
            self.assertTrue(any("email address" in error for error in errors))

    @staticmethod
    def _git(repository: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()

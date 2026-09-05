"""Create an append-only manifest for one AIHub validation archive pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aihub119 import build_evaluation_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-audio", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--collected-at", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--expected-validation-records", type=int)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.expected_validation_records is not None and (
        args.expected_validation_records <= 0
    ):
        parser.error("--expected-validation-records must be positive")
    baseline = (
        json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
        if args.baseline_manifest
        else None
    )
    manifest = build_evaluation_manifest(
        validation_audio=args.validation_audio,
        validation_labels=args.validation_labels,
        artifact_prefix=args.artifact_prefix,
        collected_at=args.collected_at,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        generated_at=args.generated_at,
        expected_validation_records=args.expected_validation_records,
        baseline_manifest=baseline,
    )
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite append-only manifest: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evaluation_id": manifest["evaluation"]["id"],
                "validation_records": manifest["inventory"]["paired_count"],
                "output_file": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

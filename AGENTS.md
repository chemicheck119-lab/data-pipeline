# AGENTS.md

## Scope

These instructions apply to the entire data-pipeline repository.

## Boundaries

- Store collection code, schemas, manifests, hashes, licenses, and only fixtures that pass the companion-metadata policy. Never commit downloaded corpora, raw incident audio, personal information, source-document dumps, or model weights.
- A facility record is historical public evidence, not proof of current inventory.
- Synthetic or augmented data must remain explicitly labeled and must never be reported as real field validation.

## Review Rules

- Flag missing provenance, license, collection date, source version, checksum, schema version, or deterministic preprocessing parameters.
- Flag train/dev/test leakage, speaker/source/event overlap, duplicate records across splits, and evaluation data used for tuning.
- Flag changes that overwrite historical manifests or make published evaluations irreproducible.
- Flag metrics or dataset names that conflate the 419-record Resolver locked evaluation with the 442-record Parser external evaluation.
- Flag secrets, access tokens, signed download URLs, personal data, large binaries, and raw restricted data.
- Require validation reports for schema, required fields, duplicates, split integrity, and source drift.

## Validation

- Run `python scripts/check_repository_policy.py`.

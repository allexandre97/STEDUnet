# Repository Guidelines

## Project Structure

- `Papers/manifest.yaml` is the authoritative catalog for cite keys, exact PDF filenames, canonical bibliographic metadata, paper types, and SHA-256 checksums.
- `Papers/` contains only the source literature PDFs listed in the manifest. PDF filenames do not need to match cite keys, but they must match the manifest exactly.
- `docs/literature/knowledge_base_brief.md` defines the scientific scope, extraction priorities, synthesis questions, and working hypotheses.
- `docs/literature/knowledge_base_extraction_schema.md` defines the structured output contract, controlled vocabularies, evidence model, and validation requirements.
- `knowledge_base/paper_cards.jsonl` is the primary structured knowledge-base artifact.
- `knowledge_base/synthesis_claims.jsonl` is the structured source for cross-paper claims.
- `knowledge_base/comparison_table.csv`, `knowledge_base/synthesis.md`, and `knowledge_base/kb_index.md` are generated from the structured JSONL sources.
- `scripts/` contains validation and artifact-generation utilities.
- `environment.yml` is the minimal portable environment for this workflow; it is not the future model-training environment.

Do not commit temporary PDF conversions, extracted page images, editor backups, caches, or ad hoc intermediate files.

## Literature-Grounded Work

Before extracting papers, modifying knowledge-base artifacts, or making literature-supported decisions about architecture, losses, supervision, skeletonization, tracing, annotation, or evaluation, read:

1. `docs/literature/knowledge_base_brief.md`
2. `docs/literature/knowledge_base_extraction_schema.md`
3. `Papers/manifest.yaml`

Apply these rules:

- Treat the brief as scientific intent and the extraction schema as the output contract.
- Treat the brief's working hypotheses as questions to test, not conclusions to reproduce.
- Verify paper-specific facts against the corresponding PDF.
- Do not copy technical values from examples, summaries, or previous outputs without PDF evidence.
- Use the schema's approved missing-value tokens; never guess absent hyperparameters, dataset sizes, splits, results, or architecture details.
- Keep reported methods, reported results, author claims, author-stated limitations, review synthesis, and project-specific inference distinct.
- Preserve contradictory, negative, ambiguous, and insufficient evidence.
- Preserve important paper terminology in raw-value or notes fields rather than forcing an inaccurate controlled term.
- Do not mark a paper card `reviewed` or `verified` until the required PDF second pass is complete and documented in provenance.

For tasks unrelated to literature evidence, do not load or restate the full literature specifications unnecessarily.

## Source and Artifact Rules

- Treat cite keys as opaque stable identifiers. Do not infer publication years from them.
- Treat all eight manifest entries, including `Togo26`, as the core corpus. Keep this status consistent across cards and generated artifacts.
- Verify that each PDF's exact filename and checksum match the manifest before extraction.
- Treat `paper_cards.jsonl` as the source of truth for paper-level content and `synthesis_claims.jsonl` as the source for cross-paper claims.
- Regenerate the CSV, synthesis, and index rather than maintaining independent conflicting copies.
- Every nontrivial reported fact or paper-attributed claim must have page-linked evidence as required by the schema.
- Every project recommendation or STED-transfer judgment must be labelled as agent inference and cite its evidential basis.
- Do not add or replace a source paper without updating the manifest and reviewing the corpus scope.
- For synthetic STED work, treat `data_manifests/*.csv` as authoritative for source identity, culture/condition/isoform/DIV metadata, split eligibility, expert-validated blank provenance, and blank-pool roles. `PN###` is `culture_id`, `3R`/`4R` are `tau_isoform`, and primary development splits use manifest `experimental_group_id` rather than reparsing filenames. Downstream code must read these manifests, and normal tests must not require local `/ssd` paths.
- Preserve the normalized 3D rendering contract unless a concrete defect is found: fluorophore values are empirical line density per pixel-equivalent contour length, PSF kernels are discrete unit-integral over truncated support, structural targets are independent of optical scaling, and synthetic samples must record annotation-compatible target availability and projected trace arrays.
- For 3D synthetic artifacts, keep QA fixtures separate from realism-calibration samples. Real-versus-synthetic appearance reports must use only `scenario_category: realism_calibration` unless an override is explicit and documented.

## Commands

Run commands from the repository root.

Basic repository checks:

```bash
rg --files
rg '^#{1,4} ' docs/literature/*.md
sha256sum Papers/*.pdf
git diff --check
```

Create or update the portable environment with `conda env create -f environment.yml` or `conda env update -f environment.yml`, then validate with:

```bash
python -m pytest -q
python scripts/build_knowledge_base.py --check
python scripts/validate_knowledge_base.py
git diff --check
```

Do not claim completion when a required command fails. Report the failure and its likely cause.

## Style and Conventions

- Use ATX Markdown headings and short, direct paragraphs.
- Use fenced code blocks for commands and tables for repeated structured fields.
- Preserve cite-key capitalization exactly.
- Use `snake_case` for generated filenames and JSON fields.
- Use two-space YAML indentation and never tabs.
- Write JSONL as one compact, valid JSON object per line.
- Do not use empty strings where the schema requires an explicit missing-value token.

For Python utilities:

- Prefer the standard library unless a dependency provides a clear benefit.
- Include a CLI entry point, useful error messages, and a nonzero exit status on validation failure.
- Keep generation deterministic: stable ordering, explicit UTF-8 encoding, and no timestamps in derived content unless required by the schema.

## Verification and Definition of Done

A literature knowledge-base change is complete only when:

- manifest entries match existing PDFs and checksums;
- exactly one valid paper card exists per manifest entry;
- required reported fields have valid evidence links;
- inference and review synthesis are correctly labelled;
- the CSV is consistent with the JSONL;
- the synthesis answers the required questions and assesses every working hypothesis;
- generated artifacts are byte-identical after rebuilding;
- `kb_index.md` reports extraction status and validation information;
- the knowledge-base validator exits successfully;
- `git diff --check` exits successfully.

A code change is complete only when the relevant tests or validation commands pass and the final response identifies the files changed and checks performed.

## Commit and Pull Request Guidelines

Use concise imperative commit subjects, for example:

```text
Add manifest metadata for Liu18
Validate evidence links for Shi20b
```

Keep commits focused on one source, schema change, validator change, or coherent generated-artifact update.

Pull requests should:

- summarize the scope;
- list affected cite keys and artifacts;
- describe validation performed;
- call out schema changes, inferred claims, unresolved source conflicts, and added or replaced PDFs;
- avoid mixing unrelated model-development and literature-extraction changes.

# Knowledge base extraction schema

**Document role:** extraction and validation contract
**Status:** working specification
**Companion brief:** `knowledge_base_brief.md`
**Executable schema:** `knowledge_base.schema.json`
**Schema version:** `1.2.0`

## 1. Authority and precedence

Use the sources in this order:

1. `Papers/manifest.yaml` for corpus membership and bibliographic identity;
2. each source PDF for methods, results, and author claims;
3. this document for extraction and derivation rules;
4. `knowledge_base.schema.json` for machine-enforced structure and vocabularies;
5. the brief for scientific questions and extraction priorities.

If this prose conflicts with the executable schema, stop and reconcile the two before extraction. Cite keys are opaque identifiers; never infer publication year from them.

## 2. Required sources and outputs

Read the manifest before opening a paper. Create exactly one card per manifest entry.

```text
knowledge_base/
├── paper_cards.jsonl
├── synthesis_claims.jsonl
├── comparison_table.csv
├── synthesis.md
└── kb_index.md
```

`paper_cards.jsonl` is the source for paper-level facts. `synthesis_claims.jsonl` is the source for cross-paper assertions. The CSV, synthesis Markdown, and index are deterministic generated artifacts.

## 3. Manifest contract

Every paper entry includes:

- `cite_key`, `corpus_role: core`, exact `filename`, canonical `title`, and integer `publication_year`;
- `doi`, using `not_reported` only when no DOI is present;
- optional persistent identifiers such as `arxiv`;
- `study_type`, `method_paradigms`, and `publication_status`;
- source version, PDF page count, and lowercase SHA-256.

The validator fails for duplicate cite keys, unlisted or missing PDFs, checksum or page-count mismatches, and card membership that differs from the manifest.

## 4. Missing values

Use:

- `not_reported`: relevant source locations were checked and the information was absent;
- `unknown`: the available source does not permit a determination;
- `not_applicable`: the field does not apply;
- `[]`: an applicable collection was checked and contains no items;
- `null`: only where the executable schema permits it.

Do not use empty strings. Every non-bibliographic `not_reported` value requires an entry in `missing_field_checks` with the exact field path and the sections, figures, tables, or appendices checked.

Never infer missing hyperparameters, dataset sizes, splits, architecture details, or numerical results.

## 5. Claim origin and evidence representation

`claim_class` records where a statement originates:

```text
paper_claim
reported_method
reported_result
author_stated_limitation
review_synthesis
agent_inference
```

`evidence_type` records how the source was represented:

```text
direct_quote
close_paraphrase
figure_reading
table_reading
equation_reading
```

These dimensions are independent. A close paraphrase does not change a paper result into an inference. Close paraphrases must preserve scope and meaning without adding interpretation.

All project recommendations, comparisons, transfer judgments, and statements about suitability for STED are `agent_inference`. Review summaries of underlying studies are `review_synthesis`, not direct experimental evidence.

## 6. Paper-card contract

Each JSONL line is one compact JSON object with these top-level fields:

```text
schema_version
cite_key
bibliography
provenance
scope
method
evaluation
paper_findings
agent_assessment
comparison_summary
missing_field_checks
field_evidence
evidence_spans
```

Important rules:

- bibliography, core-corpus role, and provenance must match the manifest;
- every evidence-bearing object has `claim_class` and `evidence_ids`;
- normalized values retain source terminology in `raw_value`;
- `other` requires a specific, nonempty `raw_value`;
- every `agent_inference` cites the evidence on which it is based;
- `extracted_on` is `not_recorded` when the original extraction date was not captured honestly;
- `reviewed_on`, `reviewer`, and `review_scope` document the second pass;
- paper cards remain `draft` after initial extraction;
- `reviewed` requires a second pass through methods, figures, tables, results, appendices or supplements, and limitations;
- `verified` requires independent rechecking of locators and key numerical values.

`comparison_summary` is the sole source for paper-specific CSV columns. It must not introduce information absent elsewhere in the card.

## 7. Evidence spans

Each evidence span contains:

```text
evidence_id
source_pdf
page_pdf
page_printed
section
figure_or_table
locator
evidence_type
text
claim_class
supports_fields
confidence
notes
```

Rules:

- IDs use `<cite_key>-E###` and are globally unique;
- `page_pdf` is one-indexed and cannot exceed the manifest page count;
- figure, table, and equation readings identify the corresponding object;
- quotations are short and used only when exact wording matters;
- one span may support multiple fields only when it genuinely supports each;
- field and inference references must resolve to an evidence span in the same card.

## 8. Paper-specific checks

Before completing each draft, check:

- `Liu18`: architecture, patch size, training recipe, annotation process, split, and metrics;
- `Shi20b`: clDice definition, topology assumptions, soft skeletonization, combined objective, datasets, and limitations;
- `Tet18`: structural targets, class balancing, patch size, synthetic pretraining, real finetuning, and subsampling findings;
- `Liu20`: keypoint targets, synthetic labels, detector outputs, losses, fast marching, and evaluation limitations;
- `Liu19`: orientation bins, synthetic generation, overlap loss, terminus pairing, instance metrics, and failure cases;
- `Xu15`: active-contour stages, inputs, outputs, parameter sensitivity, manual editing, and metrics;
- `Ozd21`: review scope, method taxonomy, microscopy constraints, annotation burden, and the boundary between review synthesis and primary evidence;
- `Togo26`: preprint status, preprocessing, DMT and persistence, skeleton refinement, graph conversion, synthetic and collagen validation, parameters, and failure modes.

The priorities are questions to check, not values that must be found.

## 9. Structured synthesis

Each line of `synthesis_claims.jsonl` contains:

```text
claim_id
section
title
claim
claim_class
supporting_evidence_ids
limiting_evidence_ids
supporting_papers
limiting_papers
confidence
sted_implication
```

Question records also contain `question_number`. Hypothesis records contain `hypothesis_id` and one of:

```text
supported
partially_supported
contradicted
insufficient_evidence
```

Write evidence-neutral findings before assessing the brief's hypotheses. Do not cite a paper because it is merely related; cite evidence that supports the specific assertion. Contradicting, negative, ambiguous, and insufficient evidence must remain visible.

## 10. Derived artifacts

Generate artifacts with:

```bash
python scripts/build_knowledge_base.py
```

The CSV uses this exact header:

```text
cite_key,corpus_role,study_type,method_paradigms,task_focus,architecture_family,dimensionality,main_targets,losses,data_strategy,postprocessing,evaluation_focus,direct_sted_evidence,best_use_case,main_transfer_risk,extraction_status
```

Multi-value cells use semicolons. Manifest order controls row order. `synthesis.md` is rendered from structured claims, and `kb_index.md` is rendered from the manifest, card statuses, hashes, and validation metadata.

Generation must use UTF-8, stable ordering, and stable line endings. Repeated builds from unchanged structured sources must be byte-identical.

## 11. Validation

Run:

```bash
python scripts/validate_knowledge_base.py
```

The validator checks:

- manifest validity, PDF presence, checksums, page counts, and cite-key uniqueness;
- one schema-compliant card per manifest entry;
- controlled values, explicit missing values, and checked locations;
- evidence-ID uniqueness, source PDF, page range, and reference integrity;
- evidence support for agent inference and review-statement classification;
- exact JSONL-to-CSV projection;
- all ten synthesis questions and all seven hypotheses;
- synthesis cite keys and evidence IDs;
- deterministic generated artifacts.

Errors are actionable and produce a nonzero exit status.

## 12. Prohibited shortcuts

Do not:

- copy technical values from examples, summaries, or generated synthesis;
- use an abstract alone when methods, figures, tables, or appendices provide the detail;
- infer values from common practice or cite-key names;
- turn review statements into primary experimental evidence;
- force an inaccurate controlled term;
- omit weak, negative, conflicting, or failed results;
- describe project inference as an author conclusion;
- edit generated CSV or Markdown independently of their structured sources;
- mark a card verified after a single extraction pass.

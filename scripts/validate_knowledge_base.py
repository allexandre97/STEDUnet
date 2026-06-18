#!/usr/bin/env python3
"""Validate the literature corpus, paper cards, evidence, and derived artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from build_knowledge_base import ROOT, build_contents, comparison_row, read_jsonl


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_type(value: object, expected: str) -> bool:
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    return checks[expected]()


def resolve_ref(root: dict, ref: str) -> dict:
    value = root
    for part in ref.removeprefix("#/").split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def schema_errors(value: object, schema: dict, root: dict, path: str = "$") -> list[str]:
    if "$ref" in schema:
        return schema_errors(value, resolve_ref(root, schema["$ref"]), root, path)
    if "oneOf" in schema:
        matches = [not schema_errors(value, option, root, path) for option in schema["oneOf"]]
        return [] if sum(matches) == 1 else [f"{path}: must match exactly one allowed shape"]
    errors = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: invalid value {value!r}")
    expected = schema.get("type")
    if expected:
        allowed = expected if isinstance(expected, list) else [expected]
        if not any(json_type(value, item) for item in allowed):
            return [f"{path}: expected {' or '.join(allowed)}, got {type(value).__name__}"]
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors += schema_errors(item, properties[key], root, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors += schema_errors(
                    item, schema["additionalProperties"], root, f"{path}.{key}"
                )
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: requires at least {schema['minItems']} items")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            errors.append(f"{path}: items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                errors += schema_errors(item, schema["items"], root, f"{path}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string is too short")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be at most {schema['maximum']}")
    return errors


def walk(value: object, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")


def pdf_pages(path: Path) -> int | None:
    try:
        output = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def validate(args: argparse.Namespace) -> list[str]:
    errors = []
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    papers = manifest.get("papers", [])
    keys = [paper.get("cite_key") for paper in papers]
    if manifest.get("schema_version") != "1.2.0":
        errors.append("manifest: schema_version must be 1.2.0")
    if len(papers) != 8:
        errors.append(f"manifest: eight core papers required, found {len(papers)}")
    if len(keys) != len(set(keys)):
        errors.append("manifest: duplicate cite keys")
    for paper in papers:
        if paper.get("corpus_role") != "core":
            errors.append(f"{paper.get('cite_key', 'unknown')}: corpus_role must be core")
    manifest_by_key = {paper["cite_key"]: paper for paper in papers}
    listed_files = {paper["filename"] for paper in papers}
    actual_files = {path.name for path in args.manifest.parent.glob("*.pdf")}
    for filename in sorted(actual_files - listed_files):
        errors.append(f"manifest: unlisted PDF {filename}")
    for filename in sorted(listed_files - actual_files):
        errors.append(f"manifest: missing PDF {filename}")
    for paper in papers:
        path = args.manifest.parent / paper["filename"]
        if not path.exists():
            continue
        if digest(path) != paper["sha256"]:
            errors.append(f"{paper['cite_key']}: PDF checksum mismatch")
        pages = pdf_pages(path)
        if pages is not None and pages != paper["pages"]:
            errors.append(f"{paper['cite_key']}: expected {paper['pages']} PDF pages, found {pages}")

    cards = read_jsonl(args.jsonl)
    card_keys = [card.get("cite_key") for card in cards]
    if len(card_keys) != len(set(card_keys)):
        errors.append("paper_cards.jsonl: duplicate cite keys")
    if set(card_keys) != set(keys):
        errors.append(
            f"paper_cards.jsonl: cite keys differ from manifest "
            f"(missing={sorted(set(keys) - set(card_keys))}, extra={sorted(set(card_keys) - set(keys))})"
        )

    all_evidence = {}
    for line, card in enumerate(cards, 1):
        key = card.get("cite_key", f"line-{line}")
        errors += [f"{key}: {item}" for item in schema_errors(card, schema, schema)]
        for path, value in walk(card):
            if value == "":
                errors.append(f"{key}: empty string at {path}")
            if isinstance(value, dict) and value.get("normalized") == "other":
                if value.get("raw_value") in (None, "", "other", "not_reported"):
                    errors.append(f"{key}: normalized 'other' needs a specific raw_value at {path}")
        if key not in manifest_by_key:
            continue
        paper = manifest_by_key[key]
        bibliography = card.get("bibliography", {})
        expected_bibliography = {
            field: paper[field]
            for field in (
                "title",
                "publication_year",
                "doi",
                "corpus_role",
                "study_type",
                "method_paradigms",
                "publication_status",
            )
        }
        if bibliography != expected_bibliography:
            errors.append(f"{key}: bibliography differs from manifest")
        provenance = card.get("provenance", {})
        if provenance.get("source_pdf") != paper["filename"]:
            errors.append(f"{key}: provenance source_pdf differs from manifest")
        if provenance.get("source_sha256") != paper["sha256"]:
            errors.append(f"{key}: provenance source_sha256 differs from manifest")
        if provenance.get("extracted_on") != "not_recorded":
            errors.append(f"{key}: original extraction date must be recorded honestly as not_recorded")
        if provenance.get("extraction_status") in {"reviewed", "verified"}:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(provenance.get("reviewed_on", ""))):
                errors.append(f"{key}: reviewed status requires an ISO review date")
            if not provenance.get("reviewer"):
                errors.append(f"{key}: reviewed status requires a reviewer")
            required_scope = {
                "methods",
                "figures",
                "tables",
                "results",
                "appendices_or_supplements",
                "limitations",
            }
            if set(provenance.get("review_scope", [])) != required_scope:
                errors.append(f"{key}: reviewed status requires the complete second-pass scope")

        evidence_ids = set()
        for span in card.get("evidence_spans", []):
            evidence_id = span.get("evidence_id")
            if evidence_id in all_evidence:
                errors.append(f"{key}: duplicate global evidence ID {evidence_id}")
            all_evidence[evidence_id] = key
            evidence_ids.add(evidence_id)
            if span.get("source_pdf") != paper["filename"]:
                errors.append(f"{key}: {evidence_id} references the wrong PDF")
            page = span.get("page_pdf")
            if not isinstance(page, int) or not 1 <= page <= paper["pages"]:
                errors.append(f"{key}: {evidence_id} has invalid PDF page {page!r}")
            if not evidence_id.startswith(f"{key}-E"):
                errors.append(f"{key}: evidence ID {evidence_id} has the wrong prefix")

        for path, value in walk(card):
            if isinstance(value, list) and (
                path.endswith(".evidence_ids") or path.startswith("$.field_evidence.")
            ):
                for evidence_id in value:
                    if evidence_id not in evidence_ids:
                        errors.append(f"{key}: {path} references unknown evidence {evidence_id}")

        checked = {item["path"] for item in card.get("missing_field_checks", [])}
        for path, value in walk(card):
            if value == "not_reported" and not path.startswith("$.bibliography."):
                normalized = path.removeprefix("$.")
                if normalized not in checked:
                    errors.append(f"{key}: {normalized}=not_reported lacks checked locations")
        if paper["study_type"] == "review":
            for path, value in walk(card.get("paper_findings", {})):
                if path.endswith(".claim_class") and value == "reported_result":
                    errors.append(f"{key}: review content cannot be labelled reported_result")

    try:
        claims = read_jsonl(args.claims)
    except (OSError, json.JSONDecodeError) as exc:
        return errors + [f"synthesis_claims.jsonl: {exc}"]
    claim_schema = schema["$defs"]["synthesis_claim"]
    claim_ids = []
    for line, claim in enumerate(claims, 1):
        claim_ids.append(claim.get("claim_id"))
        errors += [
            f"synthesis claim line {line}: {item}"
            for item in schema_errors(claim, claim_schema, schema)
        ]
        for key in claim.get("supporting_papers", []) + claim.get("limiting_papers", []):
            if key not in manifest_by_key:
                errors.append(f"{claim.get('claim_id')}: unknown cite key {key}")
        for evidence_id in claim.get("supporting_evidence_ids", []) + claim.get("limiting_evidence_ids", []):
            if evidence_id not in all_evidence:
                errors.append(f"{claim.get('claim_id')}: unknown evidence ID {evidence_id}")
        for evidence_id in claim.get("supporting_evidence_ids", []):
            if evidence_id in all_evidence and all_evidence[evidence_id] not in claim.get("supporting_papers", []):
                errors.append(
                    f"{claim.get('claim_id')}: supporting evidence {evidence_id} "
                    "does not belong to a supporting paper"
                )
        for evidence_id in claim.get("limiting_evidence_ids", []):
            if evidence_id in all_evidence and all_evidence[evidence_id] not in claim.get("limiting_papers", []):
                errors.append(
                    f"{claim.get('claim_id')}: limiting evidence {evidence_id} "
                    "does not belong to a limiting paper"
                )
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("synthesis_claims.jsonl: duplicate claim IDs")
    question_numbers = {claim.get("question_number") for claim in claims if claim.get("section") == "question"}
    if question_numbers != set(range(1, 11)):
        errors.append("synthesis_claims.jsonl: must contain questions 1 through 10 exactly once")
    hypotheses = {claim.get("hypothesis_id") for claim in claims if claim.get("section") == "hypothesis"}
    if hypotheses != {f"H{index}" for index in range(1, 8)}:
        errors.append("synthesis_claims.jsonl: must assess hypotheses H1 through H7 exactly once")

    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cards_by_key = {card["cite_key"]: card for card in cards}
    expected_rows = [comparison_row(cards_by_key[key]) for key in keys if key in cards_by_key]
    if rows != expected_rows:
        errors.append("comparison_table.csv: content is not the exact JSONL projection")

    try:
        outputs = build_contents(args.manifest, args.jsonl, args.claims)
    except (KeyError, ValueError) as exc:
        errors.append(f"build: cannot derive artifacts from invalid sources: {exc}")
        return errors
    for path, expected in outputs.items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            errors.append(f"{path.name}: generated artifact is missing or outdated")
    if outputs != build_contents(args.manifest, args.jsonl, args.claims):
        errors.append("build: output is not deterministic")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "Papers/manifest.yaml")
    parser.add_argument("--jsonl", type=Path, default=ROOT / "knowledge_base/paper_cards.jsonl")
    parser.add_argument("--csv", type=Path, default=ROOT / "knowledge_base/comparison_table.csv")
    parser.add_argument("--claims", type=Path, default=ROOT / "knowledge_base/synthesis_claims.jsonl")
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "docs/literature/knowledge_base.schema.json",
    )
    args = parser.parse_args()
    try:
        errors = validate(args)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Knowledge base validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

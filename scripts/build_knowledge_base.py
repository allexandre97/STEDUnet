#!/usr/bin/env python3
"""Build deterministic knowledge-base artifacts from structured sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = [
    "cite_key",
    "corpus_role",
    "study_type",
    "method_paradigms",
    "task_focus",
    "architecture_family",
    "dimensionality",
    "main_targets",
    "losses",
    "data_strategy",
    "postprocessing",
    "evaluation_focus",
    "direct_sted_evidence",
    "best_use_case",
    "main_transfer_risk",
    "extraction_status",
]
SECTION_ORDER = {"finding": 0, "question": 1, "hypothesis": 2, "limitation": 3}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def cell(value: object) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def comparison_row(card: dict) -> dict[str, str]:
    summary = card["comparison_summary"]
    bibliography = card["bibliography"]
    values = {
        "cite_key": card["cite_key"],
        "corpus_role": bibliography["corpus_role"],
        "study_type": bibliography["study_type"],
        "method_paradigms": bibliography["method_paradigms"],
        **summary,
        "extraction_status": card["provenance"]["extraction_status"],
    }
    return {column: cell(values[column]) for column in CSV_COLUMNS}


def render_csv(cards: list[dict], order: dict[str, int]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for card in sorted(cards, key=lambda item: order[item["cite_key"]]):
        writer.writerow(comparison_row(card))
    return output.getvalue()


def claim_block(claim: dict) -> str:
    lines = [
        f"### {claim['claim_id']}: {claim['title']}",
        "",
        f"**Claim:** {claim['claim']}",
        "",
        f"**Evidence class:** `{claim['claim_class']}`",
        "",
        f"**Supporting papers:** {', '.join(f'`{key}`' for key in claim['supporting_papers']) or 'None'}",
        "",
        f"**Supporting evidence:** {', '.join(f'`{item}`' for item in claim['supporting_evidence_ids']) or 'None'}",
        "",
        f"**Contradicting or limiting papers:** {', '.join(f'`{key}`' for key in claim['limiting_papers']) or 'None'}",
        "",
        f"**Limiting evidence:** {', '.join(f'`{item}`' for item in claim['limiting_evidence_ids']) or 'None'}",
        "",
        f"**Confidence:** `{claim['confidence']}`",
    ]
    if "hypothesis_status" in claim:
        lines += ["", f"**Hypothesis status:** `{claim['hypothesis_status']}`"]
    lines += ["", f"**Implication for the STED project:** {claim['sted_implication']}", ""]
    return "\n".join(lines)


def render_synthesis(manifest: dict, cards: list[dict], claims: list[dict]) -> str:
    by_section = {section: [] for section in SECTION_ORDER}
    for claim in claims:
        by_section[claim["section"]].append(claim)
    for values in by_section.values():
        values.sort(key=lambda item: item["claim_id"])

    lines = [
        "# Literature synthesis",
        "",
        "## Corpus and evidence scope",
        "",
        f"This synthesis is derived from the eight-paper core corpus ({len(manifest['papers'])} manifest-listed PDFs) "
        "and structured evidence spans. "
        "Paper-reported content, review synthesis, and project inference remain separately labelled. "
        "The working hypotheses are evaluated after the evidence-neutral findings.",
        "",
        "## Extraction status",
        "",
        "| Cite key | Status | Publication status | Direct STED evidence |",
        "|:--|:--|:--|:--|",
    ]
    cards_by_key = {card["cite_key"]: card for card in cards}
    for paper in manifest["papers"]:
        card = cards_by_key[paper["cite_key"]]
        lines.append(
            f"| `{paper['cite_key']}` | `{card['provenance']['extraction_status']}` | "
            f"`{paper['publication_status']}` | `{card['comparison_summary']['direct_sted_evidence']}` |"
        )

    sections = [
        ("Evidence-neutral findings", "finding"),
        ("Cross-paper synthesis questions", "question"),
        ("Working-hypothesis assessment", "hypothesis"),
        ("Contradictions, unresolved questions, and corpus limitations", "limitation"),
    ]
    for heading, section in sections:
        lines += ["", f"## {heading}", ""]
        for claim in by_section[section]:
            lines += [claim_block(claim).rstrip(), ""]

    lines += [
        "",
        "## Design-option boundary",
        "",
        "These findings identify evidence and transfer risks; they do not select or implement a final model architecture.",
        "",
    ]
    return "\n".join(lines)


def render_index(
    manifest_path: Path,
    manifest: dict,
    cards: list[dict],
    cards_text: str,
    claims_text: str,
    csv_text: str,
    synthesis_text: str,
) -> str:
    cards_by_key = {card["cite_key"]: card for card in cards}
    lines = [
        "# Knowledge-base index",
        "",
        f"- Schema version: `{manifest['schema_version']}`",
        f"- Manifest: [`Papers/manifest.yaml`](../Papers/manifest.yaml)",
        f"- Manifest SHA-256: `{sha256(manifest_path.read_bytes())}`",
        f"- Latest validation date: `{manifest['validation_date']}`",
        "",
        "## Artifacts",
        "",
        f"- [`paper_cards.jsonl`](paper_cards.jsonl) — `{sha256(cards_text)}`",
        f"- [`synthesis_claims.jsonl`](synthesis_claims.jsonl) — `{sha256(claims_text)}`",
        f"- [`comparison_table.csv`](comparison_table.csv) — `{sha256(csv_text)}`",
        f"- [`synthesis.md`](synthesis.md) — `{sha256(synthesis_text)}`",
        "",
        "## Corpus status",
        "",
        "| Cite key | Corpus role | Source PDF | Status |",
        "|:--|:--|:--|:--|",
    ]
    for paper in manifest["papers"]:
        card = cards_by_key[paper["cite_key"]]
        lines.append(
            f"| `{paper['cite_key']}` | `{paper['corpus_role']}` | "
            f"[`{paper['filename']}`](../Papers/{paper['filename']}) | "
            f"`{card['provenance']['extraction_status']}` |"
        )
    statuses = {card["provenance"]["extraction_status"] for card in cards}
    warnings = ["`Togo26` is an arXiv preprint."]
    if statuses != {"reviewed"}:
        warnings.append(
            "Cards not marked `reviewed` still have unresolved second-pass work: "
            + ", ".join(
                f"`{card['cite_key']}`"
                for card in cards
                if card["provenance"]["extraction_status"] != "reviewed"
            )
            + "."
        )
    warnings.append("All cards require independent human review before `verified`.")
    lines += [
        "",
        "## Validation",
        "",
        "```bash",
        "python scripts/validate_knowledge_base.py",
        "```",
        "",
        "Known warnings: " + " ".join(warnings),
        "",
    ]
    return "\n".join(lines)


def build_contents(manifest_path: Path, cards_path: Path, claims_path: Path) -> dict[Path, str]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    cards_text = cards_path.read_text(encoding="utf-8")
    claims_text = claims_path.read_text(encoding="utf-8")
    cards = read_jsonl(cards_path)
    claims = read_jsonl(claims_path)
    order = {paper["cite_key"]: index for index, paper in enumerate(manifest["papers"])}
    csv_text = render_csv(cards, order)
    synthesis_text = render_synthesis(manifest, cards, claims)
    output_dir = cards_path.parent
    index_text = render_index(
        manifest_path, manifest, cards, cards_text, claims_text, csv_text, synthesis_text
    )
    return {
        output_dir / "comparison_table.csv": csv_text,
        output_dir / "synthesis.md": synthesis_text,
        output_dir / "kb_index.md": index_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "Papers/manifest.yaml")
    parser.add_argument("--jsonl", type=Path, default=ROOT / "knowledge_base/paper_cards.jsonl")
    parser.add_argument("--claims", type=Path, default=ROOT / "knowledge_base/synthesis_claims.jsonl")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    outputs = build_contents(args.manifest, args.jsonl, args.claims)
    changed = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                changed.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if changed:
        for path in changed:
            print(f"outdated generated artifact: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

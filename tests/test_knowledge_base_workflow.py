from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_knowledge_base as build  # noqa: E402
import validate_knowledge_base as validate  # noqa: E402


class KnowledgeBaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.base_tmp.name)
        shutil.copytree(ROOT / "Papers", cls.base / "Papers")
        shutil.copytree(ROOT / "knowledge_base", cls.base / "knowledge_base")
        (cls.base / "docs/literature").mkdir(parents=True)
        shutil.copy2(
            ROOT / "docs/literature/knowledge_base.schema.json",
            cls.base / "docs/literature/knowledge_base.schema.json",
        )

    @classmethod
    def tearDownClass(cls):
        cls.base_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "Papers").mkdir()
        for pdf in (self.base / "Papers").glob("*.pdf"):
            os.link(pdf, self.root / "Papers" / pdf.name)
        shutil.copy2(self.base / "Papers/manifest.yaml", self.root / "Papers/manifest.yaml")
        shutil.copytree(self.base / "knowledge_base", self.root / "knowledge_base")
        (self.root / "docs/literature").mkdir(parents=True)
        shutil.copy2(
            self.base / "docs/literature/knowledge_base.schema.json",
            self.root / "docs/literature/knowledge_base.schema.json",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def args(self):
        return argparse.Namespace(
            manifest=self.root / "Papers/manifest.yaml",
            jsonl=self.root / "knowledge_base/paper_cards.jsonl",
            csv=self.root / "knowledge_base/comparison_table.csv",
            claims=self.root / "knowledge_base/synthesis_claims.jsonl",
            schema=self.root / "docs/literature/knowledge_base.schema.json",
        )

    def run_validator(self):
        with patch.object(validate, "pdf_pages", return_value=None):
            return validate.validate(self.args())

    def rewrite_jsonl(self, path: Path, records: list[dict]):
        path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_repository_fixture_is_valid(self):
        self.assertEqual(self.run_validator(), [])

    def test_duplicate_manifest_key_fails(self):
        manifest = yaml.safe_load(self.args().manifest.read_text())
        manifest["papers"].append(dict(manifest["papers"][0]))
        self.args().manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))
        self.assertTrue(any("duplicate cite keys" in error for error in self.run_validator()))

    def test_every_manifest_entry_is_core(self):
        manifest = yaml.safe_load(self.args().manifest.read_text())
        self.assertEqual(len(manifest["papers"]), 8)
        self.assertEqual({paper["corpus_role"] for paper in manifest["papers"]}, {"core"})
        self.assertIn("Togo26", {paper["cite_key"] for paper in manifest["papers"]})

    def test_non_core_manifest_entry_fails(self):
        manifest = yaml.safe_load(self.args().manifest.read_text())
        manifest["papers"][-1]["corpus_role"] = "supplementary"
        self.args().manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))
        self.assertTrue(any("corpus_role must be core" in error for error in self.run_validator()))

    def test_missing_card_fails(self):
        cards = build.read_jsonl(self.args().jsonl)
        self.rewrite_jsonl(self.args().jsonl, cards[:-1])
        self.assertTrue(any("cite keys differ" in error for error in self.run_validator()))

    def test_checksum_mismatch_fails(self):
        manifest = yaml.safe_load(self.args().manifest.read_text())
        manifest["papers"][0]["sha256"] = "0" * 64
        self.args().manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))
        self.assertTrue(any("checksum mismatch" in error for error in self.run_validator()))

    def test_invalid_controlled_value_fails_schema(self):
        cards = build.read_jsonl(self.args().jsonl)
        cards[0]["comparison_summary"]["architecture_family"] = "invented_network"
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("invalid value" in error for error in self.run_validator()))

    def test_unknown_evidence_reference_fails(self):
        cards = build.read_jsonl(self.args().jsonl)
        cards[0]["agent_assessment"]["main_strength"]["evidence_ids"] = ["Liu18-E999"]
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("unknown evidence" in error for error in self.run_validator()))

    def test_invalid_evidence_page_fails(self):
        cards = build.read_jsonl(self.args().jsonl)
        cards[0]["evidence_spans"][0]["page_pdf"] = 999
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("invalid PDF page" in error for error in self.run_validator()))

    def test_review_result_misclassification_fails(self):
        cards = build.read_jsonl(self.args().jsonl)
        review = next(card for card in cards if card["cite_key"] == "Ozd21")
        review["paper_findings"]["other_reported_findings"][0]["claim_class"] = "reported_result"
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("review content" in error for error in self.run_validator()))

    def test_missing_checked_locations_fails(self):
        cards = build.read_jsonl(self.args().jsonl)
        card = next(card for card in cards if card["cite_key"] == "Liu20")
        card["missing_field_checks"] = []
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("lacks checked locations" in error for error in self.run_validator()))

    def test_reviewed_card_requires_complete_review_metadata(self):
        cards = build.read_jsonl(self.args().jsonl)
        cards[0]["provenance"]["review_scope"] = ["methods"]
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("complete second-pass scope" in error for error in self.run_validator()))

    def test_original_extraction_date_is_not_fabricated(self):
        cards = build.read_jsonl(self.args().jsonl)
        cards[0]["provenance"]["extracted_on"] = "2026-06-18"
        self.rewrite_jsonl(self.args().jsonl, cards)
        self.assertTrue(any("recorded honestly" in error for error in self.run_validator()))

    def test_liu18_training_stages_are_distinct(self):
        cards = build.read_jsonl(self.args().jsonl)
        card = next(card for card in cards if card["cite_key"] == "Liu18")
        training = {item["field"]: item["value"] for item in card["method"]["training"]}
        self.assertEqual(training["annotation_model_epochs"], 20)
        self.assertEqual(training["final_segmentation_epochs"], 15)

    def test_csv_drift_fails(self):
        csv_path = self.args().csv
        csv_path.write_text(csv_path.read_text().replace("mask baseline", "changed baseline", 1))
        self.assertTrue(any("exact JSONL projection" in error for error in self.run_validator()))

    def test_unknown_synthesis_evidence_fails(self):
        claims = build.read_jsonl(self.args().claims)
        claims[0]["supporting_evidence_ids"] = ["Missing-E001"]
        self.rewrite_jsonl(self.args().claims, claims)
        self.assertTrue(any("unknown evidence ID" in error for error in self.run_validator()))

    def test_build_is_deterministic(self):
        first = build.build_contents(self.args().manifest, self.args().jsonl, self.args().claims)
        second = build.build_contents(self.args().manifest, self.args().jsonl, self.args().claims)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

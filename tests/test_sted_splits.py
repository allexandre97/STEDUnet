import csv

from fibras.sted_splits import create_splits, validate_splits, write_split_report


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def minimal_row(stable_id, kind, pn, acq, condition="AD", div="DIV01", round_id="1R", series="0"):
    return {
        "stable_image_id": stable_id,
        "source_kind": kind,
        "biological_group_candidate": pn,
        "acquisition_group": acq,
        "inferred_condition": condition,
        "inferred_div": div,
        "inferred_round": round_id,
        "series_index": series,
    }


def test_splits_are_grouped_and_provisional(tmp_path):
    fields = [
        "stable_image_id",
        "source_kind",
        "biological_group_candidate",
        "acquisition_group",
        "inferred_condition",
        "inferred_div",
        "inferred_round",
        "series_index",
    ]
    write_csv(
        tmp_path / "sted_images.csv",
        [
            minimal_row("img1", "fiber_image", "PN001", "PN001_1R_AD_DIV01"),
            minimal_row("img2", "fiber_image", "PN002", "PN002_1R_AD_DIV01"),
        ],
        fields,
    )
    write_csv(
        tmp_path / "sted_blanks.csv",
        [
            minimal_row("blank1", "blank_background", "PN001", "PN001_1R_AD_DIV02"),
            minimal_row("blank2", "blank_background", "PN002", "PN002_1R_AD_DIV02"),
        ],
        fields,
    )
    rows = create_splits(tmp_path)
    assert all(r["human_approved"] == "false" for r in rows)
    for pn in {r["biological_group_candidate"] for r in rows}:
        assert len({r["eligibility"] for r in rows if r["biological_group_candidate"] == pn}) == 1
    assert all(r["primary_metric_role"] == r["eligibility"] for r in rows)
    out = tmp_path / "sted_splits.csv"
    from fibras.sted_splits import write_csv as write_split_csv

    write_split_csv(out, rows)
    assert validate_splits(tmp_path, out) == []
    report = tmp_path / "sted_split_report.md"
    write_split_report(out, report)
    text = report.read_text()
    assert "PN-level split report" in text
    assert "too_few_independent_pns" in text


def test_pn_leakage_across_source_kinds_is_rejected(tmp_path):
    fields = [
        "stable_image_id",
        "source_kind",
        "biological_group_candidate",
        "acquisition_group",
        "inferred_condition",
        "inferred_div",
        "inferred_round",
        "series_index",
    ]
    write_csv(tmp_path / "sted_images.csv", [minimal_row("img1", "fiber_image", "PN001", "PN001_1R_AD_DIV01")], fields)
    write_csv(tmp_path / "sted_blanks.csv", [minimal_row("blank1", "blank_background", "PN001", "PN001_1R_AD_DIV02")], fields)
    from fibras.sted_splits import write_csv as write_split_csv

    rows = create_splits(tmp_path)
    rows[0]["eligibility"] = "training"
    rows[1]["eligibility"] = "held_out_test"
    out = tmp_path / "sted_splits.csv"
    write_split_csv(out, rows)
    assert any("PN PN001 crosses primary roles" in error for error in validate_splits(tmp_path, out))

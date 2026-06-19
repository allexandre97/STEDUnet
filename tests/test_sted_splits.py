import csv

from fibras.sted_splits import create_splits, validate_splits, write_split_report


FIELDS = [
    "stable_image_id",
    "source_kind",
    "culture_id",
    "disease",
    "tau_isoform",
    "experimental_condition",
    "div",
    "div_token",
    "series_index",
    "experimental_group_id",
    "acquisition_group",
]


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def minimal_row(stable_id, kind, culture, disease="AD", isoform="3R", div="3", series="0"):
    condition = f"{disease}_{isoform}"
    group = f"{culture}_{condition}_DIV{int(div):02d}"
    return {
        "stable_image_id": stable_id,
        "source_kind": kind,
        "culture_id": culture,
        "disease": disease,
        "tau_isoform": isoform,
        "experimental_condition": condition,
        "div": str(int(div)),
        "div_token": f"DIV{int(div):02d}",
        "series_index": series,
        "experimental_group_id": group,
        "acquisition_group": group,
    }


def test_splits_are_grouped_by_experimental_group_and_provisional(tmp_path):
    write_csv(
        tmp_path / "sted_images.csv",
        [
            minimal_row("img1", "fiber_image", "PN001", div="3", series="0"),
            minimal_row("img2", "fiber_image", "PN001", div="3", series="1"),
            minimal_row("img3", "fiber_image", "PN002", div="3", series="0"),
            minimal_row("img4", "fiber_image", "PN003", div="3", series="0"),
        ],
    )
    write_csv(tmp_path / "sted_blanks.csv", [minimal_row("blank1", "blank_background", "PN001", div="5")])
    rows = create_splits(tmp_path)
    assert all(r["human_approved"] == "false" for r in rows)
    for group in {r["experimental_group_id"] for r in rows}:
        assert len({r["eligibility"] for r in rows if r["experimental_group_id"] == group}) == 1
    assert all(r["grouping_rule"] == "experimental_group_holdout_primary" for r in rows)
    out = tmp_path / "sted_splits.csv"
    from fibras.sted_splits import write_csv as write_split_csv

    write_split_csv(out, rows)
    assert validate_splits(tmp_path, out) == []
    report = tmp_path / "sted_split_report.md"
    write_split_report(out, report)
    text = report.read_text()
    assert "experimental-group split report" in text
    assert "too_few_experimental_groups" in text
    assert "tau isoforms" in text


def test_experimental_group_leakage_is_rejected(tmp_path):
    write_csv(
        tmp_path / "sted_images.csv",
        [
            minimal_row("img1", "fiber_image", "PN001", series="0"),
            minimal_row("img2", "fiber_image", "PN001", series="1"),
        ],
    )
    write_csv(tmp_path / "sted_blanks.csv", [])
    from fibras.sted_splits import write_csv as write_split_csv

    rows = create_splits(tmp_path)
    rows[0]["eligibility"] = "training"
    rows[1]["eligibility"] = "held_out_test"
    out = tmp_path / "sted_splits.csv"
    write_split_csv(out, rows)
    assert any("experimental group PN001_AD_3R_DIV03 crosses roles" in error for error in validate_splits(tmp_path, out))


def test_culture_held_out_strategy_is_separate(tmp_path):
    write_csv(
        tmp_path / "sted_images.csv",
        [
            minimal_row("img1", "fiber_image", "PN001"),
            minimal_row("img2", "fiber_image", "PN002"),
        ],
    )
    write_csv(tmp_path / "sted_blanks.csv", [])
    rows = create_splits(tmp_path, strategy="culture_held_out")
    assert {r["grouping_rule"] for r in rows} == {"culture_held_out"}
    assert "culture_held_out" in {r["eligibility"] for r in rows}


def test_split_validator_rejects_inconsistent_isoform_and_group_metadata(tmp_path):
    write_csv(tmp_path / "sted_images.csv", [minimal_row("img1", "fiber_image", "PN001")])
    write_csv(tmp_path / "sted_blanks.csv", [])
    from fibras.sted_splits import write_csv as write_split_csv

    rows = create_splits(tmp_path)
    rows[0]["tau_isoform"] = "5R"
    rows[0]["experimental_group_id"] = "wrong"
    out = tmp_path / "sted_splits.csv"
    write_split_csv(out, rows)
    errors = validate_splits(tmp_path, out)
    assert any("invalid tau_isoform" in error for error in errors)
    assert any("inconsistent experimental_group_id" in error for error in errors)

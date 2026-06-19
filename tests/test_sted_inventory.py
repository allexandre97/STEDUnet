from pathlib import Path

import numpy as np
from PIL import Image

from fibras.sted_inventory import SourceRoot, collect_records


def save_tif(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def test_inventory_records_required_fields_and_blank_provenance(tmp_path):
    root = tmp_path / "blank"
    root.mkdir()
    save_tif(root / "PN001_3R_AD_DIV01 (Series 0) [1].tif", np.arange(64, dtype=np.uint8).reshape(8, 8))
    records = collect_records(SourceRoot("sted_blank_data", root, "blank_background"))
    assert len(records) == 1
    rec = records[0]
    assert rec["source_root_id"] == "sted_blank_data"
    assert rec["relative_path"].endswith(".tif")
    assert rec["dtype"] == "uint8"
    assert rec["shape_y"] == "8"
    assert rec["shape_x"] == "8"
    assert rec["culture_id"] == "PN001"
    assert rec["tau_isoform"] == "3R"
    assert rec["disease"] == "AD"
    assert rec["div"] == "1"
    assert rec["experimental_condition"] == "AD_3R"
    assert rec["experimental_group_id"] == "PN001_AD_3R_DIV01"
    assert rec["blank_status"] == "expert_validated"
    assert rec["validation_source"] == "human_expert_review"
    assert rec["validation_date"] == "not_recorded"


def test_inventory_includes_corrupted_files(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "PN001_3R_AD_DIV01 (Series 0) [1].tif").write_bytes(b"not a tif")
    records = collect_records(SourceRoot("sted_fiber_data", root, "fiber_image"))
    assert len(records) == 1
    assert records[0]["dtype"] == "not_readable"
    assert records[0]["validation_flags"].startswith("unreadable:")

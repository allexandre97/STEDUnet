import numpy as np
from PIL import Image, TiffImagePlugin

from fibras.sted_inventory import SourceRoot, collect_records


def save_with_description(path, arr, description):
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = description
    Image.fromarray(arr).save(path, tiffinfo=info)


def test_pixel_duplicate_not_forced_to_near_duplicate(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    save_with_description(root / "PN001_1R_AD_DIV01 (Series 0) [1].tif", arr, "a")
    save_with_description(root / "PN001_1R_AD_DIV01 (Series 1) [1].tif", arr, "b")
    records = collect_records(SourceRoot("sted_fiber_data", root, "fiber_image"))
    assert {r["pixel_duplicate_group"] for r in records} != {"none"}
    assert {r["thumbnail_duplicate_group"] for r in records} != {"none"}
    assert {r["near_duplicate_candidate_group"] for r in records} == {"none"}


def test_near_duplicate_candidate_detects_small_perturbation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    noise = np.random.default_rng(5).integers(-2, 3, size=arr.shape)
    perturbed = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(root / "PN001_1R_AD_DIV01 (Series 0) [1].tif")
    Image.fromarray(perturbed).save(root / "PN001_1R_AD_DIV01 (Series 1) [1].tif")
    records = collect_records(SourceRoot("sted_fiber_data", root, "fiber_image"))
    assert any(r["near_duplicate_candidate_group"] != "none" for r in records)

from fibras.sted_filename_parser import parse_sted_filename


def test_parse_known_sted_filename():
    parsed = parse_sted_filename("PN148_3R_AD_DIV07 (Series 8) [1].tif")
    assert parsed.inferred_pn == "PN148"
    assert parsed.inferred_round == "3R"
    assert parsed.inferred_condition == "AD"
    assert parsed.inferred_div == "DIV07"
    assert parsed.series_index == "8"
    assert parsed.acquisition_group == "PN148_3R_AD_DIV07"
    assert parsed.biological_group_candidate == "PN148"
    assert parsed.parse_status == "parsed"


def test_parse_unknown_filename_is_explicit():
    parsed = parse_sted_filename("unstructured.tif")
    assert parsed.inferred_pn == "unknown"
    assert parsed.parse_status == "unparsed"


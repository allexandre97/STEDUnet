from fibras.sted_filename_parser import parse_sted_filename


def test_parse_known_sted_filename():
    parsed = parse_sted_filename("PN148_3R_AD_DIV07 (Series 8) [1].tif")
    assert parsed.culture_id == "PN148"
    assert parsed.tau_isoform == "3R"
    assert parsed.disease == "AD"
    assert parsed.div == "7"
    assert parsed.div_token == "DIV07"
    assert parsed.series_index == "8"
    assert parsed.experimental_condition == "AD_3R"
    assert parsed.experimental_group_id == "PN148_AD_3R_DIV07"
    assert parsed.acquisition_group == "PN148_3R_AD_DIV07"
    assert parsed.parse_status == "parsed"
    assert parsed.inferred_round == "3R"


def test_parse_unknown_filename_is_explicit():
    parsed = parse_sted_filename("unstructured.tif")
    assert parsed.culture_id == "unknown"
    assert parsed.parse_status == "unparsed"


def test_parse_div3_and_div03_as_same_timepoint():
    a = parse_sted_filename("PN148_4R_PSP_DIV3 (Series 1) [1].tif")
    b = parse_sted_filename("PN148_4R_PSP_DIV03 (Series 2) [1].tif")
    assert a.tau_isoform == "4R"
    assert a.disease == "PSP"
    assert a.div == b.div == "3"
    assert a.experimental_condition == "PSP_4R"
    assert a.experimental_group_id == b.experimental_group_id == "PN148_PSP_4R_DIV03"


def test_invalid_tau_isoform_is_not_guessed():
    parsed = parse_sted_filename("PN148_5R_AD_DIV03 (Series 1) [1].tif")
    assert parsed.tau_isoform == "unknown"
    assert parsed.experimental_condition == "unknown"
    assert parsed.experimental_group_id == "unknown"
    assert parsed.parse_status == "invalid_tau_isoform"

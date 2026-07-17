from ufc_prediction.paths import RAW_FIGHTS_PATH, SELECTED_FEATURES_PATH


def test_canonical_input_files_exist():
    assert RAW_FIGHTS_PATH.is_file()
    assert SELECTED_FEATURES_PATH.is_file()

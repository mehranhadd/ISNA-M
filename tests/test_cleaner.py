import pandas as pd

from isna_m.cleaner import create_output_file, time_cleaner
from isna_m.paths import DEFAULT_CLEAN_DATASET, DEFAULT_RAW_DATASET, PROJECT_ROOT


def test_dataset_defaults_are_inside_project():
    assert DEFAULT_RAW_DATASET.is_relative_to(PROJECT_ROOT)
    assert DEFAULT_CLEAN_DATASET.is_relative_to(PROJECT_ROOT)
    assert DEFAULT_RAW_DATASET.as_posix().endswith("data/raw/Raw_ISNA_Dataset.csv")
    assert DEFAULT_CLEAN_DATASET.as_posix().endswith("data/processed/Clean_ISNA_Dataset.csv")


def test_time_cleaner_translates_persian_digits():
    dataframe = pd.DataFrame({"Time": ["۲۳:۴۵", "۰۸:۰۱"]})
    time_cleaner(dataframe)
    assert dataframe["Time"].tolist() == ["23:45", "08:01"]


def test_create_output_file_writes_expected_columns(tmp_path):
    output = tmp_path / "nested" / "clean.csv"
    dataframe = create_output_file([["خبر", "امروز"], ["متن"]], ["خبر", "ورزش"], output)
    assert dataframe.columns.tolist() == ["Content", "Category"]
    assert dataframe["Content"].tolist() == ["خبر امروز", "متن"]
    assert output.is_file()

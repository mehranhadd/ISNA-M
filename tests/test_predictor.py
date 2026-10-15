import pandas as pd
import pytest

from isna_m.predictor import evaluate_estimators


def test_predictor_reports_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_estimators(tmp_path / "missing.csv", estimators=(1,))


def test_predictor_rejects_missing_columns(tmp_path):
    source = tmp_path / "incorrect.csv"
    pd.DataFrame({"headline": ["test"], "label": ["news"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_estimators(source, estimators=(1,))

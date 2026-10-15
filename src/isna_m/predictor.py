"""Run the original bag-of-words/random-forest category experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .paths import DEFAULT_CLEAN_DATASET

DEFAULT_ESTIMATORS = (10, 25, 50, 100, 500, 1000)


def evaluate_estimators(
    dataset_path: str | Path = DEFAULT_CLEAN_DATASET,
    *,
    estimators: tuple[int, ...] = DEFAULT_ESTIMATORS,
    max_features: int = 5000,
    test_size: float = 0.4,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> dict[int, str]:
    """Evaluate the listed forest sizes and return their classification reports."""

    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Clean dataset not found: {dataset_path}")
    if not estimators or any(n < 1 for n in estimators):
        raise ValueError("estimators must contain positive integers")
    if max_features < 1:
        raise ValueError("max_features must be at least 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataframe = pd.read_csv(dataset_path)
    required_columns = {"Content", "Category"}
    missing = sorted(required_columns.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Clean dataset is missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("Clean dataset contains no rows")

    dataframe = dataframe.astype({"Content": "str", "Category": "category"})
    vectorizer = CountVectorizer(max_features=max_features)
    # Keep the matrix sparse instead of converting it to a dense array. This
    # lowers memory use without changing the features given to scikit-learn.
    features = vectorizer.fit_transform(dataframe["Content"])
    labels = np.array(dataframe["Category"].values, ndmin=2).T
    encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    reports: dict[int, str] = {}
    for estimator_count in estimators:
        classifier = RandomForestClassifier(n_estimators=estimator_count, random_state=random_state)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        reports[estimator_count] = classification_report(y_test, predictions, zero_division=0)

    for estimator_count, report in reports.items():
        print(f"n_estimators={estimator_count}\n{report}\n{'-' * 40}")

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for estimator_count, report in reports.items():
                handle.write(f"n_estimators={estimator_count}\n{report}\n{'-' * 40}\n")
        print(f"Saved evaluation reports to {destination}")

    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate random-forest baselines for ISNA article categories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN_DATASET, help="Cleaned CSV dataset path.")
    parser.add_argument("--estimators", type=int, nargs="+", default=list(DEFAULT_ESTIMATORS), help="Forest sizes to evaluate.")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum bag-of-words features.")
    parser.add_argument("--test-size", type=float, default=0.4, help="Fraction of samples used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split and classifier.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for saving text reports.")
    args = parser.parse_args(argv)
    evaluate_estimators(
        args.input,
        estimators=tuple(args.estimators),
        max_features=args.max_features,
        test_size=args.test_size,
        random_state=args.random_state,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

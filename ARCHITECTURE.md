# Project architecture

## Runtime modules

- `src/isna_m/crawler.py` contains ISNA archive navigation, page parsing, and raw dataset creation. Network access runs only from `crawl()` or the `isna-crawl` command—not when the module is imported.
- `src/isna_m/cleaner.py` Persian text normalization, tokenization, stop-word removal, stemming, and descriptive plots. Inputs and outputs are explicit parameters and default paths are derived from the project location.
- `src/isna_m/predictor.py` runs the bag-of-words and random-forest baseline across configurable estimator counts. It uses the one-hot target approach.
- `src/isna_m/paths.py` centralizes repository-relative paths.

## Data and reports

- `data/raw/Raw_ISNA_Dataset.csv` is the collected dataset.
- `data/processed/Clean_ISNA_Dataset.csv` is the preprocessed dataset.
- `reports/figures/` contains the figures. Running `isna-clean` regenerates these figures.

## Scope and limitations

The crawler relies on the current ISNA HTML structure, which may change. Its default request delay is two seconds. The refactor avoids launching network requests on import and improves missing-element handling.

The classification experiment has a small, imbalanced dataset and uses one-hot encoded labels with a random-forest multi-output target. A statistically sound multiclass evaluation would be a separate modelling change.
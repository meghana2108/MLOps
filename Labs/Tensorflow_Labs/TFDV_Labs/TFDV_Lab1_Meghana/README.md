# TFDV Lab 1 — Meghana

TensorFlow Data Validation (TFDV) lab on the UCI Census Income (Adult) dataset. Starting from the course notebook, this version computes and visualizes dataset statistics, infers a schema, detects and fixes anomalies in the evaluation set, compares data slices, and adds three extensions.

## Files

- `TFDV_Lab1_Meghana.ipynb` — the completed notebook
- `util.py`, `data/`, `img/` — unchanged from the course lab
- `output/schema.pbtxt` — the curated schema, written when the notebook runs

## What the notebook covers

1. Generate and visualize training statistics with `generate_statistics_from_dataframe` and `visualize_statistics`.
2. Infer a schema from the training statistics.
3. Compare eval vs. train statistics, filter out invalid ages and missing values, and re-check.
4. Detect anomalies in the eval set (unexpected `race`, `native-country`, `occupation` values) and fix them by relaxing `min_domain_mass`, adding domain values, and restricting `age` to 17–90.
5. Compare statistics across slices (sex, and sex × race).

## Changes from the original lab

- **CSV export fix:** `train_df.to_csv(CSV_PATH, index=False)`. The original wrote the pandas index, which TFDV then treated as an extra feature column in the sliced statistics.
- **Slices looked up by name:** added a `get_slice()` helper that returns a slice as a `DatasetFeatureStatisticsList` by its name (e.g. `race_Black_sex_Male`) instead of by hard-coded list index, which would silently compare the wrong groups if TFDV returned the slices in a different order. Also added extra slice comparisons and a per-slice example count.
- **Drift detection:** added L-infinity drift comparators on `workclass`, `occupation`, `education`, and `marital-status`, and validated eval stats against train stats with `previous_statistics`. Ran it at a strict (0.01) and a relaxed (0.05) threshold.
- **Training vs. serving environments:** created serving data by dropping `label`, showed the resulting "missing column" anomaly, then declared `TRAINING` and `SERVING` environments with `label` excluded from serving so serving data validates cleanly.
- **Schema export:** saved the curated schema with `write_schema_text`, reloaded it with `load_schema_text`, and confirmed it still validates the eval data.

## How to run

```bash
cd Labs/Tensorflow_Labs/TFDV_Labs/TFDV_Lab1_Meghana
python3 -m venv tfdv_env
source tfdv_env/bin/activate
pip install tensorflow-data-validation scikit-learn pandas jupyter
jupyter notebook TFDV_Lab1_Meghana.ipynb
```

Then run all cells (Kernel → Restart & Run All).

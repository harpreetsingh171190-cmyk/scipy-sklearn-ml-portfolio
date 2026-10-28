# Modernization notes (2026)

## Changes made in this refresh

- Replaced the original Python 3.6-era Conda environment with a Python 3.11+ environment and current package families.
- Added `requirements.txt` and refreshed `environment.yml`.
- Rebuilt `check_env.ipynb` to print installed Python and core scientific-package versions.
- Updated notebook kernel metadata to the standard `python3` / `ipykernel` kernel.
- Updated common removed module paths such as `sklearn.cross_validation`, `sklearn.grid_search`, and `sklearn.learning_curve` to `sklearn.model_selection`.
- Updated the old `Imputer` reference to `SimpleImputer` and replaced the retired Boston housing dataset exercise references with California housing.
- Added refreshed graphics under `images/` and rewrote the README for modern setup instructions.

## Important compatibility caveats

This project began as a tutorial written for the Python and scikit-learn ecosystem in 2017. API migration is not always a simple text replacement: function signatures, default values, dataset sources, plotting behavior, and estimator parameters can change. Some notebooks may still need cell-by-cell edits before they run cleanly on a specific modern package release.

The original notebooks also include external data, plotting helpers, and HTML used by the historic course. This update has **not** been represented as a full execution test of all 23 lessons. Please run each notebook from a fresh kernel and fix errors as they appear.

## Suggested compatibility checks

1. Run `check_env.ipynb`.
2. Run each lesson with **Restart Kernel and Run All Cells**.
3. Search for removed APIs in a failing notebook and compare against the official scikit-learn migration notes.
4. For model evaluation, prefer `sklearn.model_selection` and `sklearn.metrics`.
5. For missing values, use `sklearn.impute.SimpleImputer`.
6. For a replacement for the retired Boston housing example, use `sklearn.datasets.fetch_california_housing` where appropriate. It may need a first-time network download.
7. Review any dataset downloads before running them; URLs and dataset licenses can change.

## Version policy

The dependencies are specified with minimum versions rather than a frozen lockfile. This improves flexibility but does not guarantee identical behavior across all systems. For a reproducible project, run the notebooks in a clean environment and save a tested lockfile after confirming compatibility on your target operating system.


## Additional 2026 feature update

This update adds four new notebooks (24–27) using modern scikit-learn patterns: `ColumnTransformer` and mixed-type preprocessing; held-out permutation importance and error analysis; validation-only threshold selection and calibration; and `joblib` persistence with JSON metadata. It also adds three generated diagrams under `images/` and an optional Streamlit Iris demo at `apps/streamlit_iris_demo.py`. Streamlit is intentionally optional and is not part of the core notebook requirements.

The new notebooks are syntax/structure checked, but should still be executed in the target environment before relying on their numerical outputs. Original 2017 lessons remain legacy teaching material and may need further cell-by-cell migration.

## v3.0.0 (2026 feature release)

Adds the model/dataset registries, cross-validated leaderboard, permutation importance, CLI, checksum-verified model persistence, JSON experiment logging, an expanded dashboard, notebooks 28–29, Docker/Make/pre-commit tooling and a safer `fetch_data.py`. The library, CLI, tests and notebooks 28–29 were executed during this upgrade; the Streamlit dashboard was syntax-checked only (Streamlit was not installed in the build environment), so launch it once locally. The original 2017 lessons (01–23) are still legacy material.


## v3.2.0 — legacy notebook repair status

Every code cell of notebooks 01–23 was executed in a headless runner (matplotlib Agg, per-cell figure close) against scikit-learn 1.8 / pandas 3 / Python 3.12 and the failures fixed. Result: 01–17 and 19–22 run without errors. Not fully verified here:

- **18 (Trees & Forests):** two cells use `ipywidgets` interactive sliders, which were not installed in the test environment (now in requirements).
- **23 (Out-of-core text):** needs the 84 MB IMDb download (`python fetch_data.py`); it now stops with a clear message if the data is missing. Its early `CountVectorizer` cells run.
- **Lesson solutions 06A and 15A** now use `fetch_california_housing` (matching the exercise text), which downloads once and was not run offline.
- Numerical outputs and plots were not compared against the 2017 originals; interactive widgets and `%load` solution cells were not exercised in a live Jupyter session.

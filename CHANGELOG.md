# Changelog

## 3.8.0 — review, lint and GitHub-readiness pass

- Fixed `.dockerignore` (it contained literal `\n` characters, so nothing was ignored and `.git` was copied into images); added `.gitattributes`; extended `.gitignore`.
- Lint: removed unused imports, sorted/formatted every import block the way ruff's isort rules expect, added `strict=` to `zip()` calls (B905), removed a call-in-default argument (B008) and declared `src`, `api` and `fetch_data` as first-party in `pyproject.toml`.
- `fetch_data.py`: archive extraction now stays inside `notebooks/` and rejects path-traversal entries; the old code extracted into the current directory.
- Versions aligned (`pyproject.toml`, API, `CITATION.cff` were 3.7.0 / 3.7.0 / 3.3.0); `CITATION.cff` gained authors, license and repository URL.
- `pyproject.toml`: pytest `pythonpath`, keywords, classifiers and project URLs.
- Added `scripts/check_notebooks.py` (JSON + syntax check of all notebooks, no Jupyter needed), `tests/test_notebooks_and_fetch.py`, `make check`, and a notebook check step in CI; CI now cancels superseded runs and has job timeouts.
- README: correct notebook count (35), dashboard tabs (14), Python versions (3.11-3.13), full CLI and API listings, notebooks 34-35 in the index, real badge URLs, updated upload checklist.
- Dockerfile/compose: Python env flags and an API health check.
- Added issue/PR templates.

## 3.7.0 — CSV upload API, time-series forecasting, HTML report

- API: upload a CSV as the raw request body (`Content-Type: text/csv`, no extra dependency) to `POST /csv/profile`, `/csv/explain`, `/csv/imbalance` and `/csv/forecast`; 2 MB / 50,000-row limits; `GET /report.html` serves the comparison report.
- Added `src/forecast.py`: naive, seasonal-naive, Ridge-on-lags and Random-Forest-on-differences forecasters, automatic seasonality detection, rolling-origin backtests (MAE, RMSE, MASE) and rough error bands.
- Added `src/htmlreport.py`: self-contained HTML model-comparison report (inline CSS and SVG, light/dark aware, no scripts or external requests).
- New CLI commands: `forecast`, `html-report`; dashboard tab "Forecast & report".
- `tests/test_forecast_report.py` (8 tests); CI smoke test covers the new commands.

## 3.6.0 — your own CSV: explanations and imbalance

- Added `src/csvtools.py`: SHAP-style explanation of any row of your CSV (numeric + categorical + missing values) and imbalance-strategy comparison for a binary target.
- New CLI commands: `csv-explain`, `csv-imbalance`; dashboard "Your CSV" tab gets both as buttons.
- `src/imbalance.py`: new `compare_strategies_xy` for arbitrary data.
- Notebook 35; CI smoke test now exercises the new CLI commands; 2 more tests.

## 3.5.0 — imbalance strategies and regression explanations

- Added `src/imbalance.py`: skews a binary dataset to a chosen minority share and compares baseline, class weights, oversampling and undersampling on minority-class recall, F1, balanced accuracy and PR-AUC (resampling on the training split only).
- `src/explain.py`: `explain_regression` gives SHAP-style contributions for regressors.
- New CLI commands: `imbalance`, `explain-regression`; API endpoint `GET /imbalance`.
- Dashboard: class-imbalance section in "Ensembles & features" and a regression explanation in the "SHAP-style explain" tab.
- Version bumped; 3 more tests.

## 3.4.0 — ensembles, calibration, PCA, feature selection

- Added `src/more.py`: soft-voting and stacking ensembles vs base models, probability calibration report (Brier score + expected calibration error), PCA explained-variance summary, and feature ranking (ANOVA F-test or mutual information).
- New CLI commands: `ensemble`, `calibrate`, `pca`, `select`.
- Added `src/explain.py`: dependency-free SHAP-style explanations (Monte Carlo Shapley values that sum exactly to prediction − baseline) with local and global views.
- API: new `/explain`, `/feature-ranking` and `/pca` endpoints (same optional API-key auth); CLI `explain`.
- Dashboard: new "Ensembles & features" and "SHAP-style explain" tabs.
- Notebook 34; `tests/test_more.py` (5 new tests).

## 3.3.0 — data quality, drift, anomalies, costs, API security

- Added `src/advanced.py`: data-quality profiling, drift detection (PSI + KS test), Isolation Forest anomaly scores, and cost-sensitive threshold tuning from out-of-fold probabilities.
- New CLI commands: `profile`, `drift`, `anomaly`, `threshold`.
- Dashboard: new "Quality & risk" tab (anomalies, drift between two CSVs, cost-based threshold).
- API: optional API-key authentication (`ML_API_KEY` env var + `X-API-Key` header, constant-time comparison).
- Notebook 33; 50 automated tests. Fixed a missing import found by the new CLI tests.

## 3.2.0 — text, API and notebook repairs

- Added `src/text.py`: TF-IDF text classification (logistic regression, naive Bayes, linear SVM) on the bundled SMS spam data, with top-term inspection.
- Added FastAPI service (`api/main.py`) with `/health`, `/info`, `/predict`, `/predict-text`; request validation and batch/length limits; framework-free logic in `src/serving.py`; models are checksum-verified.
- Added CLI `text-compare` and `text-predict`, a dashboard "Text spam" tab, notebook 32, `requirements-api.txt`, docker-compose, non-root Docker user.
- Repaired the legacy 2017 notebooks: replaced removed scikit-learn/matplotlib APIs (`get_feature_names`, `sklearn.externals.six`, `metrics.scorer`, `neighbors.kde`, `affinity`, `cm.spectral`, `IsolationForest.threshold_`, positional `DummyClassifier`, pandas bool dummies), replaced the vendored private tree exporter with `sklearn.tree.plot_tree`, fixed lesson solutions, added `ipywidgets` to requirements, and added a clear IMDb-download guard to notebook 23.
- Added a regression test that fails if removed APIs reappear in notebooks, solutions or figure helpers.

## 3.1.0 — more features

- Added hyperparameter tuning (`tune_model`, grid or random search with held-out test score).
- Added regression (diabetes dataset: Ridge, Random Forest, HistGradientBoosting) with R², MAE, RMSE.
- Added KMeans silhouette analysis, training on your own CSV (`train_from_csv`), and Markdown model reports.
- New CLI commands: `tune`, `regress`, `cluster`, `report`, `csv`.
- Dashboard: new Tune, Regression, Clustering and Your CSV tabs plus report download.
- Notebooks 30 and 31; 35 automated tests.

## 3.0.0 — 2026 feature release

- Added model registry (Logistic Regression, Random Forest, HistGradientBoosting, SVM, KNN) and dataset registry (Iris, Wine, Breast Cancer, Digits).
- Added cross-validated `compare_models` leaderboard and held-out permutation importance.
- Added `ml-portfolio` CLI (`train`, `compare`, `predict`, `runs`) via `python -m src.cli`.
- Model persistence now writes a `.meta.json` sidecar with SHA-256 checksum; `load_model` rejects tampered files.
- Added JSON Lines experiment tracking (`log_experiment`, `read_experiments`).
- Rebuilt Streamlit dashboard with Results, Leaderboard, Explain, Predict and Experiment-log tabs, ROC/PR curves, learning curves and CSV download.
- Added notebooks 28 (leaderboard) and 29 (experiment tracking).
- Tests expanded from 5 to 28 cases; CI adds Python 3.13, ruff linting, coverage and a CLI smoke test.
- Added Dockerfile, Makefile, pre-commit config, Dependabot and `requirements-dev.txt`.
- `fetch_data.py`: removed Python 2 fallbacks, switched to HTTPS and safe tar extraction (`filter="data"`).

## 2.1.0 — 2026 portfolio upgrade

- Added reusable, reproducible Iris pipeline utilities.
- Added pytest coverage for model training, metrics, model serialization, and notebook JSON integrity.
- Added GitHub Actions CI for Python 3.11 and 3.12.
- Added Streamlit portfolio dashboard with three bundled datasets and two classifier choices.
- Added runnable Iris and Breast Cancer examples.
- Added architecture documentation and a generated architecture image.
- Added GitHub upload checklist, contribution guidance, and security notes.

## 2.0.0 — 2026 modernization

- Added updated setup documentation, new notebooks, and an Iris demo.
- Retained original course materials for learning context.

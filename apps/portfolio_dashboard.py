"""Interactive local Streamlit dashboard (v3, 2026): streamlit run apps/portfolio_dashboard.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import LearningCurveDisplay

from src.advanced import detect_anomalies, detect_drift, profile_dataframe, tune_threshold
from src.csvtools import csv_imbalance, explain_csv_row
from src.explain import dataset_background, explain_regression, explain_row, global_importance
from src.extras import (
    PARAM_GRIDS,
    REGRESSORS,
    cluster_analysis,
    compare_regressors,
    generate_report,
    train_from_csv,
    train_regressor,
    tune_model,
)
from src.forecast import demo_series, forecast_series, load_series
from src.htmlreport import build_comparison_html
from src.imbalance import compare_imbalance_strategies
from src.ml_portfolio import (
    DATASETS,
    MODELS,
    build_model,
    compare_models,
    feature_importance,
    load_dataset,
    log_experiment,
    read_experiments,
    save_model,
    train_model,
)
from src.more import calibration_report, compare_ensembles, pca_summary, select_features
from src.text import predict_texts, top_terms, train_text_model

st.set_page_config(page_title="ML Portfolio | 2026", page_icon="📊", layout="wide")
st.title("Machine Learning Portfolio")
st.caption("2026 Edition v3 · Compare, explain, predict, track")

with st.sidebar:
    st.header("Experiment settings")
    dataset_name = st.selectbox("Dataset", sorted(DATASETS), format_func=lambda s: s.replace("_", " ").title())
    model_name = st.selectbox("Model", sorted(MODELS), format_func=lambda s: s.replace("_", " ").title())
    test_size = st.slider("Test set (%)", 15, 40, 25, 5)
    seed = int(st.number_input("Random seed", 0, 9999, 42))
    cv = st.slider("CV folds (leaderboard)", 3, 10, 5)


@st.cache_data(show_spinner=False)
def run_training(dataset, model, test, rs):
    result = train_model(dataset, model, test / 100, rs)
    return {k: v for k, v in result.items() if k != "model"}, result["model"]


@st.cache_data(show_spinner=False)
def run_leaderboard(dataset, folds, rs):
    return compare_models(dataset, cv=folds, random_state=rs)


info, model = run_training(dataset_name, model_name, test_size, seed)
data = load_dataset(dataset_name)
proba = hasattr(model, "predict_proba")
(tab_results, tab_compare, tab_explain, tab_predict, tab_runs,
 tab_tune, tab_reg, tab_cluster, tab_csv, tab_text, tab_quality, tab_more, tab_shap, tab_fc) = st.tabs(
    ["Results", "Leaderboard", "Explain", "Predict", "Experiment log",
     "Tune", "Regression", "Clustering", "Your CSV", "Text spam", "Quality & risk", "Ensembles & features", "SHAP-style explain", "Forecast & report"])

with tab_results:
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{info['accuracy']:.1%}")
    b.metric("Macro F1", f"{info['macro_f1']:.1%}")
    c.metric("Training rows", f"{len(info['y_train']):,}")
    d.metric("Test rows", f"{len(info['y_test']):,}")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(info["y_test"], info["predictions"], ax=ax, colorbar=False)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(info["report"]).T.round(3), width="stretch")
        if len(info["target_names"]) == 2 and proba:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            RocCurveDisplay.from_estimator(model, info["X_test"], info["y_test"], ax=axes[0])
            PrecisionRecallDisplay.from_estimator(model, info["X_test"], info["y_test"], ax=axes[1])
            st.pyplot(fig, clear_figure=True)
    if st.button("Save model + log this run"):
        path = save_model(model, "artifacts/dashboard_model.joblib",
                          {"dataset": dataset_name, "model": model_name, "accuracy": info["accuracy"]})
        log_experiment({**info, "dataset": dataset_name, "model_name": model_name})
        st.success(f"Saved to {path} with checksum sidecar and logged the run.")

with tab_compare:
    st.subheader(f"Cross-validated leaderboard · {cv} folds")
    board = run_leaderboard(dataset_name, cv, seed)
    st.dataframe(board.round(4), width="stretch")
    st.bar_chart(board.set_index("model")["accuracy_mean"])
    st.download_button("Download leaderboard (CSV)", board.to_csv(index=False), "leaderboard.csv", "text/csv")

with tab_explain:
    st.subheader("Permutation importance (held-out data)")
    full = train_model(dataset_name, model_name, test_size / 100, seed)
    imp = feature_importance(full, n_repeats=5, random_state=seed)
    st.bar_chart(imp.head(15).set_index("feature")["importance_mean"])
    st.dataframe(imp.round(4), width="stretch")
    st.subheader("Learning curve")
    fig, ax = plt.subplots(figsize=(6, 4))
    LearningCurveDisplay.from_estimator(build_model(model_name, seed), data.data, data.target, cv=3, ax=ax)
    st.pyplot(fig, clear_figure=True)

with tab_predict:
    st.subheader("Try your own input")
    frame = pd.DataFrame(data.data, columns=getattr(data, "feature_names", None))
    if frame.shape[1] > 30:
        st.info("This dataset has many features; showing the first 12 and using medians for the rest.")
    shown = list(frame.columns[:12]) if frame.shape[1] > 30 else list(frame.columns)
    row = frame.median().copy()
    cols = st.columns(4)
    for i, name in enumerate(shown):
        row[name] = cols[i % 4].number_input(str(name), float(frame[name].min()), float(frame[name].max()),
                                              float(frame[name].median()))
    if st.button("Predict", type="primary"):
        sample = pd.DataFrame([row.values], columns=frame.columns)
        pred = int(model.predict(sample.values)[0])
        st.success(f"Predicted class: {info['target_names'][pred]}")
        if proba:
            st.bar_chart(pd.Series(model.predict_proba(sample.values)[0], index=info["target_names"]))

with tab_runs:
    st.subheader("Logged experiments")
    runs = read_experiments()
    st.dataframe(runs, width="stretch") if not runs.empty else st.write("No runs logged yet.")

with tab_tune:
    st.subheader("Hyperparameter search")
    tune_choice = st.selectbox("Model to tune", sorted(PARAM_GRIDS))
    method = st.radio("Search", ["grid", "random"], horizontal=True)
    if st.button("Run search"):
        with st.spinner("Searching..."):
            tuned = tune_model(dataset_name, tune_choice, cv=min(cv, 5), method=method, random_state=seed)
        a, b = st.columns(2)
        a.metric("CV score", f"{tuned['cv_score']:.1%}")
        b.metric("Held-out test score", f"{tuned['test_score']:.1%}")
        st.json({k: str(v) for k, v in tuned["best_params"].items()})
        st.dataframe(tuned["results"].astype({"params": str}), width="stretch")

with tab_reg:
    st.subheader("Regression leaderboard · diabetes dataset")
    st.dataframe(compare_regressors(cv=cv, random_state=seed).round(4), width="stretch")
    reg_name = st.selectbox("Regressor", ["ridge", "random_forest", "hist_gradient_boosting"])
    reg = train_regressor(reg_name, test_size / 100, seed)
    a, b, c = st.columns(3)
    a.metric("R²", f"{reg['r2']:.3f}")
    b.metric("MAE", f"{reg['mae']:.1f}")
    c.metric("RMSE", f"{reg['rmse']:.1f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(reg["y_test"], reg["predictions"], alpha=0.7)
    lims = [reg["y_test"].min(), reg["y_test"].max()]
    ax.plot(lims, lims, "--", color="gray")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig, clear_figure=True)

with tab_cluster:
    st.subheader("KMeans: choosing k")
    table = cluster_analysis(dataset_name, range(2, 9), seed)
    st.line_chart(table.set_index("k")["silhouette"])
    st.dataframe(table.round(4), width="stretch")
    st.caption("Higher silhouette means better-separated clusters; look for a peak, and sanity-check against domain knowledge.")

with tab_csv:
    st.subheader("Train on your own CSV")
    st.caption("The file is processed in this session only. Numeric and text columns and missing values are handled automatically.")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        import tempfile
        frame_up = pd.read_csv(upload)
        st.dataframe(frame_up.head(), width="stretch")
        target_col = st.selectbox("Target column (class to predict)", list(frame_up.columns))
        if st.button("Train on my data"):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as handle:
                frame_up.to_csv(handle, index=False)
            try:
                out = train_from_csv(handle.name, target_col, test_size / 100, seed)
                st.success(f"Held-out accuracy {out['accuracy']:.1%} on {out['n_rows']:,} rows")
                st.write(f"Numeric: {out['numeric']} · Categorical: {out['categorical']}")
            except ValueError as exc:
                st.error(str(exc))
        csv_row = st.number_input("Row to explain", 0, max(len(frame_up) - 1, 0), 0, key="csv_row")
        if st.button("Explain this row (SHAP-style)"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as handle2:
                    frame_up.to_csv(handle2, index=False)
                ex_csv = explain_csv_row(handle2.name, target_col, int(csv_row), 10, 30, seed)
                st.metric(f"P({ex_csv['predicted_class']})", f"{ex_csv['prediction']:.3f}",
                          f"{ex_csv['prediction'] - ex_csv['base_value']:+.3f} vs baseline")
                st.bar_chart(ex_csv["table"].set_index("feature")[["contribution"]])
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Check class imbalance strategies"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as handle3:
                    frame_up.to_csv(handle3, index=False)
                st.dataframe(csv_imbalance(handle3.name, target_col, "logistic_regression", seed).round(4),
                             width="stretch")
            except ValueError as exc:
                st.error(str(exc))
    st.download_button("Download Markdown model report", generate_report(dataset_name, cv=min(cv, 5), random_state=seed),
                       f"report_{dataset_name}.md", "text/markdown")

with tab_text:
    st.subheader("SMS spam classifier (TF-IDF)")
    text_model_name = st.selectbox("Text model", ["logistic_regression", "naive_bayes", "linear_svm"])
    text_result = train_text_model(text_model_name, test_size / 100, seed)
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{text_result['accuracy']:.1%}")
    b.metric("Spam precision", f"{text_result['spam_precision']:.1%}")
    c.metric("Spam recall", f"{text_result['spam_recall']:.1%}")
    message = st.text_area("Try a message", "WINNER!! Claim your free prize now, call 0800")
    if message.strip():
        st.write(predict_texts(text_result["model"], [message])[0])
    st.write(top_terms(text_result["model"], 10))

with tab_quality:
    st.subheader("Anomalies in the selected dataset")
    contamination = st.slider("Expected anomaly share", 0.01, 0.20, 0.05, 0.01)
    found = detect_anomalies(data.data, contamination, seed)
    st.metric("Rows flagged", int(found["is_anomaly"].sum()))
    st.dataframe(found.head(15).round(4), width="stretch")
    st.subheader("Data drift: reference vs. new CSV")
    ref_file = st.file_uploader("Reference CSV", type="csv", key="ref")
    cur_file = st.file_uploader("New CSV", type="csv", key="cur")
    if ref_file is not None and cur_file is not None:
        ref_df, cur_df = pd.read_csv(ref_file), pd.read_csv(cur_file)
        try:
            st.dataframe(detect_drift(ref_df, cur_df), width="stretch")
            st.json(profile_dataframe(cur_df))
        except ValueError as exc:
            st.error(str(exc))
    if len(set(data.target)) == 2:
        st.subheader("Cost-sensitive decision threshold")
        cfn = st.number_input("Cost of a false negative", 0.1, 100.0, 5.0)
        cfp = st.number_input("Cost of a false positive", 0.1, 100.0, 1.0)
        res = tune_threshold(dataset_name, model_name if hasattr(build_model(model_name), "predict_proba") else "logistic_regression",
                             cfn, cfp, cv=min(cv, 5), random_state=seed)
        st.metric("Best threshold", f"{res['best_threshold']:.2f}", f"cost {res['best_cost']:.0f} vs {res['default_cost']:.0f} at 0.50")
        st.line_chart(res["table"].set_index("threshold")[["cost"]])
        st.caption(res["note"])

with tab_more:
    st.subheader("Ensembles vs base models")
    st.dataframe(compare_ensembles(dataset_name, min(cv, 5), seed).round(4), width="stretch")
    st.subheader("PCA explained variance")
    target_var = st.slider("Variance to keep", 0.50, 0.99, 0.95, 0.01)
    pca_res = pca_summary(dataset_name, target_var)
    st.metric("Components needed", pca_res["components_needed"], f"of {pca_res['n_features']} features")
    st.line_chart(pd.DataFrame({"cumulative variance": pca_res["cumulative"]}))
    st.subheader("Feature ranking")
    rank_method = st.radio("Method", ["f_test", "mutual_info"], horizontal=True)
    top_k = st.slider("Top k", 1, data.data.shape[1], min(5, data.data.shape[1]))
    st.dataframe(select_features(dataset_name, top_k, rank_method).round(4), width="stretch")
    if len(set(data.target)) == 2:
        st.subheader("Probability calibration")
        cal_method = st.radio("Calibration method", ["sigmoid", "isotonic"], horizontal=True)
        st.json(calibration_report(dataset_name, "random_forest", cal_method, seed))
        st.subheader("Class-imbalance strategies")
        frac = st.slider("Minority share of the data", 0.02, 0.40, 0.10, 0.01)
        st.dataframe(compare_imbalance_strategies(dataset_name, frac, "logistic_regression", seed).round(4),
                     width="stretch")
        st.caption("Resampling is applied to the training split only; metrics are for the minority class.")

with tab_shap:
    with st.expander("Explain a regression prediction (diabetes)"):
        reg_name = st.selectbox("Regressor", sorted(REGRESSORS), key="reg_explain_model")
        reg = train_regressor(reg_name, 0.25, seed)
        reg_row = st.number_input("Test row", 0, len(reg["X_test"]) - 1, 0, key="reg_explain_row")
        rex = explain_regression(reg["model"], reg["X_test"][int(reg_row)], reg["X_test"],
                                 reg["feature_names"], random_state=seed)
        st.metric("Predicted value", f"{rex['prediction']:.1f}",
                  f"{rex['prediction'] - rex['base_value']:+.1f} vs baseline {rex['base_value']:.1f}")
        st.bar_chart(rex["table"].set_index("feature")[["contribution"]])
    st.subheader("Why did the model predict this?")
    if not proba:
        st.info("Pick a model with probabilities (e.g. logistic regression, random forest, SVM).")
    else:
        feat_names = [str(n) for n in getattr(data, "feature_names", [])] or None
        background = dataset_background(dataset_name, 50, seed)
        row_idx = st.number_input("Row index", 0, len(data.data) - 1, 0)
        perms = st.slider("Sampling permutations (more = smoother)", 10, 100, 30, 10)
        res = explain_row(model, data.data[int(row_idx)], background, feat_names,
                          n_permutations=perms, random_state=seed, top_k=12)
        label = info["target_names"][res["class_index"]]
        st.metric(f"P({label})", f"{res['prediction']:.3f}", f"{res['prediction'] - res['base_value']:+.3f} vs baseline {res['base_value']:.3f}")
        st.bar_chart(res["table"].set_index("feature")[["contribution"]])
        st.dataframe(res["table"].round(4), width="stretch")
        if st.button("Compute global importance (mean |contribution|)"):
            st.dataframe(global_importance(model, data.data, background, feat_names,
                                           n_rows=15, n_permutations=15, random_state=seed).head(15).round(4),
                         width="stretch")

with tab_fc:
    st.subheader("Time-series forecast")
    fc_file = st.file_uploader("CSV with a numeric column (optional date column)", type="csv", key="fc_csv")
    try:
        if fc_file is None:
            series = demo_series()
            st.caption("Using a bundled synthetic monthly series. Upload a CSV to forecast your own data.")
        else:
            fc_df = pd.read_csv(fc_file)
            num_cols = fc_df.select_dtypes(include="number").columns.tolist()
            value_col = st.selectbox("Value column", num_cols)
            date_choice = st.selectbox("Date column (optional)", ["(none)"] + list(fc_df.columns))
            series = load_series(fc_df, value_col, None if date_choice == "(none)" else date_choice)
        fc_h = st.slider("Horizon (steps)", 1, 36, 12)
        fc_res = forecast_series(series, fc_h, "auto", None, 3, seed)
        st.metric("Chosen model", fc_res["model"], f"season {fc_res['season']}")
        joined = pd.concat([series.rename("history"), fc_res["forecast"]], axis=1)
        st.line_chart(joined[["history", "forecast", "lower", "upper"]])
        st.dataframe(fc_res["backtest"].round(3), width="stretch")
        st.caption(fc_res["note"])
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
    st.subheader("HTML model-comparison report")
    st.download_button("Download HTML report", build_comparison_html(dataset_name, cv=min(cv, 5), random_state=seed),
                       f"comparison_{dataset_name}.html", "text/html")

st.info("Educational demo only. Results depend on dataset, split and model settings; not medical or production advice.")

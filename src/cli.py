"""Command-line interface: python -m src.cli {train,compare,predict,runs,tune,regress,cluster,report,csv,text-compare,text-predict,anomaly,drift,threshold,profile,ensemble,calibrate,pca,select,explain,imbalance,explain-regression,csv-explain,csv-imbalance,forecast,html-report}."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .advanced import detect_anomalies, detect_drift, profile_dataframe, tune_threshold
from .csvtools import csv_imbalance, explain_csv_row
from .explain import dataset_background, explain_regression, explain_row
from .extras import (
    PARAM_GRIDS,
    REGRESSORS,
    cluster_analysis,
    compare_regressors,
    generate_report,
    train_from_csv,
    train_regressor,
    tune_model,
)
from .forecast import demo_series, forecast_series, load_series
from .htmlreport import write_comparison_html
from .imbalance import compare_imbalance_strategies
from .ml_portfolio import (
    DATASETS,
    MODELS,
    compare_models,
    feature_importance,
    load_dataset,
    load_model,
    log_experiment,
    read_experiments,
    save_model,
    train_model,
)
from .more import calibration_report, compare_ensembles, pca_summary, select_features
from .text import compare_text_models, predict_texts, train_text_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-portfolio", description="2026 ML portfolio CLI")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train, evaluate, save and log a model")
    t.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    t.add_argument("--model", choices=sorted(MODELS), default="logistic_regression")
    t.add_argument("--test-size", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", default="artifacts/model.joblib")
    t.add_argument("--importance", action="store_true", help="print permutation importance")

    c = sub.add_parser("compare", help="cross-validated model leaderboard")
    c.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    c.add_argument("--cv", type=int, default=5)
    c.add_argument("--seed", type=int, default=42)

    pr = sub.add_parser("predict", help="predict with a saved model")
    pr.add_argument("--model-path", default="artifacts/model.joblib")
    pr.add_argument("--features", required=True, help="comma-separated feature values")

    sub.add_parser("runs", help="show logged experiments")

    tu = sub.add_parser("tune", help="hyperparameter search")
    tu.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    tu.add_argument("--model", choices=sorted(PARAM_GRIDS), default="logistic_regression")
    tu.add_argument("--method", choices=["grid", "random"], default="grid")
    tu.add_argument("--cv", type=int, default=5)

    sub.add_parser("regress", help="regression leaderboard on the diabetes dataset")

    cl = sub.add_parser("cluster", help="KMeans silhouette analysis")
    cl.add_argument("--dataset", choices=sorted(DATASETS), default="iris")

    rp = sub.add_parser("report", help="write a Markdown model report")
    rp.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    rp.add_argument("--out", default="artifacts/report.md")

    tx = sub.add_parser("text-compare", help="compare TF-IDF text classifiers on the SMS spam data")
    tx.add_argument("--cv", type=int, default=5)

    tp = sub.add_parser("text-predict", help="classify messages as spam/ham")
    tp.add_argument("--text", action="append", required=True, help="message (repeat for several)")

    an = sub.add_parser("anomaly", help="Isolation Forest anomaly scores for a bundled dataset or CSV")
    an.add_argument("--dataset", choices=sorted(DATASETS), default="wine")
    an.add_argument("--contamination", type=float, default=0.05)
    an.add_argument("--top", type=int, default=5)

    dr = sub.add_parser("drift", help="compare two CSV files for data drift (PSI + KS)")
    dr.add_argument("--reference", required=True)
    dr.add_argument("--current", required=True)

    th = sub.add_parser("threshold", help="cost-sensitive decision threshold (binary datasets)")
    th.add_argument("--dataset", choices=sorted(DATASETS), default="breast_cancer")
    th.add_argument("--model", choices=sorted(MODELS), default="logistic_regression")
    th.add_argument("--cost-fn", type=float, default=5.0)
    th.add_argument("--cost-fp", type=float, default=1.0)

    pf = sub.add_parser("profile", help="data-quality profile of a CSV file")
    pf.add_argument("--path", required=True)
    pf.add_argument("--target")

    en = sub.add_parser("ensemble", help="base models vs voting and stacking ensembles")
    en.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    en.add_argument("--cv", type=int, default=5)

    ca = sub.add_parser("calibrate", help="Brier score / ECE before and after calibration (binary data)")
    ca.add_argument("--dataset", choices=sorted(DATASETS), default="breast_cancer")
    ca.add_argument("--model", choices=sorted(MODELS), default="random_forest")
    ca.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")

    pc = sub.add_parser("pca", help="PCA explained-variance summary")
    pc.add_argument("--dataset", choices=sorted(DATASETS), default="wine")
    pc.add_argument("--variance", type=float, default=0.95)

    se = sub.add_parser("select", help="rank features (F-test or mutual information)")
    se.add_argument("--dataset", choices=sorted(DATASETS), default="wine")
    se.add_argument("--k", type=int, default=5)
    se.add_argument("--method", choices=["f_test", "mutual_info"], default="f_test")

    ex = sub.add_parser("explain", help="SHAP-style feature contributions for one row of a bundled dataset")
    ex.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    ex.add_argument("--model", choices=sorted(MODELS), default="logistic_regression")
    ex.add_argument("--row", type=int, default=0, help="row index in the dataset")
    ex.add_argument("--top", type=int, default=5)

    im = sub.add_parser("imbalance", help="compare class-imbalance strategies on a skewed binary dataset")
    im.add_argument("--dataset", choices=sorted(DATASETS), default="breast_cancer")
    im.add_argument("--minority-fraction", type=float, default=0.1)
    im.add_argument("--model", choices=["logistic_regression", "random_forest"], default="logistic_regression")

    er = sub.add_parser("explain-regression", help="SHAP-style contributions for the diabetes regressors")
    er.add_argument("--model", choices=sorted(REGRESSORS), default="ridge")
    er.add_argument("--row", type=int, default=0, help="row index in the held-out test split")
    er.add_argument("--top", type=int, default=5)

    ce = sub.add_parser("csv-explain", help="train on your CSV and explain one row (SHAP-style)")
    ce.add_argument("--path", required=True)
    ce.add_argument("--target", required=True)
    ce.add_argument("--row", type=int, default=0)
    ce.add_argument("--top", type=int, default=8)

    ci = sub.add_parser("csv-imbalance", help="compare imbalance strategies on your binary-target CSV")
    ci.add_argument("--path", required=True)
    ci.add_argument("--target", required=True)
    ci.add_argument("--model", choices=["logistic_regression", "random_forest"], default="logistic_regression")

    fc = sub.add_parser("forecast", help="time-series forecast with backtest (demo series if --path omitted)")
    fc.add_argument("--path", help="CSV file; omit to use a bundled synthetic series")
    fc.add_argument("--value", default="value", help="numeric column to forecast")
    fc.add_argument("--date", help="optional date column")
    fc.add_argument("--horizon", type=int, default=12)
    fc.add_argument("--season", type=int, help="seasonal period (auto-detected if omitted)")
    fc.add_argument("--model", default="auto")

    hr = sub.add_parser("html-report", help="write a self-contained HTML model-comparison report")
    hr.add_argument("--dataset", choices=sorted(DATASETS), default="iris")
    hr.add_argument("--cv", type=int, default=5)
    hr.add_argument("--out", default="artifacts/report.html")
    hr.add_argument("--no-regression", action="store_true")

    cs = sub.add_parser("csv", help="train on your own CSV file")
    cs.add_argument("--path", required=True)
    cs.add_argument("--target", required=True)
    cs.add_argument("--out", default="artifacts/csv_model.joblib")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            r = train_model(args.dataset, args.model, args.test_size, args.seed)
            save_model(r["model"], args.out, {"dataset": args.dataset, "model": args.model,
                                              "accuracy": r["accuracy"]})
            log_experiment(r)
            print(f"{args.dataset} / {args.model}: accuracy={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f}")
            print(f"Saved model to {args.out}")
            if args.importance:
                print(feature_importance(r).head(10).round(4).to_string(index=False))
        elif args.command == "compare":
            print(compare_models(args.dataset, cv=args.cv, random_state=args.seed).round(4).to_string(index=False))
        elif args.command == "predict":
            values = [float(v) for v in args.features.split(",")]
            model = load_model(args.model_path)
            pred = model.predict([values])[0]
            out = {"prediction": int(pred)}
            if hasattr(model, "predict_proba"):
                out["probabilities"] = [round(float(x), 4) for x in model.predict_proba([values])[0]]
            print(json.dumps(out))
        elif args.command == "tune":
            r = tune_model(args.dataset, args.model, args.cv, args.method)
            print(f"best params: {r['best_params']}")
            print(f"cv score={r['cv_score']:.3f}  held-out test score={r['test_score']:.3f}")
        elif args.command == "regress":
            print(compare_regressors().round(4).to_string(index=False))
        elif args.command == "cluster":
            print(cluster_analysis(args.dataset).round(4).to_string(index=False))
        elif args.command == "report":
            generate_report(args.dataset, path=args.out)
            print(f"Report written to {args.out}")
        elif args.command == "text-compare":
            print(compare_text_models(args.cv).round(4).to_string(index=False))
        elif args.command == "text-predict":
            result = train_text_model()
            for msg, pred in zip(args.text, predict_texts(result["model"], args.text), strict=True):
                print(json.dumps({"text": msg[:60], **pred}))
        elif args.command == "anomaly":
            res = detect_anomalies(load_dataset(args.dataset).data, args.contamination)
            print(f"{int(res['is_anomaly'].sum())} of {len(res)} rows flagged; top {args.top}:")
            print(res.head(args.top).round(4).to_string(index=False))
        elif args.command == "drift":
            print(detect_drift(pd.read_csv(args.reference), pd.read_csv(args.current)).to_string(index=False))
        elif args.command == "threshold":
            r = tune_threshold(args.dataset, args.model, args.cost_fn, args.cost_fp)
            print(f"best threshold={r['best_threshold']:.2f} cost={r['best_cost']:.1f} (at 0.50: {r['default_cost']:.1f})")
            print(r["note"])
        elif args.command == "profile":
            print(json.dumps(profile_dataframe(pd.read_csv(args.path), args.target), indent=2))
        elif args.command == "csv":
            r = train_from_csv(args.path, args.target)
            save_model(r["model"], args.out, {"source": args.path, "target": args.target, "accuracy": r["accuracy"]})
            print(f"rows={r['n_rows']} accuracy={r['accuracy']:.3f} classes={r['classes']}")
            print(f"numeric={r['numeric']} categorical={r['categorical']}\nSaved model to {args.out}")
        elif args.command == "ensemble":
            print(compare_ensembles(args.dataset, args.cv).round(4).to_string(index=False))
        elif args.command == "calibrate":
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                              for k, v in calibration_report(args.dataset, args.model, args.method).items()}, indent=2))
        elif args.command == "pca":
            print(json.dumps(pca_summary(args.dataset, args.variance), indent=2))
        elif args.command == "select":
            print(select_features(args.dataset, args.k, args.method).round(4).to_string(index=False))
        elif args.command == "explain":
            data = load_dataset(args.dataset)
            if not 0 <= args.row < len(data.data):
                raise ValueError(f"row must be between 0 and {len(data.data) - 1}")
            r = train_model(args.dataset, args.model)
            ex = explain_row(r["model"], data.data[args.row], dataset_background(args.dataset),
                             r["feature_names"] or None, top_k=args.top)
            print(f"P({r['target_names'][ex['class_index']]})={ex['prediction']:.3f} (baseline {ex['base_value']:.3f})")
            print(ex["table"].round(4).to_string(index=False))
        elif args.command == "imbalance":
            print(compare_imbalance_strategies(args.dataset, args.minority_fraction, args.model)
                  .round(4).to_string(index=False))
        elif args.command == "explain-regression":
            r = train_regressor(args.model)
            if not 0 <= args.row < len(r["X_test"]):
                raise ValueError(f"row must be between 0 and {len(r['X_test']) - 1}")
            ex = explain_regression(r["model"], r["X_test"][args.row], r["X_test"], r["feature_names"],
                                    top_k=args.top)
            print(f"prediction={ex['prediction']:.1f} (baseline {ex['base_value']:.1f}, actual {r['y_test'][args.row]:.1f})")
            print(ex["table"].round(4).to_string(index=False))
        elif args.command == "csv-explain":
            ex = explain_csv_row(args.path, args.target, args.row, args.top)
            print(f"row {args.row}: predicted {ex['predicted_class']!r} P={ex['prediction']:.3f} "
                  f"(baseline {ex['base_value']:.3f}; model held-out accuracy {ex['accuracy']:.3f})")
            print(ex["table"].round(4).to_string(index=False))
        elif args.command == "csv-imbalance":
            print(csv_imbalance(args.path, args.target, args.model).round(4).to_string(index=False))
        elif args.command == "forecast":
            y = demo_series() if args.path is None else load_series(pd.read_csv(args.path), args.value, args.date)
            res = forecast_series(y, args.horizon, args.model, args.season)
            print(f"model={res['model']} season={res['season']}")
            print(res["backtest"].round(3).to_string(index=False))
            print(res["forecast"].round(2).to_string())
            print(res["note"])
        elif args.command == "html-report":
            out = write_comparison_html(args.out, args.dataset, args.cv, not args.no_regression)
            print(f"HTML report written to {out}")
        elif args.command == "runs":
            df = read_experiments()
            print(df.to_string(index=False) if not df.empty else "No experiments logged yet.")
    except (ValueError, FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

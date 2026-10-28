"""v3.7: self-contained HTML model-comparison report (no external files, scripts or fonts)."""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .extras import compare_regressors
from .ml_portfolio import compare_models
from .more import compare_ensembles, pca_summary, select_features

__all__ = ["build_comparison_html", "write_comparison_html"]

_CSS = """
:root{--bg:#fff;--fg:#1c1f23;--muted:#5f6670;--card:#f5f6f8;--bar:#3b6fd4;--best:#1f9d55;--line:#d9dce1}
@media (prefers-color-scheme:dark){:root{--bg:#15181c;--fg:#e8eaed;--muted:#9aa1ab;--card:#1f2328;--bar:#6c9cf0;--best:#46c37b;--line:#33383f}}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:900px;margin:0 auto;padding:16px;line-height:1.5}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:1.8rem;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}.wrap{overflow-x:auto}.muted{color:var(--muted);font-size:.85rem}
.card{background:var(--card);border-radius:8px;padding:10px 14px;margin:10px 0}svg text{fill:var(--fg);font-size:12px}
"""


def _table(df: pd.DataFrame, digits: int = 4) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for rec in df.itertuples(index=False):
        cells = "".join(f"<td>{html.escape(f'{v:.{digits}f}' if isinstance(v, float) else str(v))}</td>" for v in rec)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="wrap"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


def _bars(labels, values, errors=None, vmax: float | None = None, best_first: bool = True) -> str:
    vmax = vmax or max(max(values), 1e-9)
    h, left, width = 26, 190, 560
    parts = [f'<svg viewBox="0 0 {left + width + 70} {h * len(values) + 8}" width="100%" role="img" '
             f'aria-label="bar chart">']
    for i, (lab, val) in enumerate(zip(labels, values, strict=False)):
        y = 4 + i * h
        w = max(1.0, width * val / vmax)
        colour = "var(--best)" if (i == 0 and best_first) else "var(--bar)"
        parts.append(f'<text x="{left - 8}" y="{y + 16}" text-anchor="end">{html.escape(str(lab))}</text>'
                     f'<rect x="{left}" y="{y + 3}" width="{w:.1f}" height="{h - 8}" rx="3" fill="{colour}"/>')
        if errors is not None:
            e = width * errors[i] / vmax
            parts.append(f'<line x1="{left + w - e:.1f}" x2="{left + w + e:.1f}" y1="{y + h / 2 + 1}" '
                         f'y2="{y + h / 2 + 1}" stroke="var(--fg)" stroke-width="1.5"/>')
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + 16}">{val:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_comparison_html(dataset: str = "iris", cv: int = 5, include_regression: bool = True,
                          random_state: int = 42) -> str:
    """Build the full report as one HTML string."""
    board = compare_models(dataset, cv=cv, random_state=random_state)
    ens = compare_ensembles(dataset, cv=cv, random_state=random_state)
    pca = pca_summary(dataset)
    feats = select_features(dataset, k=min(5, pca["n_features"])).head(10)
    best = board.iloc[0]
    sections = [
        f"<h1>Model comparison report</h1><p class='muted'>Dataset: <b>{html.escape(dataset)}</b> · "
        f"{cv}-fold cross-validation · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
        "educational demo, not production advice</p>",
        f"<div class='card'>Best single model: <b>{html.escape(str(best['model']))}</b> with mean accuracy "
        f"<b>{best['accuracy_mean']:.3f}</b> (± {best['accuracy_std']:.3f}).</div>",
        "<h2>Leaderboard (accuracy, mean ± std across folds)</h2>",
        _bars(board["model"], board["accuracy_mean"], board["accuracy_std"], vmax=1.0),
        _table(board),
        "<h2>Ensembles vs base models</h2>",
        _bars(ens["model"], ens["accuracy_mean"], ens["accuracy_std"], vmax=1.0),
        "<p class='muted'>Ensembles help most when base models make different mistakes; "
        "on easy datasets the gain can be zero.</p>",
        f"<h2>PCA</h2><p>{pca['components_needed']} of {pca['n_features']} components explain "
        f"{pca['target_variance']:.0%} of the variance.</p>",
        "<h2>Top features (ANOVA F-test)</h2>", _table(feats[["feature", "score"]], 2),
    ]
    if include_regression:
        reg = compare_regressors(cv=min(cv, 5), random_state=random_state)
        sections += ["<h2>Regression leaderboard (diabetes, R²)</h2>",
                     _bars(reg["model"], reg["r2_mean"].clip(lower=0), reg["r2_std"], vmax=1.0), _table(reg)]
    sections.append("<p class='muted'>Cross-validated scores estimate generalisation; differences smaller than "
                    "the standard deviation are not meaningful.</p>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Model comparison – {html.escape(dataset)}</title><style>{_CSS}</style></head>"
            f"<body>{''.join(sections)}</body></html>")


def write_comparison_html(path: str | Path, dataset: str = "iris", cv: int = 5,
                          include_regression: bool = True) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_comparison_html(dataset, cv, include_regression), encoding="utf-8")
    return out

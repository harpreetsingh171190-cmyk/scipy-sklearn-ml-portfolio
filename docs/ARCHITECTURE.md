# Project architecture

```mermaid
flowchart TD
    A[Learning notebooks] --> B[Reusable utilities: src/]
    C[Bundled scikit-learn datasets] --> D[Training and evaluation]
    B --> D
    B --> K[CLI: train / compare / predict]
    D --> L[JSON experiment log]
    D --> E[Metrics and visualizations]
    D --> F[Saved model artifact]
    E --> G[Streamlit portfolio dashboard]
    H[pytest tests] --> I[GitHub Actions CI]
    I --> J[Quality checks on pushes and pull requests]
```

## Design principles

- Keep data preparation and estimators inside scikit-learn pipelines to reduce leakage.
- Fix random seeds for reproducible demonstrations.
- Test model behavior and notebook file integrity automatically.
- Use built-in example datasets by default to avoid network dependencies.
- Verify model files with a SHA-256 sidecar before loading.
- Treat model artifacts as trusted executable data: only load files from sources you trust.

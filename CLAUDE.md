# CLAUDE.md — Explainable anomaly detection (CICIDS2017)

S6 AI mini-project: explainable anomaly detection on network logs (CICIDS2017).
Model bake-off: compare 7 supervised classifiers by 10-fold cross-validation, show all metrics +
a confusion matrix for each, and keep the best (selected on F1). A static rule baseline is kept
for comparison. Streamlit dashboard. See `plan.md`.

---

## GOLDEN RULE — coding style

Follow the "Python - Partie 2" course style: **flat, procedural, notebook-style Python**.
**No functions, no classes, no type hints, no docstrings** in the notebooks.
Keep the code **simple, direct and readable — not fancy.** Use **few, short comments**.

**Language: English.** Variable names, prints, and plot titles/labels are all in English.
(Only exception for functions: `app/app.py` Streamlit, kept minimal + `@st.cache_resource`.)

## Imports (these aliases only)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
```

Import only what you use. `pd`, `np`, `plt`, `sns` — never other aliases.

## Naming
- DataFrame = `df` (`df_train`, `df_test`, `df_clean`).
- Features = `X`, target = `y`; `X_train, X_test, y_train, y_test = train_test_split(...)`.
- Feature/target lists at the top: `features = [...]`, `target = "Label_binary"`.
- Short instances: `scaler`, `rf`, `model`, `iso`, `grid`.
- `snake_case`, English names. Keep the dataset's original English column names.

## Comments
- **Few and short**, in English, with `#`. No docstrings.
- A one-line comment before a block is enough: `# load data`, `# train/test split`, `# evaluate`.
- Don't over-comment, don't number every step, don't write long markdown.

## Structure
- Top-to-bottom procedural scripts. No `def`, no `class`, no type hints, no `if __name__`.
- The only allowed structuring idiom is **dict of models + for loop**:
  ```python
  models = {"Random Forest": ..., "XGBoost": ...}
  for name, model in models.items():
      model.fit(X_train, y_train)
      print(f"{name}: {accuracy_score(y_test, model.predict(X_test)):.4f}")
  ```
- Notebooks authored as jupytext "percent" (`# %%` cells); a minimal `# %% [markdown]` title per
  section is fine. Absolute `BASE` path constant at the top.

## Idioms (pandas / sklearn)
```python
df = pd.read_parquet(BASE + "/data/processed/cicids_clean.parquet")
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- `random_state=42` everywhere.

## Plots
- seaborn over matplotlib. **English titles and labels.** Each plot ends with `plt.show()`.
  ```python
  plt.figure(figsize=(8, 6))
  sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
  plt.title("Correlation matrix")
  plt.show()
  ```

## ML / sklearn
- Default params unless needed. `model.fit(X_train, y_train)` → `model.predict(X_test)`.
- Metrics: `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`.
- Save/load: `joblib.dump(model, BASE + "/models/...")` / `joblib.load(...)`.

## Output
- f-strings, `:.4f` precision. `print()` to show results. English headers:
  `print("=== Model comparison ===")`.

## Don't
- ❌ No functions / classes / type hints / docstrings in notebooks (except minimal Streamlit).
- ❌ No French — English everywhere (names, prints, plot labels, comments).
- ❌ No long or numbered comment blocks — keep comments short and sparse.
- ❌ No `try/except`, no validation, no over-engineering — stay at the course level.

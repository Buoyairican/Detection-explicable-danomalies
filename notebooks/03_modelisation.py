# %% [markdown]
# # Phase 3 - Modeling: supervised model bake-off (CICIDS2017)
#
# Prep the data, build two interpretable baselines (static rules + Isolation
# Forest), then run a BAKE-OFF of 7 supervised models compared by 10-fold
# cross-validation on a stratified subsample. Each model is tuned with a small
# GridSearchCV and the best F1 wins. The winner is retrained on the FULL train
# and saved as a pipeline (StandardScaler + model) that takes RAW features. We
# also produce every artifact consumed by evaluation (Phase 4) and the Streamlit
# dashboard (Phase 5).

# %% [markdown]
# ## 1. Imports and loading the frozen parquet

# %%
# imports
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score
from xgboost import XGBClassifier
import joblib

sns.set(style="whitegrid")

# project root
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# load the cleaned dataset (Phase 1)
df = pd.read_parquet(BASE + "/data/processed/cicids_clean.parquet")
print(f"Loaded DataFrame shape: {df.shape}")
print(df["Label_binary"].value_counts())

# %% [markdown]
# ## 2. Features and target
#
# Features are every column except the three label columns. The target is
# binary: `Label_binary` (0 = BENIGN, 1 = attack).

# %%
# features = all columns except labels; target = Label_binary
label_cols = ["Label", "Label_binary", "Label_group"]
features = [c for c in df.columns if c not in label_cols]
target = "Label_binary"
print(f"Number of candidate features: {len(features)}")
print(f"Target column: {target}")

# %% [markdown]
# ## 3. Drop zero-variance features
#
# Constant columns (zero std) carry no information, so we drop them. We count
# how many are removed, then save the final ordered feature list.

# %%
# drop constant (zero-variance) features
stds = df[features].std()
constant_features = stds[stds == 0].index.tolist()
print(f"Number of constant (zero-variance) features dropped: {len(constant_features)}")
print(f"Dropped features: {constant_features}")

# final feature list (stable order)
features = [c for c in features if c not in constant_features]
print(f"Number of features kept: {len(features)}")

# save the ordered feature list
joblib.dump(features, BASE + "/models/features.pkl")
print("Feature list saved -> models/features.pkl")

# %% [markdown]
# ## 4. Stratified train/test split
#
# 70% train / 30% test, stratified on `Label_binary` to keep the same
# BENIGN/attack proportion in both sets.

# %%
# stratified train/test split on the binary target
df_train, df_test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Label_binary"]
)

# feature matrices and targets
X_train = df_train[features]
X_test = df_test[features]
y_train = df_train["Label_binary"]
y_test = df_test["Label_binary"]

print(f"Train: {X_train.shape[0]} rows")
print(f"Test:  {X_test.shape[0]} rows")
print(f"Attack ratio (train): {y_train.mean():.4f}")
print(f"Attack ratio (test):  {y_test.mean():.4f}")

# %% [markdown]
# ## 5. Standardize the features
#
# The `StandardScaler` is fit ONLY on train, then applied to train and test.
# This frozen scaler feeds the Isolation Forest baseline and stays a contract
# artifact (Phase 4). The bake-off pipelines carry their OWN StandardScaler and
# take RAW features.

# %%
# standardize: fit on train, transform train and test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Mean of features (scaled train): {X_train_s.mean():.4f}")
print(f"Std of features (scaled train): {X_train_s.std():.4f}")

# save the scaler
joblib.dump(scaler, BASE + "/models/scaler.pkl")
print("Scaler saved -> models/scaler.pkl")

# %% [markdown]
# ## 6. Static rule baseline (interpretable)
#
# A deliberately simplistic baseline: a few fixed thresholds on RAW
# volume/rate features (not scaled). Thresholds sit near the 95th/99th
# percentiles of BENIGN traffic. A row is flagged as attack if AT LEAST one
# rule fires (logical OR). We expect a high false-positive rate: that is the
# comparison point against the learned models.

# %%
# quick look at a few BENIGN volume/rate features
benign_train = X_train[y_train == 0]
for f in ["Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets",
          "Fwd Packet Length Max", "Destination Port", "Bwd Packets/s",
          "Average Packet Size", "Flow Duration"]:
    p95 = benign_train[f].quantile(0.95)
    p99 = benign_train[f].quantile(0.99)
    print(f"{f:25s} BENIGN p95={p95:.2f}  p99={p99:.2f}")

# %%
# thresholds (near BENIGN p95/p99) and the 5 rules
th_avg_packet_size = 496.0       # high average packet size (DoS/DDoS payload floods)
th_bwd_packets_s = 58823.0       # high backward packet rate (BENIGN p99)
th_flow_packets_s = 500000.0     # very high total packet rate (BENIGN p95)
th_fwd_packets_short = 3         # very few forward packets...
th_flow_duration_long = 1000000.0  # ...but a long flow (slow/probe connections)
th_flow_bytes_s = 12000000.0     # very high byte rate (BENIGN p99)

# individual rules on RAW features (X_test)
rule_avg_size = X_test["Average Packet Size"] > th_avg_packet_size
rule_bwd_rate = X_test["Bwd Packets/s"] > th_bwd_packets_s
rule_flow_rate = X_test["Flow Packets/s"] > th_flow_packets_s
rule_short_long = (X_test["Total Fwd Packets"] <= th_fwd_packets_short) & (X_test["Flow Duration"] > th_flow_duration_long)
rule_bytes_rate = X_test["Flow Bytes/s"] > th_flow_bytes_s

# logical OR of the 5 rules -> baseline prediction
y_pred_rule = (rule_avg_size | rule_bwd_rate | rule_flow_rate | rule_short_long | rule_bytes_rate).astype(int)

# rule baseline metrics on test
recall_rule = recall_score(y_test, y_pred_rule)
cm_rule = confusion_matrix(y_test, y_pred_rule)
# false positives = BENIGN (y=0) predicted attack (1) -> cell [0, 1]
fp_rule = int(cm_rule[0, 1])
print(f"\nRule baseline - Recall (attack): {recall_rule:.4f}")
print(f"Rule baseline - False positives (BENIGN flagged attack): {fp_rule}")
print(f"Rule baseline - False positive rate: {fp_rule / (y_test == 0).sum():.4f}")
print("\nConfusion matrix (rules):")
print(cm_rule)

# %%
# save the rules and thresholds to JSON
# NOTE: the JSON keys (logique/regles/operateur/seuil/justification) are the schema read by
# app/app.py (the dashboard) and notebook 04 - keep them as-is so the contract stays valid.
rule_baseline = {
    "description": "Static rule baseline on raw features. A row is flagged as attack if "
                   "AT LEAST one rule fires (logical OR). Thresholds chosen near the "
                   "95th/99th percentiles of BENIGN training traffic. Deliberately "
                   "simplistic baseline, high false positive rate expected.",
    "logique": "OR",
    "regles": [
        {"feature": "Average Packet Size", "operateur": ">", "seuil": th_avg_packet_size,
         "justification": "High average packet size typical of DoS/DDoS payload floods."},
        {"feature": "Bwd Packets/s", "operateur": ">", "seuil": th_bwd_packets_s,
         "justification": "High backward packet rate (99th percentile of BENIGN)."},
        {"feature": "Flow Packets/s", "operateur": ">", "seuil": th_flow_packets_s,
         "justification": "Very high total packet rate (95th percentile of BENIGN)."},
        {"feature_1": "Total Fwd Packets", "operateur_1": "<=", "seuil_1": th_fwd_packets_short,
         "feature_2": "Flow Duration", "operateur_2": ">", "seuil_2": th_flow_duration_long,
         "justification": "Very few forward packets but a long flow: probe/slow connections."},
        {"feature": "Flow Bytes/s", "operateur": ">", "seuil": th_flow_bytes_s,
         "justification": "Very high byte rate (99th percentile of BENIGN)."},
    ],
    "recall_attaque_test": round(float(recall_rule), 4),
    "faux_positifs_test": fp_rule,
}
with open(BASE + "/models/rule_baseline.json", "w") as f:
    json.dump(rule_baseline, f, indent=2, ensure_ascii=False)
print("Rules saved -> models/rule_baseline.json")

# %% [markdown]
# ## 7. Isolation Forest (learn normal behavior)
#
# The Isolation Forest learns the structure of NORMAL traffic: we train it only
# on BENIGN train rows (scaled features). At prediction time it returns -1 for
# an anomaly and 1 for a normal point; we convert -1 -> 1 (attack) and
# 1 -> 0 (BENIGN).

# %%
# Isolation Forest trained on BENIGN train rows only
X_train_s_benign = X_train_s[y_train.values == 0]
print(f"BENIGN train rows for training: {X_train_s_benign.shape[0]}")

iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_train_s_benign)

# predict on test: -1 = anomaly -> 1, 1 = normal -> 0
iso_pred_raw = iso.predict(X_test_s)
y_pred_iso = np.where(iso_pred_raw == -1, 1, 0)  # -1 anomaly -> attack (1), 1 normal -> BENIGN (0)

# metrics
recall_iso = recall_score(y_test, y_pred_iso)
cm_iso = confusion_matrix(y_test, y_pred_iso)
fp_iso = int(cm_iso[0, 1])
print(f"\nIsolation Forest - Recall (attack): {recall_iso:.4f}")
print(f"Isolation Forest - False positives: {fp_iso}")
print("Confusion matrix (Isolation Forest):")
print(cm_iso)

# save the model
joblib.dump(iso, BASE + "/models/isolation_forest.pkl")
print("Isolation Forest saved -> models/isolation_forest.pkl")

# %% [markdown]
# ## 8. Bake-off of 7 supervised models (selected by cross-validation)
#
# We compare 7 supervised models and pick the best F1 (attack class). Each model
# is wrapped in `make_pipeline(StandardScaler(), model)` for a uniform RAW input
# (the inner scaler re-scales; harmless for trees). The comparison uses
# stratified 10-fold cross-validation on a SUBSAMPLE of the train (for speed),
# then each model is tuned with a small GridSearchCV.

# %% [markdown]
# ### 8.1. Stratified train subsample (~150,000 rows)
#
# Cross-validating 1.76M rows x 7 models x 10 folds would be too slow. We draw a
# stratified subsample of about 150,000 rows from the TRAIN, on the RAW features
# `df_train[features]` (pipelines re-scale internally).

# %%
# stratified subsample of about 150,000 train rows (RAW features)
n_sub = 150000
frac_sub = n_sub / len(df_train)
df_sub = df_train.groupby("Label_binary", group_keys=False).sample(frac=frac_sub, random_state=42)
X_sub = df_sub[features]                 # RAW features (pipelines re-scale)
y_sub = df_sub["Label_binary"]
print(f"Subsample: {X_sub.shape[0]} rows (out of {len(df_train)} in train)")
print(f"Attack ratio (subsample): {y_sub.mean():.4f}")

# %% [markdown]
# ### 8.2. Dictionary of the 7 pipelines
#
# Each pipeline = `make_pipeline(StandardScaler(), model)`. We keep model
# `n_jobs` at default to avoid nested over-parallelism: parallelism happens at
# the cross-validation level (`n_jobs=-1`).

# %%
# the 7 pipelines (StandardScaler + model)
pipelines = {
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(C=0.5, random_state=42)),
    "DecisionTree": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3, random_state=42)),
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    "GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
    "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
    "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}
print(f"Number of models in the panel: {len(pipelines)}")
for name in pipelines:
    print(f"  - {name}")

# %% [markdown]
# ### 8.3. Cross-validation comparison (cross_val_score, F1)
#
# For each model: `cross_val_score` with stratified 10-fold (shuffle,
# random_state=42), `f1` scoring, `n_jobs=-1`. We print mean F1 +/- std.

# %%
# cross-validation of each model (mean F1 +/- std)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_f1_mean = {}
cv_f1_std = {}
print("=== Cross-validation (10-fold, F1 attack class) ===")
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="f1", n_jobs=-1)
    cv_f1_mean[name] = float(scores.mean())
    cv_f1_std[name] = float(scores.std())
    print(f"{name:20s} F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

# %% [markdown]
# ### 8.4. Tuning with GridSearchCV (small grid per model)
#
# Same cross-validation scheme, `f1` scoring, `n_jobs=-1`. Small grid per model.
# We keep `best_params_` and `best_score_` (tuned CV F1) of each. The grid
# targets the model inside the pipeline: keys are prefixed by the lowercase step
# name generated by `make_pipeline` (e.g. `randomforestclassifier__...`).

# %%
# small hyperparameter grid per model (keys = pipeline step __ param)
grids = {
    "LinearSVC": {"linearsvc__C": [0.5, 1, 1.5, 2, 2.5, 3]},
    "DecisionTree": {"decisiontreeclassifier__max_depth": [3, 5, 10, 15, 20, 30]},
    "LogisticRegression": {"logisticregression__C": [0.5, 1, 5, 10, 20, 30]},
    "GaussianNB": {"gaussiannb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]},
    "RandomForest": {"randomforestclassifier__max_depth": [10, 20, 30, 40, None], "randomforestclassifier__n_estimators": [100, 300]},
    "XGBoost": {"xgbclassifier__max_depth": [4, 6, 8, 10, 12], "xgbclassifier__learning_rate": [0.1, 0.3, 0.5], "xgbclassifier__n_estimators": [100, 300]},
    "KNN": {"kneighborsclassifier__n_neighbors": [1, 2, 3, 4, 5, 7]},
}

# tuning: run GridSearchCV per model, keep best_params_/best_score_
tuned_cv_f1 = {}
best_params = {}
print("=== Tuning with GridSearchCV (10-fold, F1) ===")
for name, pipe in pipelines.items():
    grid = GridSearchCV(pipe, grids[name], cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_sub, y_sub)
    tuned_cv_f1[name] = float(grid.best_score_)
    best_params[name] = grid.best_params_
    print(f"{name:20s} tuned F1 = {grid.best_score_:.4f}  best_params = {grid.best_params_}")

# %% [markdown]
# ### 8.5. Save the results table (cv_results.json)
#
# For each model: `cv_f1_mean`, `cv_f1_std`, `tuned_cv_f1`, `best_params`, plus a
# `selected` flag for the winner.

# %%
# pick the winner (highest tuned CV F1) and save cv_results.json
winner_name = max(tuned_cv_f1, key=tuned_cv_f1.get)
print(f"Winning model (highest tuned CV F1): {winner_name}  (F1 = {tuned_cv_f1[winner_name]:.4f})")

cv_results = {}
for name in pipelines:
    cv_results[name] = {
        "cv_f1_mean": round(cv_f1_mean[name], 4),
        "cv_f1_std": round(cv_f1_std[name], 4),
        "tuned_cv_f1": round(tuned_cv_f1[name], 4),
        "best_params": best_params[name],
        "selected": name == winner_name,
    }
with open(BASE + "/models/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2, ensure_ascii=False)
print("Cross-validation results saved -> models/cv_results.json")

# %% [markdown]
# ### 8.6. Retrain the winner on the FULL RAW train
#
# We rebuild the winning pipeline with its best hyperparameters, then RETRAIN it
# on the full train (`df_train[features]` RAW, ~1.76M rows). For XGBoost /
# RandomForest we set `n_jobs=-1` for the final retraining.

# %%
# rebuild the winner with its best_params + retrain on the full train
# best_params is prefixed by the pipeline step: strip the prefix to build the model
winner_params = {key.split("__", 1)[1]: value for key, value in best_params[winner_name].items()}
print(f"Winner hyperparameters ({winner_name}): {winner_params}")

# instantiate the winning model with its best hyperparameters
if winner_name == "LinearSVC":
    winner_model = LinearSVC(random_state=42, **winner_params)
elif winner_name == "DecisionTree":
    winner_model = DecisionTreeClassifier(random_state=42, **winner_params)
elif winner_name == "LogisticRegression":
    winner_model = LogisticRegression(max_iter=1000, random_state=42, **winner_params)
elif winner_name == "GaussianNB":
    winner_model = GaussianNB(**winner_params)
elif winner_name == "RandomForest":
    winner_model = RandomForestClassifier(n_jobs=-1, random_state=42, **winner_params)
elif winner_name == "XGBoost":
    winner_model = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42, **winner_params)
else:  # KNN
    winner_model = KNeighborsClassifier(**winner_params)

# winning pipeline: StandardScaler + winning model (RAW input)
best_model = make_pipeline(StandardScaler(), winner_model)
best_model.fit(df_train[features], y_train)
print(f"Winning pipeline retrained on the full train: {df_train.shape[0]} rows")

# %% [markdown]
# ### 8.7. Guarantee predict_proba (CalibratedClassifierCV if needed)
#
# `best_model.pkl` MUST expose `predict_proba` (risk score). If the winner lacks
# it (LinearSVC), we wrap the winning pipeline in
# `CalibratedClassifierCV(..., cv=3)` and retrain it.

# %%
# if the winner has no predict_proba, calibrate it (CalibratedClassifierCV cv=3)
has_proba = hasattr(best_model, "predict_proba")
if not has_proba:
    print(f"{winner_name} has no predict_proba -> calibration (CalibratedClassifierCV cv=3)")
    best_model = CalibratedClassifierCV(make_pipeline(StandardScaler(), winner_model), cv=3)
    best_model.fit(df_train[features], y_train)
    has_proba = hasattr(best_model, "predict_proba")
print(f"Winning model exposes predict_proba: {has_proba}")

# %% [markdown]
# ### 8.8. Save the winning model and its metadata
#
# We save `best_model.pkl` (winning pipeline) and `best_model_meta.json` (name,
# best_params, mean/std CV F1, has_proba, retrained_on_full).

# %%
# save best_model.pkl + best_model_meta.json
joblib.dump(best_model, BASE + "/models/best_model.pkl")
print("Winning model saved -> models/best_model.pkl")

best_model_meta = {
    "name": winner_name,
    "best_params": best_params[winner_name],
    "cv_f1_mean": round(cv_f1_mean[winner_name], 4),
    "cv_f1_std": round(cv_f1_std[winner_name], 4),
    "has_proba": True,
    "retrained_on_full": True,
}
with open(BASE + "/models/best_model_meta.json", "w") as f:
    json.dump(best_model_meta, f, indent=2, ensure_ascii=False)
print("Winner metadata saved -> models/best_model_meta.json")
print(json.dumps(best_model_meta, indent=2, ensure_ascii=False))

# %% [markdown]
# ### 8.9. Quick check of the winner on TEST (RAW features)
#
# Prediction convention: `X = df_test[features]` (RAW);
# `best_model.predict(X)` / `best_model.predict_proba(X)[:, 1]` (= risk score).
# We print accuracy, F1, and the confusion matrix on test.

# %%
# check: reload best_model.pkl then predict on the RAW test
best_model_reloaded = joblib.load(BASE + "/models/best_model.pkl")
X_test_raw = df_test[features]                      # RAW features (the pipeline re-scales)
y_pred_best = best_model_reloaded.predict(X_test_raw)
proba_best = best_model_reloaded.predict_proba(X_test_raw)[:, 1]   # risk score

acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
cm_best = confusion_matrix(y_test, y_pred_best)
print(f"Winning model ({winner_name}) - Accuracy (test): {acc_best:.4f}")
print(f"Winning model ({winner_name}) - F1 attack (test): {f1_best:.4f}")
print(f"Risk score preview (first 5): {np.round(proba_best[:5], 4)}")
print("\n=== Classification report (winning model) ===")
print(classification_report(y_test, y_pred_best, target_names=["BENIGN", "Attack"], digits=4))
print("Confusion matrix (winning model):")
print(cm_best)

# %% [markdown]
# ## 9. "Historic" Random Forest (kept for Phase 4)
#
# Phase 4 (evaluation) consumes `random_forest.pkl` via the frozen scaler. We
# keep this dedicated Random Forest (ALREADY-scaled input), distinct from the
# bake-off `best_model.pkl` (RAW input). It keeps `class_weight='balanced'` and
# `max_depth=20` to limit model size and overfitting while still separating the
# attack families well.

# %%
# historic Random Forest (scaled input) for Phase 4
rf = RandomForestClassifier(
    n_estimators=100, max_depth=20, class_weight="balanced",
    n_jobs=-1, random_state=42
)
rf.fit(X_train_s, y_train)

# predict on test (scaled features)
y_pred_rf = rf.predict(X_test_s)

# metrics
acc_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
fp_rf = int(cm_rf[0, 1])
print(f"Historic Random Forest - Accuracy: {acc_rf:.4f}")
print(f"Historic Random Forest - Recall (attack): {recall_rf:.4f}")
print(f"Historic Random Forest - False positives: {fp_rf}")
print("Confusion matrix (historic Random Forest):")
print(cm_rf)

# save the model
joblib.dump(rf, BASE + "/models/random_forest.pkl")
print("Random Forest saved -> models/random_forest.pkl")

# %% [markdown]
# ## 10. Explainability: BENIGN feature means
#
# We save the means (RAW scale) of each feature over the BENIGN train. The
# dashboard uses them to explain "why this alert" by comparing a suspicious
# row's value to the average normal behavior.

# %%
# BENIGN means (raw scale) for explainability
benign_means = X_train[y_train == 0].mean()
print("BENIGN means preview (raw scale):")
print(benign_means.head())
joblib.dump(benign_means, BASE + "/models/benign_means.pkl")
print("BENIGN means saved -> models/benign_means.pkl")

# %% [markdown]
# ## 11. Save the test set and an application sample
#
# The full test set (RAW features + labels) is saved for Phase 4 (evaluation).
# A stratified sample of about 3000 rows per attack family (`Label_group`) is
# saved for the dashboard.

# %%
# save the full test set (raw features + labels)
test_cols = features + ["Label", "Label_binary", "Label_group"]
df_test_out = df_test[test_cols]
df_test_out.to_parquet(BASE + "/data/processed/test_set.parquet", index=False)
print(f"Test set saved -> data/processed/test_set.parquet  (shape {df_test_out.shape})")

# %%
# stratified sample ~3000 rows per family (min per group), random_state=42
n_target = 3000
n_groups = df_test_out["Label_group"].nunique()
n_per_group = max(1, n_target // n_groups)
# take at most n_per_group rows per group (without exceeding the group size)
# simple loop over families then concat (keeps all columns)
parts = []
for group, group_rows in df_test_out.groupby("Label_group"):
    n_take = min(len(group_rows), n_per_group)
    parts.append(group_rows.sample(n=n_take, random_state=42))
app_sample = pd.concat(parts, ignore_index=True)
print(f"Application sample: {app_sample.shape[0]} rows")
print(app_sample["Label_group"].value_counts())
app_sample.to_parquet(BASE + "/data/processed/app_sample.parquet", index=False)
print("Sample saved -> data/processed/app_sample.parquet")

# %% [markdown]
# ## 12. Bake-off recap and winner
#
# Comparison table of the 7 models (mean CV F1 +/- std and tuned CV F1), then a
# reminder of the winner and its test performance.

# %%
# recap: comparison table of the 7 models + winner
print("=== Bake-off: comparison of the 7 models (10-fold cross-validation, F1) ===")
print(f"{'Model':20s} {'CV F1':>10s} {'+/- std':>10s} {'tuned F1':>10s} {'winner':>9s}")
for name in pipelines:
    mark = "  <===" if name == winner_name else ""
    print(f"{name:20s} {cv_f1_mean[name]:>10.4f} {cv_f1_std[name]:>10.4f} {tuned_cv_f1[name]:>10.4f} {mark:>9s}")

print(f"\nWinning model: {winner_name}")
print(f"CV F1 (subsample): {cv_f1_mean[winner_name]:.4f} +/- {cv_f1_std[winner_name]:.4f}")
print(f"Tuned CV F1 (GridSearch): {tuned_cv_f1[winner_name]:.4f}")
print(f"Accuracy on test: {acc_best:.4f}")
print(f"F1 (attack) on test: {f1_best:.4f}")

# %% [markdown]
# ## 13. Artifact contract check
#
# We check that ALL expected files (kept existing + new bake-off) exist.

# %%
# check that all contract artifacts exist
artifacts = [
    # existing artifacts kept
    BASE + "/models/features.pkl",
    BASE + "/models/scaler.pkl",
    BASE + "/models/benign_means.pkl",
    BASE + "/models/isolation_forest.pkl",
    BASE + "/models/rule_baseline.json",
    BASE + "/models/random_forest.pkl",
    BASE + "/data/processed/test_set.parquet",
    BASE + "/data/processed/app_sample.parquet",
    # new bake-off artifacts
    BASE + "/models/best_model.pkl",
    BASE + "/models/best_model_meta.json",
    BASE + "/models/cv_results.json",
]
for path in artifacts:
    exists = os.path.exists(path)
    size = os.path.getsize(path) / 1024**2 if exists else 0.0
    print(f"{'OK ' if exists else 'MISSING '} {path}  ({size:.4f} MB)")

all_present = all(os.path.exists(p) for p in artifacts)
print(f"\nAll contract artifacts present: {all_present}")

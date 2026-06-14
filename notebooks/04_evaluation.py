# %% [markdown]
# # Phase 4 - Evaluation and explainability (CICIDS2017)
#
# This notebook EVALUATES and COMPARES the three detection approaches on the full
# TEST set: the static rules baseline, the Isolation Forest, and the WINNING
# MODEL of the Phase 3 bake-off (`best_model.pkl`, a StandardScaler + model
# pipeline that consumes RAW features). It shows the bake-off ranking (cross-
# validation F1), the confusion matrices, a comparison metrics table, the false
# positive reduction versus the rules baseline, the winner's global feature
# importances, and the "why this alert" explainability for a few detected attack
# flows. All numbers are saved to reports/metrics.json.

# %% [markdown]
# ## 1. Imports and loading the Phase 3 artifacts
#
# We load the test set (RAW features + labels), the feature list, the winning
# bake-off model + its metadata, the cross-validation results table, the
# Isolation Forest and the frozen scaler (for the iso baseline), the BENIGN means
# (for explainability) and the static rules.

# %%
# imports
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib

sns.set(style="whitegrid")

# absolute project root path
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# load the full test set (RAW features + labels)
df = pd.read_parquet(BASE + "/data/processed/test_set.parquet")
print(f"Test set shape: {df.shape}")
print(df["Label_binary"].value_counts())

# load the ordered feature list and the contract artifacts
features = joblib.load(BASE + "/models/features.pkl")
best_model = joblib.load(BASE + "/models/best_model.pkl")   # winning pipeline (RAW input)
iso = joblib.load(BASE + "/models/isolation_forest.pkl")    # unsupervised baseline
scaler = joblib.load(BASE + "/models/scaler.pkl")           # frozen scaler (used by iso)
benign_means = joblib.load(BASE + "/models/benign_means.pkl")  # BENIGN means (explainability)

# winner metadata + bake-off cross-validation table
with open(BASE + "/models/best_model_meta.json") as f:
    best_model_meta = json.load(f)
with open(BASE + "/models/cv_results.json") as f:
    cv_results = json.load(f)
with open(BASE + "/models/rule_baseline.json") as f:
    rule_baseline = json.load(f)

# winning model name (used as a label throughout the notebook)
winner_name = best_model_meta["name"]
print(f"Number of features: {len(features)}")
print(f"Bake-off winning model: {winner_name}")
print(f"Winner CV F1: {best_model_meta['cv_f1_mean']:.4f} +/- {best_model_meta['cv_f1_std']:.4f}")
print(f"Rules logic: {rule_baseline['logique']}")

# %% [markdown]
# ## 2. Predictions of the three approaches on the TEST set
#
# Contract convention: the WINNER consumes RAW features
# (`best_model.predict(df[features])`, its internal scaler rescales them). The
# Isolation Forest predicts on features scaled by the frozen scaler
# (`scaler.transform`). The static rules apply on RAW features. The ground truth
# is `Label_binary` (0 = BENIGN, 1 = attack).

# %%
# feature matrices and ground truth
X = df[features]                 # RAW features (winner + rules + explainability)
Xs = scaler.transform(X)         # scaled features (for the Isolation Forest)
y_true = df["Label_binary"]      # ground truth: 0 = BENIGN, 1 = attack

# winner: binary prediction + risk score (attack proba) on RAW features
y_pred_best = best_model.predict(X)
proba_best = best_model.predict_proba(X)[:, 1]   # risk score = proba of the attack class

# Isolation Forest: -1 = anomaly -> 1 (attack), 1 = normal -> 0 (BENIGN)
iso_pred_raw = iso.predict(Xs)
y_pred_iso = np.where(iso_pred_raw == -1, 1, 0)  # -1 anomaly -> attack (1), 1 normal -> BENIGN (0)

# static rules baseline on RAW features (logical OR of the 5 rules)
thr_avg_packet_size = 496.0
thr_bwd_packets_s = 58823.0
thr_flow_packets_s = 500000.0
thr_fwd_packets_short = 3
thr_flow_duration_long = 1000000.0
thr_flow_bytes_s = 12000000.0

rule_avg_size = X["Average Packet Size"] > thr_avg_packet_size
rule_bwd_rate = X["Bwd Packets/s"] > thr_bwd_packets_s
rule_flow_rate = X["Flow Packets/s"] > thr_flow_packets_s
rule_short_long = (X["Total Fwd Packets"] <= thr_fwd_packets_short) & (X["Flow Duration"] > thr_flow_duration_long)
rule_bytes_rate = X["Flow Bytes/s"] > thr_flow_bytes_s

# logical OR of the 5 rules -> baseline prediction
y_pred_rule = (rule_avg_size | rule_bwd_rate | rule_flow_rate | rule_short_long | rule_bytes_rate).astype(int)

print("Predictions computed for the 3 approaches.")
print(f"Positives (attack) - Rules            : {int(y_pred_rule.sum())}")
print(f"Positives (attack) - Isolation Forest : {int(y_pred_iso.sum())}")
print(f"Positives (attack) - {winner_name:18s} : {int(y_pred_best.sum())}")

# %% [markdown]
# ## 3. Bake-off: ranking of the 7 models by cross-validation
#
# We plot the mean F1 (10-fold cross-validation on the train subsample) +/- std
# of the 7 models compared in Phase 3, read from `cv_results.json`. The winning
# model is highlighted in color. This is the justification of the main model
# choice.

# %%
# bar chart of the bake-off: mean CV F1 +/- std of the 7 models (winner highlighted)
# read the cross-validation table + sort by ascending mean F1 (readable)
model_names = []
f1_means = []
f1_stds = []
selected_flags = []
for name, res in cv_results.items():
    model_names.append(name)
    f1_means.append(res["cv_f1_mean"])
    f1_stds.append(res["cv_f1_std"])
    selected_flags.append(res["selected"])

df_bakeoff = pd.DataFrame({
    "Model": model_names,
    "F1_mean": f1_means,
    "F1_std": f1_stds,
    "selected": selected_flags,
}).sort_values("F1_mean").reset_index(drop=True)

print("=== Bake-off: F1 by cross-validation (10-fold) ===")
print(df_bakeoff.round(4).to_string(index=False))

# colors: winner in green, the others in gray-blue
bakeoff_colors = ["seagreen" if sel else "steelblue" for sel in df_bakeoff["selected"]]

plt.figure(figsize=(10, 6))
plt.barh(df_bakeoff["Model"], df_bakeoff["F1_mean"],
         xerr=df_bakeoff["F1_std"], color=bakeoff_colors,
         capsize=4, error_kw={"ecolor": "black"})
plt.title("Bake-off of the 7 models: mean F1 by cross-validation (10-fold)\nThe winner (" + winner_name + ") is highlighted")
plt.xlabel("Mean F1 (attack class) +/- std")
plt.ylabel("Model")
plt.xlim(0, 1.05)
# annotate the F1 value at the end of each bar
for i, (val, std) in enumerate(zip(df_bakeoff["F1_mean"], df_bakeoff["F1_std"])):
    plt.text(val + std + 0.01, i, f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(BASE + "/reports/figures/11_bakeoff_cv_f1.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Confusion matrices of the three approaches
#
# For each approach we show the confusion matrix (rows = truth, columns =
# prediction). The false positives (BENIGN flagged as attack) are the top-right
# cell: they represent the useless alerts that SOC analysts have to handle.

# %%
# confusion matrix - rules baseline
cm_rule = confusion_matrix(y_true, y_pred_rule)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rule, annot=True, fmt="d", cmap="Reds",
            xticklabels=["BENIGN", "Attack"], yticklabels=["BENIGN", "Attack"])
plt.title("Confusion matrix - Rules baseline")
plt.xlabel("Prediction")
plt.ylabel("Ground truth")
plt.tight_layout()
plt.savefig(BASE + "/reports/figures/06_cm_regles.png", dpi=120)
plt.show()

# %%
# confusion matrix - Isolation Forest
cm_iso = confusion_matrix(y_true, y_pred_iso)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_iso, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["BENIGN", "Attack"], yticklabels=["BENIGN", "Attack"])
plt.title("Confusion matrix - Isolation Forest")
plt.xlabel("Prediction")
plt.ylabel("Ground truth")
plt.tight_layout()
plt.savefig(BASE + "/reports/figures/07_cm_isolation_forest.png", dpi=120)
plt.show()

# %%
# confusion matrix - bake-off winning model
cm_best = confusion_matrix(y_true, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens",
            xticklabels=["BENIGN", "Attack"], yticklabels=["BENIGN", "Attack"])
plt.title("Confusion matrix - " + winner_name + " (winning model)")
plt.xlabel("Prediction")
plt.ylabel("Ground truth")
plt.tight_layout()
plt.savefig(BASE + "/reports/figures/08_cm_best_model.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Comparison metrics table
#
# We compute for the three approaches: precision, recall, f1 (on the attack
# class), global accuracy, number of false positives and false positive rate
# (FP / (FP + TN), i.e. the proportion of BENIGN wrongly flagged). The false
# positive rate is the central metric of the project.

# %%
# false positives (FP) and true negatives (TN) read from each confusion matrix
# cell [0, 1] = BENIGN (truth 0) predicted attack (1) = false positive
# cell [0, 0] = BENIGN (truth 0) predicted BENIGN (0) = true negative
fp_rule = int(cm_rule[0, 1])
tn_rule = int(cm_rule[0, 0])
fp_iso = int(cm_iso[0, 1])
tn_iso = int(cm_iso[0, 0])
fp_best = int(cm_best[0, 1])
tn_best = int(cm_best[0, 0])

# false positive rate = FP / (FP + TN)
fp_rate_rule = fp_rule / (fp_rule + tn_rule)
fp_rate_iso = fp_iso / (fp_iso + tn_iso)
fp_rate_best = fp_best / (fp_best + tn_best)

# build the comparison table (DataFrame)
models = ["Rules", "Isolation Forest", winner_name]
predictions = {"Rules": y_pred_rule, "Isolation Forest": y_pred_iso, winner_name: y_pred_best}
fp_per_model = {"Rules": fp_rule, "Isolation Forest": fp_iso, winner_name: fp_best}
fp_rate_per_model = {"Rules": fp_rate_rule, "Isolation Forest": fp_rate_iso, winner_name: fp_rate_best}

# dict of metrics per model + for loop (allowed idiom)
comparison = {"Model": [], "Precision": [], "Recall": [], "F1": [],
              "Accuracy": [], "False positives": [], "FP rate": []}
for name in models:
    y_pred = predictions[name]
    comparison["Model"].append(name)
    comparison["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
    comparison["Recall"].append(recall_score(y_true, y_pred, zero_division=0))
    comparison["F1"].append(f1_score(y_true, y_pred, zero_division=0))
    comparison["Accuracy"].append(accuracy_score(y_true, y_pred))
    comparison["False positives"].append(fp_per_model[name])
    comparison["FP rate"].append(fp_rate_per_model[name])

df_comparison = pd.DataFrame(comparison)
print("=== Comparison metrics table (test set) ===")
print(df_comparison.round(4).to_string(index=False))

# %% [markdown]
# ## 6. False positive reduction versus the rules baseline
#
# The business goal targets a strong reduction of false positives compared to the
# rules baseline. We compute the actual reduction achieved by the winning model
# and by the Isolation Forest, then honestly place it against the 60-80 %
# reference range (we report the real numbers even if they fall outside).

# %%
# false positive reduction versus the rules baseline
fp_reduction_best = (fp_rule - fp_best) / fp_rule * 100
fp_reduction_iso = (fp_rule - fp_iso) / fp_rule * 100

print(f"False positives - Rules baseline   : {fp_rule}")
print(f"False positives - Isolation Forest : {fp_iso}")
print(f"False positives - {winner_name:18s} : {fp_best}")
print(f"\nFP reduction ({winner_name} vs Rules)         : {fp_reduction_best:.4f} %")
print(f"FP reduction (Isolation Forest vs Rules) : {fp_reduction_iso:.4f} %")

# honest placement against the 60-80 % reference range
target_min = 60.0
target_max = 80.0
in_target_best = (fp_reduction_best >= target_min) and (fp_reduction_best <= target_max)
in_target_iso = (fp_reduction_iso >= target_min) and (fp_reduction_iso <= target_max)
print(f"\nReference: FP reduction between {target_min:.0f} % and {target_max:.0f} %")
print(f"{winner_name} within the 60-80 % range    : {in_target_best}")
print(f"Isolation Forest within the 60-80 % range : {in_target_iso}")
if fp_reduction_best > target_max:
    print(f"-> The winning model ({winner_name}) EXCEEDS the range (even stronger reduction): favorable result.")
elif fp_reduction_best < target_min:
    print(f"-> The winning model ({winner_name}) is BELOW the range: to be documented honestly in the report.")

# %% [markdown]
# ## 7. Global feature importance (winning model)
#
# We explain the winning model globally. We get the final model in the pipeline
# via `named_steps`. Depending on the model type:
# - if it exposes `feature_importances_` (trees: RF / XGBoost / DecisionTree) we
#   use the native importance (gain / impurity reduction);
# - if it exposes `coef_` (LogReg / LinearSVC) we use the absolute value of the
#   coefficients;
# - otherwise (GaussianNB / KNN) we compute a `permutation_importance` on a
#   sample (~5000 rows) of the test set.
# We show the 20 most important features.

# %%
# get the final model in the winning pipeline (last step of named_steps)
if hasattr(best_model, "named_steps"):
    final_step_name = list(best_model.named_steps.keys())[-1]
    final_model = best_model.named_steps[final_step_name]
else:
    final_model = best_model
print(f"Final model in the pipeline: {type(final_model).__name__}")

# compute the importance depending on the model type
if hasattr(final_model, "feature_importances_"):
    importance_method = "feature_importances_ (gain / impurity)"
    importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
elif hasattr(final_model, "coef_"):
    importance_method = "absolute value of coefficients (|coef_|)"
    coefs = np.ravel(final_model.coef_)
    importances = pd.Series(np.abs(coefs), index=features).sort_values(ascending=False)
else:
    importance_method = "permutation_importance (~5000 rows sample)"
    # stratified sample of about 5000 test rows to limit the cost
    n_perm = 5000
    df_perm = df.groupby("Label_binary", group_keys=False).sample(
        n=min(n_perm // 2, int(y_true.value_counts().min())), random_state=42
    )
    X_perm = df_perm[features]
    y_perm = df_perm["Label_binary"]
    perm = permutation_importance(best_model, X_perm, y_perm, scoring="f1",
                                  n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

print(f"Global explainability method used: {importance_method}")
top20 = importances.head(20)
print("=== Top 20 features by importance (winning model) ===")
print(top20.round(4).to_string())

# bar chart of the 20 most important features
plt.figure(figsize=(10, 8))
sns.barplot(x=top20.values, y=top20.index, color="seagreen")
plt.title("Global importance of the top 20 features - " + winner_name + "\n(" + importance_method + ")")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(BASE + "/reports/figures/09_importance_features.png", dpi=120)
plt.show()

# %% [markdown]
# ## 8. Per-alert explainability: "why this alert"
#
# To make alerts actionable for an analyst, we explain a few attack flows
# CORRECTLY detected by the winning model. For each flow, we take the most
# important features of the model and compare the flow's value to the mean of the
# BENIGN traffic (benign_means). The red bars highlight the features whose value
# strongly deviates from normal: these are the reasons for the alert.

# %%
# select attack flows well detected by the winner (one per attack family)
# conditions: true attack (Label_binary == 1), correctly predicted attack by the winner,
# with a high risk score. We pick one per family to vary the cases.
df_eval = df.copy()
df_eval["pred_best"] = y_pred_best
df_eval["proba_best"] = proba_best

# targeted attack families (present in the test set, enough volume)
target_families = ["DoS", "DDoS", "PortScan"]
alert_indices = []
for family in target_families:
    mask = (df_eval["Label_group"] == family) & (df_eval["Label_binary"] == 1) & (df_eval["pred_best"] == 1)
    candidates = df_eval[mask]
    if len(candidates) > 0:
        # the most "confident" flow (highest risk score) of the family
        idx = candidates["proba_best"].idxmax()
        alert_indices.append(idx)

print(f"Number of attack flows selected for explainability: {len(alert_indices)}")
for idx in alert_indices:
    row = df_eval.loc[idx]
    print(f"  - index {idx} | family {row['Label_group']:10s} | label {row['Label']:20s} | risk score {row['proba_best']:.4f}")

# %%
# top features by importance kept for the explanation
n_top_explain = 8
top_features_explain = importances.head(n_top_explain).index.tolist()
print(f"Features used for the explanation (top {n_top_explain} by importance):")
print(top_features_explain)

# subplots: one chart per flow, flow value vs BENIGN mean comparison
n_alerts = len(alert_indices)
fig, axs = plt.subplots(1, n_alerts, figsize=(7 * n_alerts, 6))
# if a single alert, axs is not an array: wrap it for the loop
if n_alerts == 1:
    axs = [axs]

for i, idx in enumerate(alert_indices):
    row = df_eval.loc[idx]
    # flow values and BENIGN means on the top features
    flow_values = X.loc[idx, top_features_explain].astype(float)
    benign_values = benign_means[top_features_explain].astype(float)

    # readable printed table (flow value vs mean normal + ratio)
    table = pd.DataFrame({
        "Flow (alert)": flow_values,
        "BENIGN mean": benign_values,
    })
    table["Flow / BENIGN ratio"] = table["Flow (alert)"] / table["BENIGN mean"].replace(0, np.nan)
    print(f"\n=== Why this alert? Flow index {idx} - {row['Label']} (family {row['Label_group']}) ===")
    print(f"{winner_name} risk score: {row['proba_best']:.4f}")
    print(table.round(4).to_string())

    # highlight the abnormal features: relative deviation > 2x the BENIGN mean
    ratio = (flow_values / benign_values.replace(0, np.nan)).abs()
    colors = ["crimson" if (pd.notna(r) and r > 2.0) else "steelblue" for r in ratio]

    # grouped bars: flow vs BENIGN per feature
    y_pos = np.arange(len(top_features_explain))
    height = 0.4
    axs[i].barh(y_pos + height / 2, flow_values.values, height=height,
                color=colors, label="Flow (alert)")
    axs[i].barh(y_pos - height / 2, benign_values.values, height=height,
                color="lightgray", label="BENIGN mean")
    axs[i].set_yticks(y_pos)
    axs[i].set_yticklabels(top_features_explain)
    axs[i].invert_yaxis()
    axs[i].set_xlabel("Value (raw scale)")
    axs[i].set_title(f"Alert {row['Label_group']} - {row['Label']}\nrisk score {row['proba_best']:.3f}")
    axs[i].legend(loc="lower right")

plt.suptitle("Why this alert? Suspicious flow value vs mean BENIGN traffic (red = abnormal)")
plt.tight_layout()
plt.savefig(BASE + "/reports/figures/10_explication_alerte.png", dpi=120)
plt.show()

# %% [markdown]
# ## 9. Saving all the numeric metrics
#
# We serialize to reports/metrics.json: the bake-off ranking (the 7 CV F1), the
# name of the selected model, the test metrics of the three approaches
# (precision, recall, f1, accuracy, false positives, false positive rate) and the
# false positive reductions (winner + iso) versus the rules baseline.

# %%
# test metrics per approach (loop over the predictions dict)
metrics = {}
metrics_test = {}
for name in models:
    y_pred = predictions[name]
    block = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "fp": fp_per_model[name],
        "fp_rate": round(float(fp_rate_per_model[name]), 4),
    }
    metrics[name] = block
    metrics_test[name] = block

# bake-off block: mean/std CV F1 + selected for the 7 models
metrics["bakeoff"] = {}
for name, res in cv_results.items():
    metrics["bakeoff"][name] = {
        "cv_f1_mean": res["cv_f1_mean"],
        "cv_f1_std": res["cv_f1_std"],
        "selected": res["selected"],
    }

# selected model, FP reductions and reference range
metrics["selected_model"] = winner_name
metrics["fp_reduction_best"] = round(float(fp_reduction_best), 4)
metrics["fp_reduction_iso"] = round(float(fp_reduction_iso), 4)
metrics["target_60_80"] = {
    "cible_min": target_min,
    "cible_max": target_max,
    "best_dans_cible": bool(in_target_best),
    "iso_dans_cible": bool(in_target_iso),
}
# grouped test block (handy for the Phase 5 dashboard)
metrics["test"] = metrics_test

with open(BASE + "/reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("Metrics saved -> reports/metrics.json")
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 10. Final summary
#
# Plain recap of the bake-off, the comparison of the three approaches on the test
# set and the false positive reduction.

# %%
# final printed summary
print("=== Final summary - Phase 4 ===")
print(f"Bake-off winning model: {winner_name} "
      f"(CV F1 {best_model_meta['cv_f1_mean']:.4f} +/- {best_model_meta['cv_f1_std']:.4f})")
print("\nComparison on the test set:")
print(df_comparison.round(4).to_string(index=False))
print(f"\nFalse positive reduction ({winner_name} vs Rules)         : {fp_reduction_best:.4f} %")
print(f"False positive reduction (Isolation Forest vs Rules) : {fp_reduction_iso:.4f} %")
print(f"Global explainability method: {importance_method}")
print("\nGenerated figures: 06_cm_regles, 07_cm_isolation_forest, 08_cm_best_model, "
      "09_importance_features, 10_explication_alerte, 11_bakeoff_cv_f1 (reports/figures/).")
print("Numeric metrics: reports/metrics.json")

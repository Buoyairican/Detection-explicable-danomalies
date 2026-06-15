# SOC dashboard — Explainable anomaly detection (CICIDS2017)
# Phase 5: Streamlit app for the analyst.
# Notebook style: flat code, short English comments.
# Allowed exception: a few minimal functions for cached loading.

# imports
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# absolute artifact paths (Phase 3 contract)
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"
SCALER_PATH = BASE + "/models/scaler.pkl"
BEST_MODEL_PATH = BASE + "/models/best_model.pkl"
BEST_META_PATH = BASE + "/models/best_model_meta.json"
FEATURES_PATH = BASE + "/models/features.pkl"
BENIGN_MEANS_PATH = BASE + "/models/benign_means.pkl"
RULES_PATH = BASE + "/models/rule_baseline.json"
SAMPLE_PATH = BASE + "/data/processed/app_sample.parquet"
FEEDBACK_PATH = BASE + "/app/feedback.csv"


# 1. load the models and artifacts (cached)
@st.cache_resource
def load_models():
    # scaler (to scale before prediction) + best model + metadata
    scaler = joblib.load(SCALER_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)
    with open(BEST_META_PATH, "r") as f:
        best_meta = json.load(f)
    features = joblib.load(FEATURES_PATH)
    benign_means = joblib.load(BENIGN_MEANS_PATH)
    with open(RULES_PATH, "r") as f:
        rules = json.load(f)
    return scaler, best_model, best_meta, features, benign_means, rules


@st.cache_data
def load_sample():
    # stratified test sample (raw features + labels)
    df = pd.read_parquet(SAMPLE_PATH)
    return df


# small helper: static rule verdict on a raw row
def rule_verdict(row, rules):
    # OR logic: a single fired rule is enough to flag an attack
    detail = []
    fired = False
    # go through the rules described in rule_baseline.json (French keys kept)
    for r in rules["regles"]:
        if "feature" in r:
            # simple rule: feature OP threshold
            value = row[r["feature"]]
            if r["operateur"] == ">":
                ok = value > r["seuil"]
            else:
                ok = value < r["seuil"]
            label = f"{r['feature']} {r['operateur']} {r['seuil']:.0f}"
        else:
            # compound rule: feature_1 OP threshold_1 AND feature_2 OP threshold_2
            v1 = row[r["feature_1"]]
            v2 = row[r["feature_2"]]
            if r["operateur_1"] == ">":
                ok1 = v1 > r["seuil_1"]
            else:
                ok1 = v1 <= r["seuil_1"]
            if r["operateur_2"] == ">":
                ok2 = v2 > r["seuil_2"]
            else:
                ok2 = v2 <= r["seuil_2"]
            ok = ok1 and ok2
            label = (
                f"{r['feature_1']} {r['operateur_1']} {r['seuil_1']:.0f} AND "
                f"{r['feature_2']} {r['operateur_2']} {r['seuil_2']:.0f}"
            )
        if ok:
            fired = True
        detail.append({"Rule": label, "Fired": "Yes" if ok else "No"})
    return fired, detail


# actual loading
scaler, best_model, best_meta, features, benign_means, rules = load_models()
df = load_sample()

# name of the best model (from the Phase 3 bake-off)
model_name = best_meta["name"]

# page configuration
st.set_page_config(page_title="SOC — CICIDS2017 anomaly detection", layout="wide")

# 2. title and short introduction
st.title("Explainable network anomaly detection — SOC dashboard")
st.markdown(
    f"This tool helps the analyst **triage network flows** from the CICIDS2017 dataset. "
    f"For each flow we compare two approaches (static rules and **{model_name}**, kept in "
    f"Phase 3), show a **risk score** and explain **why** an alert fires. The true labels are "
    f"shown for reference only."
)
# model kept by the bake-off (cross-validation F1)
st.success(
    f"Model kept: **{model_name}** "
    f"(CV F1 = {best_meta['cv_f1_mean']:.4f} ± {best_meta['cv_f1_std']:.4f})"
)

# 3. sidebar: pick the flow to inspect
st.sidebar.header("Flow selection")

# optional filter by family (Label_group)
families = ["All"] + sorted(df["Label_group"].unique().tolist())
family = st.sidebar.selectbox("Filter by family (Label_group)", families)

# subset of available indices given the filter
if family == "All":
    available_idx = df.index.tolist()
else:
    available_idx = df[df["Label_group"] == family].index.tolist()

# keep the current index in session state
if "flow_idx" not in st.session_state:
    st.session_state.flow_idx = available_idx[0]

# random flow button: draw an index from the filtered subset
if st.sidebar.button("Random flow"):
    st.session_state.flow_idx = int(np.random.choice(available_idx))

# if the current index is no longer in the filter, reset it to the first available
if st.session_state.flow_idx not in available_idx:
    st.session_state.flow_idx = available_idx[0]

# selector for the flow index in the filtered subset
flow_idx = st.sidebar.selectbox(
    "Flow index to inspect",
    available_idx,
    index=available_idx.index(st.session_state.flow_idx),
)
st.session_state.flow_idx = flow_idx

st.sidebar.caption(f"{len(available_idx)} flows available in the selection.")

# 4. computations for the selected flow
# selected raw row
row = df.loc[flow_idx]

# raw flow features, then scaling (the model expects scaled features)
X = df.loc[[flow_idx], features]
Xs = scaler.transform(X)

# best model verdict + risk score (attack probability in %)
best_pred = int(best_model.predict(Xs)[0])  # 0 = BENIGN, 1 = attack
risk_score = float(best_model.predict_proba(Xs)[:, 1][0]) * 100.0

# static rule verdict (on raw features)
rule_flag, rule_detail = rule_verdict(row, rules)

# true family of the flow (for reference)
true_family = row["Label_group"]
true_label = row["Label"]

# 5. VERDICT panel
st.header(f"Main model verdict ({model_name})")
col_a, col_b, col_c = st.columns([2, 2, 2])

with col_a:
    # clear green/red badge based on the model verdict
    if best_pred == 1:
        st.error("ANOMALY — suspicious flow")
    else:
        st.success("NORMAL — benign flow")

with col_b:
    # risk score shown as a metric + progress bar
    st.metric("Risk score (attack proba)", f"{risk_score:.2f} %")
    st.progress(min(int(round(risk_score)), 100))

with col_c:
    # true family for reference (not used for the decision)
    st.metric("True family (reference)", true_family)
    st.caption(f"Detailed label: {true_label}")

# 6. COMPARISON panel (two approaches)
st.header("Comparison of the two approaches")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Static rules")
    if rule_flag:
        st.error("ANOMALY")
    else:
        st.success("NORMAL")
    st.caption("OR logic on raw thresholds (simplistic baseline).")

with col2:
    st.subheader(model_name)
    if best_pred == 1:
        st.error("ANOMALY")
    else:
        st.success("NORMAL")
    st.caption(f"Risk score: {risk_score:.2f} %.")

# detail of the fired rules
with st.expander("Static rules detail"):
    df_rules = pd.DataFrame(rule_detail)
    st.dataframe(df_rules, hide_index=True, use_container_width=True)

# 7. WHY THIS ALERT panel
st.header("Why this alert? (explainability)")
st.markdown(
    f"We show the most important features for **{model_name}**, comparing the "
    f"**flow value** to the **mean of the BENIGN** training traffic. "
    f"Features whose value deviates strongly from the benign mean are highlighted."
)

# number of features to explain (slider)
top_n = st.slider("Number of features to explain", 5, 20, 10)

# importances of the best model (directly on the model)
final_model = best_model
if hasattr(final_model, "feature_importances_"):
    # trees / gradient boosting: native importances
    weights = np.asarray(final_model.feature_importances_)
    importance_label = "Importance"
elif hasattr(final_model, "coef_"):
    # linear model: absolute value of the coefficients
    weights = np.abs(np.ravel(final_model.coef_))
    importance_label = "Importance (|coef|)"
else:
    # fallback: uniform importances (no usable weight info)
    weights = np.ones(len(features))
    importance_label = "Importance (uniform)"

# sorted importances, keep the top N
importances = pd.Series(weights, index=features)
top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

# build the comparison table: flow value vs BENIGN mean
rows = []
for f in top_features:
    val = float(row[f])
    mean_val = float(benign_means[f])
    # relative deviation (in multiples of the benign mean), avoid division by zero
    if mean_val != 0:
        deviation = (val - mean_val) / abs(mean_val)
    else:
        deviation = 0.0 if val == 0 else np.inf
    # flag as abnormal a deviation of more than 100% from BENIGN
    abnormal = abs(deviation) > 1.0
    rows.append(
        {
            "Feature": f,
            importance_label: float(importances[f]),
            "Flow value": val,
            "BENIGN mean": mean_val,
            "Relative deviation": deviation,
            "Abnormal": "Yes" if abnormal else "No",
        }
    )

df_explain = pd.DataFrame(rows)


# highlight the abnormal rows (light red background)
def highlight_abnormal(line):
    if line["Abnormal"] == "Yes":
        return ["background-color: #ffcccc"] * len(line)
    return [""] * len(line)


# styled table with numeric formatting
style_explain = (
    df_explain.style.apply(highlight_abnormal, axis=1).format(
        {
            importance_label: "{:.4f}",
            "Flow value": "{:.2f}",
            "BENIGN mean": "{:.2f}",
            "Relative deviation": "{:+.2f}",
        }
    )
)
st.dataframe(style_explain, hide_index=True, use_container_width=True)

# bar chart: flow value vs BENIGN mean
st.subheader("Chart: flow vs BENIGN mean")
df_bar = df_explain.set_index("Feature")[["Flow value", "BENIGN mean"]]
st.bar_chart(df_bar)

# 8. ANALYST FEEDBACK panel
st.header("Analyst feedback")
st.markdown(
    f"Confirm the **{model_name}** verdict for this flow. The feedback is saved to "
    f"`app/feedback.csv` and can be used to improve the model."
)

col_tp, col_fp = st.columns(2)
click_tp = col_tp.button("True positive (confirmed alert)")
click_fp = col_fp.button("False positive (false alarm)")

# save the feedback if one of the two buttons is clicked
if click_tp or click_fp:
    feedback_label = "True positive" if click_tp else "False positive"
    # row to append to the feedback log
    new_row = pd.DataFrame(
        [
            {
                "flow_index": int(flow_idx),
                "model_verdict": "ANOMALY" if best_pred == 1 else "NORMAL",
                "risk_score": round(risk_score, 4),
                "analyst_feedback": feedback_label,
            }
        ]
    )
    # write the header if the file does not exist yet, otherwise append
    if os.path.exists(FEEDBACK_PATH):
        new_row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)
    st.success(
        f"Feedback saved: flow {flow_idx} -> {feedback_label} "
        f"(verdict {model_name}: {'ANOMALY' if best_pred == 1 else 'NORMAL'}, "
        f"score {risk_score:.2f} %)."
    )

# show the existing feedback log
if os.path.exists(FEEDBACK_PATH):
    with st.expander("View the saved feedback log"):
        df_feedback = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(df_feedback, hide_index=True, use_container_width=True)

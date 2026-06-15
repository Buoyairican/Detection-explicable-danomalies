# SOC dashboard — Explainable anomaly detection (CICIDS2017)
# Phase 5: Streamlit app for the analyst (dark mode).
# Notebook style: flat code, short English comments.
# Allowed exception: a few minimal functions for cached loading.

# imports
import os
import json
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# absolute artifact paths (Phase 3 contract)
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"
SCALER_PATH = BASE + "/models/scaler.pkl"
BEST_MODEL_PATH = BASE + "/models/best_model.pkl"
BEST_META_PATH = BASE + "/models/best_model_meta.json"
FEATURES_PATH = BASE + "/models/features.pkl"
BENIGN_MEANS_PATH = BASE + "/models/benign_means.pkl"
SAMPLE_PATH = BASE + "/data/processed/app_sample.parquet"
FEEDBACK_PATH = BASE + "/app/feedback.csv"


# load the models and artifacts (cached)
@st.cache_resource
def load_models():
    scaler = joblib.load(SCALER_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)
    with open(BEST_META_PATH, "r") as f:
        best_meta = json.load(f)
    features = joblib.load(FEATURES_PATH)
    benign_means = joblib.load(BENIGN_MEANS_PATH)
    return scaler, best_model, best_meta, features, benign_means


@st.cache_data
def load_sample():
    df = pd.read_parquet(SAMPLE_PATH)
    return df


# page configuration
st.set_page_config(page_title="SOC — CICIDS2017 anomaly detection",
                   page_icon="🛡️", layout="wide")

# design system (Fira fonts + data-dense dashboard palette, DARK mode, WCAG AA)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --primary:#3B82F6; --secondary:#60A5FA; --accent:#FBBF24;
  --bg:#0F172A; --surface:#1E293B; --inset:#0F172A; --muted:#334155; --border:#334155;
  --danger:#F87171; --danger-soft:rgba(248,113,113,.14); --danger-bd:rgba(248,113,113,.45);
  --success:#4ADE80; --success-soft:rgba(74,222,128,.14); --success-bd:rgba(74,222,128,.45);
  --text:#E2E8F0; --text-muted:#94A3B8;
}
.stApp{ background:var(--bg); color:var(--text); font-family:'Fira Sans',sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1280px; }
h1,h2,h3{ font-family:'Fira Sans',sans-serif; letter-spacing:-0.01em; color:var(--text); }

.app-header{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }
.app-title{ font-size:1.7rem; font-weight:700; margin:0; color:var(--text); }
.app-sub{ color:var(--text-muted); font-size:0.92rem; margin-top:2px; }
.app-sub b{ font-family:'Fira Code',monospace; color:var(--secondary); }
.model-chip{ background:rgba(59,130,246,.14); color:var(--secondary); border:1px solid rgba(59,130,246,.4);
  padding:10px 16px; border-radius:12px; font-size:0.82rem; font-weight:700; white-space:nowrap;
  text-align:right; line-height:1.35; }
.model-chip small{ display:block; opacity:0.95; font-weight:500; font-family:'Fira Code',monospace; color:var(--text-muted); }

.card{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:22px;
  box-shadow:0 2px 4px rgba(0,0,0,.35), 0 12px 34px rgba(0,0,0,.45); }

.hero{ display:grid; grid-template-columns:220px 1fr; gap:26px; align-items:center; }
@media (max-width:820px){ .hero{ grid-template-columns:1fr; } }

.gauge{ width:200px; height:200px; border-radius:50%; display:grid; place-items:center; margin:0 auto;
  transition:background .35s ease; }
.gauge-inner{ width:150px; height:150px; border-radius:50%; background:var(--surface); display:grid;
  place-items:center; box-shadow:inset 0 2px 7px rgba(0,0,0,.55); }
.gauge-value{ font-family:'Fira Code',monospace; font-size:2rem; font-weight:700; font-variant-numeric:tabular-nums; }
.gauge-label{ font-size:0.74rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em; }

.verdict{ display:inline-flex; align-items:center; gap:10px; padding:12px 18px; border-radius:12px;
  font-weight:700; font-size:1.1rem; }
.verdict .dot{ width:12px; height:12px; border-radius:50%; }
.verdict.anomaly{ background:var(--danger-soft); color:var(--danger); border:1px solid var(--danger-bd); }
.verdict.anomaly .dot{ background:var(--danger); box-shadow:0 0 0 4px rgba(248,113,113,.20); }
.verdict.benign{ background:var(--success-soft); color:var(--success); border:1px solid var(--success-bd); }
.verdict.benign .dot{ background:var(--success); box-shadow:0 0 0 4px rgba(74,222,128,.20); }

.kpi-row{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:18px; }
@media (max-width:560px){ .kpi-row{ grid-template-columns:1fr; } }
.kpi{ background:var(--inset); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }
.kpi-label{ display:block; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.07em;
  color:var(--text-muted); margin-bottom:5px; }
.kpi-value{ font-family:'Fira Code',monospace; font-size:1.02rem; font-weight:600; color:var(--text);
  font-variant-numeric:tabular-nums; word-break:break-word; }

.section-title{ font-size:1.18rem; font-weight:700; margin:30px 0 6px; display:flex; align-items:center;
  gap:10px; color:var(--text); }
.section-title .bar{ width:4px; height:20px; background:var(--accent); border-radius:2px; }
.section-sub{ color:var(--text-muted); font-size:0.9rem; margin:0 0 12px; }
.section-sub b{ color:var(--secondary); }
.section-sub code{ background:var(--muted); color:var(--text); padding:1px 6px; border-radius:5px; }

.stButton>button{ border-radius:10px; font-weight:600; border:1px solid var(--border);
  background:var(--surface); color:var(--text); transition:all .15s ease; }
.stButton>button:hover{ border-color:var(--secondary); color:var(--secondary); transform:translateY(-1px);
  box-shadow:0 6px 16px rgba(0,0,0,.5); }
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# load everything
scaler, best_model, best_meta, features, benign_means = load_models()
df = load_sample()
model_name = best_meta["name"]

# sidebar: pick the flow to inspect
st.sidebar.header("Flow selection")
families = ["All"] + sorted(df["Label_group"].unique().tolist())
family = st.sidebar.selectbox("Filter by family (Label_group)", families)
if family == "All":
    available_idx = df.index.tolist()
else:
    available_idx = df[df["Label_group"] == family].index.tolist()

if "flow_idx" not in st.session_state:
    st.session_state.flow_idx = available_idx[0]
if st.sidebar.button("Random flow", use_container_width=True):
    st.session_state.flow_idx = int(np.random.choice(available_idx))
if st.session_state.flow_idx not in available_idx:
    st.session_state.flow_idx = available_idx[0]

flow_idx = st.sidebar.selectbox(
    "Flow index to inspect", available_idx,
    index=available_idx.index(st.session_state.flow_idx))
st.session_state.flow_idx = flow_idx
st.sidebar.caption(f"{len(available_idx)} flows available · {len(df)} in sample")

# compute the verdict for the selected flow
row = df.loc[flow_idx]
X = df.loc[[flow_idx], features]
Xs = scaler.transform(X)                      # the model expects scaled features
best_pred = int(best_model.predict(Xs)[0])    # 0 = BENIGN, 1 = attack
risk_score = float(best_model.predict_proba(Xs)[:, 1][0]) * 100.0
true_family = row["Label_group"]
true_label = row["Label"]

# header
st.markdown(
    f"""
<div class="app-header">
  <div>
    <h1 class="app-title">Network Anomaly Detection</h1>
    <div class="app-sub">Explainable SOC triage · CICIDS2017 · inspecting flow <b>#{flow_idx}</b></div>
  </div>
  <div class="model-chip">{model_name}
    <small>CV F1 {best_meta['cv_f1_mean']:.4f} ± {best_meta['cv_f1_std']:.4f}</small>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# verdict hero card: risk gauge + verdict badge + KPIs (dark severity colors)
gauge_color = "#4ADE80" if risk_score < 30 else ("#FBBF24" if risk_score < 70 else "#F87171")
gauge_deg = risk_score / 100.0 * 360.0
verdict_class = "anomaly" if best_pred == 1 else "benign"
verdict_text = "ANOMALY — suspicious flow" if best_pred == 1 else "NORMAL — benign flow"

st.markdown(
    f"""
<div class="card hero">
  <div class="gauge" style="background:conic-gradient({gauge_color} {gauge_deg}deg, var(--muted) {gauge_deg}deg);">
    <div class="gauge-inner">
      <span class="gauge-value" style="color:{gauge_color};">{risk_score:.1f}%</span>
      <span class="gauge-label">risk</span>
    </div>
  </div>
  <div>
    <div class="verdict {verdict_class}"><span class="dot"></span>{verdict_text}</div>
    <div class="kpi-row">
      <div class="kpi"><span class="kpi-label">True family (reference)</span><span class="kpi-value">{true_family}</span></div>
      <div class="kpi"><span class="kpi-label">Detailed label</span><span class="kpi-value">{true_label}</span></div>
      <div class="kpi"><span class="kpi-label">Attack probability</span><span class="kpi-value">{risk_score:.2f} %</span></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# why this alert: feature importances + flow value vs BENIGN mean
st.markdown(
    '<div class="section-title"><span class="bar"></span>Why this alert?</div>'
    f'<p class="section-sub">Top features driving <b>{model_name}</b>, comparing the flow value to the '
    'mean of the BENIGN training traffic. Red bars deviate strongly from normal — these are the reasons.</p>',
    unsafe_allow_html=True,
)

top_n = st.slider("Number of features to explain", 5, 20, 10)

# importances from the model (XGBoost / trees -> feature_importances_)
if hasattr(best_model, "feature_importances_"):
    weights = np.asarray(best_model.feature_importances_)
elif hasattr(best_model, "coef_"):
    weights = np.abs(np.ravel(best_model.coef_))
else:
    weights = np.ones(len(features))

importances = pd.Series(weights, index=features)
top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

# table: flow value vs BENIGN mean + relative deviation
rows = []
for f in top_features:
    val = float(row[f])
    mean_val = float(benign_means[f])
    if mean_val != 0:
        deviation = (val - mean_val) / abs(mean_val)
    else:
        deviation = 0.0 if val == 0 else np.inf
    abnormal = abs(deviation) > 1.0
    rows.append({
        "Feature": f,
        "Flow value": val,
        "BENIGN mean": mean_val,
        "Relative deviation": deviation,
        "Abnormal": "Yes" if abnormal else "No",
    })
df_explain = pd.DataFrame(rows)

# chart: deviation per feature (capped for readability, exact values in tooltip)
df_explain["dev_plot"] = df_explain["Relative deviation"].clip(-25, 25)
bar = (
    alt.Chart(df_explain)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("dev_plot:Q", title="Deviation from BENIGN mean (× of mean, capped at ±25)"),
        y=alt.Y("Feature:N", sort="-x", title=None),
        color=alt.condition(
            alt.FieldEqualPredicate(field="Abnormal", equal="Yes"),
            alt.value("#F87171"), alt.value("#60A5FA")),
        tooltip=[
            alt.Tooltip("Feature:N"),
            alt.Tooltip("Flow value:Q", format=",.2f"),
            alt.Tooltip("BENIGN mean:Q", format=",.2f"),
            alt.Tooltip("Relative deviation:Q", format="+,.2f"),
            alt.Tooltip("Abnormal:N"),
        ],
    )
    .properties(height=30 * len(df_explain) + 20, background="transparent")
    .configure_axis(labelFont="Fira Sans", titleFont="Fira Sans", labelColor="#94A3B8",
                    titleColor="#94A3B8", grid=True, gridColor="#1E293B",
                    domainColor="#334155", tickColor="#334155")
    .configure_view(strokeWidth=0)
)
st.altair_chart(bar, use_container_width=True, theme=None)


# details table (highlight abnormal rows, dark red tint + light red text)
def highlight_abnormal(line):
    if line["Abnormal"] == "Yes":
        return ["background-color:#422023; color:#FCA5A5"] * len(line)
    return [""] * len(line)


with st.expander("Feature details (table)"):
    style_explain = df_explain.drop(columns=["dev_plot"]).style.apply(highlight_abnormal, axis=1).format(
        {"Flow value": "{:,.2f}", "BENIGN mean": "{:,.2f}", "Relative deviation": "{:+.2f}"})
    st.dataframe(style_explain, hide_index=True, use_container_width=True)

# analyst feedback
st.markdown(
    '<div class="section-title"><span class="bar"></span>Analyst feedback</div>'
    f'<p class="section-sub">Confirm the <b>{model_name}</b> verdict. Saved to '
    '<code>app/feedback.csv</code> to improve the model.</p>',
    unsafe_allow_html=True,
)

col_tp, col_fp = st.columns(2)
click_tp = col_tp.button("Confirm — True positive (real alert)", use_container_width=True)
click_fp = col_fp.button("Reject — False positive (false alarm)", use_container_width=True)

if click_tp or click_fp:
    feedback_label = "True positive" if click_tp else "False positive"
    new_row = pd.DataFrame([{
        "flow_index": int(flow_idx),
        "model_verdict": "ANOMALY" if best_pred == 1 else "NORMAL",
        "risk_score": round(risk_score, 4),
        "analyst_feedback": feedback_label,
    }])
    if os.path.exists(FEEDBACK_PATH):
        new_row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)
    st.success(f"Feedback saved: flow {flow_idx} → {feedback_label} "
               f"(verdict {'ANOMALY' if best_pred == 1 else 'NORMAL'}, score {risk_score:.2f} %).")

if os.path.exists(FEEDBACK_PATH):
    with st.expander("View the saved feedback log"):
        df_feedback = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(df_feedback, hide_index=True, use_container_width=True)

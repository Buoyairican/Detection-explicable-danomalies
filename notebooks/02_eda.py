# %% [markdown]
# # Phase 2 - Exploratory data analysis (EDA) of CICIDS2017
#
# Explainable network anomaly detection.
# This notebook loads the frozen dataset (clean parquet) and explores the
# class distribution, the binary BENIGN/attack imbalance, the correlations
# between flow features, the distributions of a few discriminating features
# and the signature of each attack family. Each figure is saved as a PNG in
# reports/figures.

# %% [markdown]
# ## 1. Imports and configuration

# %%
# imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# absolute project root path
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# figures output folder
FIGURES_DIR = BASE + "/reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# global seaborn style
sns.set(style="whitegrid")

# %% [markdown]
# ## 2. Load the frozen dataset
#
# Load the clean parquet (2,520,798 rows, 78 numeric features + Label +
# Label_binary + Label_group).

# %%
# load the clean parquet dataset
df = pd.read_parquet(BASE + "/data/processed/cicids_clean.parquet")
print(f"DataFrame shape: {df.shape}")
print(df.head())

# numeric features (everything except the 3 label columns)
target_cols = ["Label", "Label_binary", "Label_group"]
features = [c for c in df.columns if c not in target_cols]
print(f"\nNumber of numeric features: {len(features)}")
print(f"Remaining NaN check: {df.isnull().sum().sum()}")

# %% [markdown]
# ## 3. Class distribution of Label_group
#
# Count of attack families (BENIGN, DoS, DDoS, PortScan, BruteForce,
# WebAttack, Bot, Infiltration, Heartbleed).

# %%
# class distribution of Label_group (sorted countplot)
group_order = df["Label_group"].value_counts().index

print("=== Label_group class distribution ===")
print(df["Label_group"].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(data=df, y="Label_group", order=group_order, palette="viridis")
plt.title("Class distribution by family (Label_group)")
plt.xlabel("Number of flows")
plt.ylabel("Traffic family")
# log scale because rare classes (Heartbleed, Infiltration) are crushed
plt.xscale("log")
plt.savefig(FIGURES_DIR + "/01_repartition_labels.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Binary BENIGN vs attack imbalance
#
# Barplot of the Label_binary target proportions (0 = BENIGN, 1 = attack).

# %%
# binary imbalance: BENIGN vs attack proportions
counts_bin = df["Label_binary"].value_counts().sort_index()
prop_bin = df["Label_binary"].value_counts(normalize=True).sort_index() * 100
labels_bin = ["BENIGN (0)", "Attack (1)"]

print("=== Binary imbalance (Label_binary) ===")
print(f"BENIGN : {counts_bin[0]} flows ({prop_bin[0]:.4f} %)")
print(f"Attack : {counts_bin[1]} flows ({prop_bin[1]:.4f} %)")

plt.figure(figsize=(7, 6))
ax = sns.barplot(x=labels_bin, y=prop_bin.values, palette=["#2c7fb8", "#d95f0e"])
# annotate the percentage above each bar
for i, v in enumerate(prop_bin.values):
    ax.text(i, v + 1, f"{v:.2f} %", ha="center", fontweight="bold")
plt.title("Class imbalance: BENIGN vs Attack")
plt.xlabel("Binary class")
plt.ylabel("Proportion (%)")
plt.ylim(0, 100)
plt.savefig(FIGURES_DIR + "/02_desequilibre_binaire.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Correlation matrix of numeric features
#
# Heatmap of Pearson correlations over the 78 features. We sample 150,000
# rows to stay fast. No annotations (too many features).

# %%
# correlation matrix of numeric features (on a sample)
df_sample = df.sample(150000, random_state=42)
corr = df_sample[features].corr()

# find strongly correlated feature pairs (|corr| > 0.95)
corr_abs = corr.abs()
upper_mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
strong_pairs = corr_abs.where(upper_mask).stack().sort_values(ascending=False)
print("=== Most correlated feature pairs (|r| > 0.95) ===")
print(strong_pairs[strong_pairs > 0.95].head(15))

plt.figure(figsize=(16, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6})
plt.title("Correlation matrix of the 78 numeric features")
plt.xlabel("Features")
plt.ylabel("Features")
plt.savefig(FIGURES_DIR + "/03_correlation.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Distributions of discriminating flow features
#
# Boxplots of 4 flow features by binary class (BENIGN vs Attack), on the
# sample. Log scale on the y axis because the values are very spread out.

# %%
# distributions of 4 discriminating features by binary class
features_distrib = ["Flow Duration", "Flow Bytes/s", "Total Fwd Packets", "Fwd Packet Length Max"]

# copy the sample with a readable binary label
df_box = df_sample[features_distrib + ["Label_binary"]].copy()
df_box["Class"] = np.where(df_box["Label_binary"] == 0, "BENIGN", "Attack")

print("=== Median of discriminating features by class ===")
print(df_box.groupby("Class")[features_distrib].median())

# for the log scale we work on positive values (shift by 1)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()
for i, col in enumerate(features_distrib):
    serie = df_box.copy()
    serie[col] = serie[col].clip(lower=0) + 1
    sns.boxplot(data=serie, x="Class", y=col, ax=axs[i], palette=["#2c7fb8", "#d95f0e"])
    axs[i].set_yscale("log")
    axs[i].set_title(f"Distribution of '{col}' by class")
    axs[i].set_xlabel("Binary class")
    axs[i].set_ylabel(col + " (log scale)")
plt.suptitle("Distributions of discriminating flow features (150k sample)", fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR + "/04_distributions_features.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Heatmap of standardized means per attack family
#
# We standardize (z-score) the most variable features, then compute their
# mean per family (Label_group). This reveals which features characterize
# each attack type (signature).

# %%
# select the top features that best discriminate between families
# z-score standardization of all features on the sample
means = df_sample[features].mean()
stds = df_sample[features].std().replace(0, 1)  # avoid division by zero
df_std = (df_sample[features] - means) / stds
df_std["Label_group"] = df_sample["Label_group"].values

# standardized mean of each feature per family
group_profile = df_std.groupby("Label_group")[features].mean()

# keep the 20 features whose mean varies the most between families
variability = group_profile.std().sort_values(ascending=False)
top_features = variability.head(20).index.tolist()
print("=== Top 20 features that most differentiate the families ===")
for f in top_features:
    print(f"{f:35s} -> inter-family spread {variability[f]:.4f}")

# sub-matrix (families x top features), transposed for vertical reading
heat = group_profile[top_features].T

plt.figure(figsize=(12, 11))
sns.heatmap(heat, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            linewidths=0.5, cbar_kws={"label": "Standardized mean (z-score)"})
plt.title("Attack family signature (standardized means of top features)")
plt.xlabel("Family (Label_group)")
plt.ylabel("Feature")
plt.savefig(FIGURES_DIR + "/05_features_par_attaque.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Summary of saved figures

# %%
# check that the 5 figures are present on disk
figure_files = sorted(os.listdir(FIGURES_DIR))
print("=== Figures saved in reports/figures ===")
for f in figure_files:
    path = FIGURES_DIR + "/" + f
    size = os.path.getsize(path) / 1024
    print(f"{f:35s} -> {size:.4f} KB")

# %% [markdown]
# ## 9. Main observations (EDA summary)
#
# **1. Strong class imbalance.** Traffic is overwhelmingly BENIGN (~83.1 %,
# i.e. 2,095,057 flows) against ~16.9 % attacks (425,741 flows). Within the
# attacks, the families are also imbalanced: DoS and DDoS dominate (193,745
# and 128,014 flows), while Heartbleed (11) and Infiltration (36) are almost
# nonexistent. So an adapted strategy will be needed (stratify on the split,
# `class_weight`, or even undersampling / SMOTE) so the rare attacks are not
# ignored.
#
# **2. Discriminating flow features.** The boxplots show that 'Flow Bytes/s',
# 'Flow Duration', 'Total Fwd Packets' and 'Fwd Packet Length Max' clearly
# separate BENIGN from attacks: the attacks (DoS/DDoS in particular) show
# atypical throughput and packet volumes. These variables will be good
# predictors for the supervised and unsupervised models.
#
# **3. Strong correlations / redundancy.** Many feature pairs are almost
# perfectly correlated (|r| > 0.95), by construction of the dataset: e.g.
# 'Subflow Fwd Packets' ~ 'Total Fwd Packets', 'Fwd Header Length' ~
# 'Fwd Header Length.1', the Mean/Avg Segment Size and Packet Length
# Std/Variance pairs. This redundancy will later justify feature selection /
# reduction.
#
# **4. Per-family signatures.** The heatmap of standardized means reveals
# that each family activates a distinct subset of features: throughput
# attacks (DoS/DDoS) stand out on the packet and byte counters, PortScan on
# the flags and flow duration, BruteForce/WebAttack on the packet lengths.
# These signatures make explainable detection plausible (top features per
# alert).
#
# **5. Clean and usable data.** The frozen dataset no longer contains any NaN
# or Infinity and all features are numeric, which allows going straight to
# modeling (rule baseline + a bake-off of 7 supervised models).

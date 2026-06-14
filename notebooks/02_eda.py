# %% [markdown]
# # Phase 2 - Analyse exploratoire (EDA) du jeu CICIDS2017
#
# Detection explicable d'anomalies reseau.
# Ce notebook charge le dataset gele (parquet propre), explore la
# repartition des classes, le desequilibre binaire BENIGN/attaque, les
# correlations entre features de flux, les distributions de quelques
# features discriminantes et les signatures de chaque famille d'attaque.
# Chaque figure est sauvegardee en PNG dans reports/figures.

# %% [markdown]
# ## 1. Imports et configuration

# %%
# 1. Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chemin absolu de la racine du projet
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# Dossier de sortie des figures
DOSSIER_FIGURES = BASE + "/reports/figures"
os.makedirs(DOSSIER_FIGURES, exist_ok=True)

# Style global seaborn
sns.set(style="whitegrid")

# %% [markdown]
# ## 2. Chargement du dataset gele
#
# On charge le parquet propre (2 520 798 lignes, 78 features numeriques
# + Label + Label_binary + Label_group).

# %%
# 2. Chargement du dataset propre au format parquet
df = pd.read_parquet(BASE + "/data/processed/cicids_clean.parquet")
print(f"Shape du DataFrame : {df.shape}")
print(df.head())

# Liste des features numeriques (tout sauf les 3 colonnes de label)
target_cols = ["Label", "Label_binary", "Label_group"]
features = [c for c in df.columns if c not in target_cols]
print(f"\nNombre de features numeriques : {len(features)}")
print(f"Verification NaN restants : {df.isnull().sum().sum()}")

# %% [markdown]
# ## 3. Repartition des classes Label_group
#
# Comptage des familles d'attaques (BENIGN, DoS, DDoS, PortScan,
# BruteForce, WebAttack, Bot, Infiltration, Heartbleed).

# %%
# 3. Repartition des classes Label_group (countplot trie)
ordre_group = df["Label_group"].value_counts().index

print("=== Repartition des classes Label_group ===")
print(df["Label_group"].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(data=df, y="Label_group", order=ordre_group, palette="viridis")
plt.title("Repartition des classes par famille (Label_group)")
plt.xlabel("Nombre de flux")
plt.ylabel("Famille de trafic")
# Echelle logarithmique car les classes rares (Heartbleed, Infiltration) sont ecrasees
plt.xscale("log")
plt.savefig(DOSSIER_FIGURES + "/01_repartition_labels.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Desequilibre binaire BENIGN vs attaque
#
# Barplot des proportions de la cible Label_binary (0 = BENIGN, 1 = attaque).

# %%
# 4. Desequilibre binaire : proportions BENIGN vs attaque
counts_bin = df["Label_binary"].value_counts().sort_index()
prop_bin = df["Label_binary"].value_counts(normalize=True).sort_index() * 100
labels_bin = ["BENIGN (0)", "Attaque (1)"]

print("=== Desequilibre binaire (Label_binary) ===")
print(f"BENIGN  : {counts_bin[0]} flux ({prop_bin[0]:.4f} %)")
print(f"Attaque : {counts_bin[1]} flux ({prop_bin[1]:.4f} %)")

plt.figure(figsize=(7, 6))
ax = sns.barplot(x=labels_bin, y=prop_bin.values, palette=["#2c7fb8", "#d95f0e"])
# Annotation des pourcentages au-dessus de chaque barre
for i, v in enumerate(prop_bin.values):
    ax.text(i, v + 1, f"{v:.2f} %", ha="center", fontweight="bold")
plt.title("Desequilibre des classes : BENIGN vs Attaque")
plt.xlabel("Classe binaire")
plt.ylabel("Proportion (%)")
plt.ylim(0, 100)
plt.savefig(DOSSIER_FIGURES + "/02_desequilibre_binaire.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Matrice de correlation des features numeriques
#
# Heatmap des correlations de Pearson sur les 78 features. On echantillonne
# 150 000 lignes pour rester rapide. Pas d'annotations (trop de features).

# %%
# 5. Matrice de correlation des features numeriques (sur echantillon)
df_sample = df.sample(150000, random_state=42)
corr = df_sample[features].corr()

# Reperage des paires de features fortement correlees (|corr| > 0.95)
corr_abs = corr.abs()
masque_haut = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
paires_fortes = corr_abs.where(masque_haut).stack().sort_values(ascending=False)
print("=== Paires de features les plus correlees (|r| > 0.95) ===")
print(paires_fortes[paires_fortes > 0.95].head(15))

plt.figure(figsize=(16, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6})
plt.title("Matrice de correlation des 78 features numeriques")
plt.xlabel("Features")
plt.ylabel("Features")
plt.savefig(DOSSIER_FIGURES + "/03_correlation.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Distributions de features de flux discriminantes
#
# Boxplots de 4 features de flux par classe binaire (BENIGN vs Attaque),
# sur echantillon. Axe y en echelle logarithmique car les valeurs sont
# tres etalees.

# %%
# 6. Distributions de 4 features discriminantes par classe binaire
features_distrib = ["Flow Duration", "Flow Bytes/s", "Total Fwd Packets", "Fwd Packet Length Max"]

# Copie de l'echantillon avec un libelle binaire lisible
df_box = df_sample[features_distrib + ["Label_binary"]].copy()
df_box["Classe"] = np.where(df_box["Label_binary"] == 0, "BENIGN", "Attaque")

print("=== Statistiques des features discriminantes par classe ===")
print(df_box.groupby("Classe")[features_distrib].median())

# Pour le log on travaille sur des valeurs positives (decalage de 1)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()
for i, col in enumerate(features_distrib):
    serie = df_box.copy()
    serie[col] = serie[col].clip(lower=0) + 1
    sns.boxplot(data=serie, x="Classe", y=col, ax=axs[i], palette=["#2c7fb8", "#d95f0e"])
    axs[i].set_yscale("log")
    axs[i].set_title(f"Distribution de '{col}' par classe")
    axs[i].set_xlabel("Classe binaire")
    axs[i].set_ylabel(col + " (echelle log)")
plt.suptitle("Distributions de features de flux discriminantes (echantillon 150k)", fontsize=14)
plt.tight_layout()
plt.savefig(DOSSIER_FIGURES + "/04_distributions_features.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Heatmap des moyennes standardisees par famille d'attaque
#
# On standardise (z-score) les features les plus variables, puis on calcule
# leur moyenne par famille (Label_group). Cela revele quelles features
# caracterisent chaque type d'attaque (signature).

# %%
# 7. Selection des top features les plus discriminantes entre familles
# Standardisation z-score de toutes les features sur l'echantillon
moyennes = df_sample[features].mean()
ecarts = df_sample[features].std().replace(0, 1)  # eviter la division par zero
df_std = (df_sample[features] - moyennes) / ecarts
df_std["Label_group"] = df_sample["Label_group"].values

# Moyenne standardisee de chaque feature par famille
profil_groupes = df_std.groupby("Label_group")[features].mean()

# On garde les 20 features dont la moyenne varie le plus entre familles
variabilite = profil_groupes.std().sort_values(ascending=False)
top_features = variabilite.head(20).index.tolist()
print("=== Top 20 features qui differencient le plus les familles ===")
for f in top_features:
    print(f"{f:35s} -> ecart inter-familles {variabilite[f]:.4f}")

# Sous-matrice (familles x top features), transposee pour lecture verticale
heat = profil_groupes[top_features].T

plt.figure(figsize=(12, 11))
sns.heatmap(heat, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            linewidths=0.5, cbar_kws={"label": "Moyenne standardisee (z-score)"})
plt.title("Signature des familles d'attaque (moyennes standardisees des top features)")
plt.xlabel("Famille (Label_group)")
plt.ylabel("Feature")
plt.savefig(DOSSIER_FIGURES + "/05_features_par_attaque.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Recapitulatif des figures sauvegardees

# %%
# 8. Verification que les 5 figures sont bien presentes sur le disque
fichiers_figures = sorted(os.listdir(DOSSIER_FIGURES))
print("=== Figures sauvegardees dans reports/figures ===")
for f in fichiers_figures:
    chemin = DOSSIER_FIGURES + "/" + f
    taille = os.path.getsize(chemin) / 1024
    print(f"{f:35s} -> {taille:.4f} Ko")

# %% [markdown]
# ## 9. Observations principales (resume EDA)
#
# **1. Fort desequilibre des classes.** Le trafic est tres majoritairement
# BENIGN (~83,1 %, soit 2 095 057 flux) contre ~16,9 % d'attaques
# (425 741 flux). Au sein des attaques, les familles sont elles aussi
# desequilibrees : DoS et DDoS dominent (193 745 et 128 014 flux), tandis
# que Heartbleed (11) et Infiltration (36) sont quasi inexistantes. Il
# faudra donc une strategie adaptee (stratify au split, `class_weight`,
# voire sous-echantillonnage / SMOTE) pour ne pas ignorer les attaques rares.
#
# **2. Features de flux discriminantes.** Les boxplots montrent que
# 'Flow Bytes/s', 'Flow Duration', 'Total Fwd Packets' et
# 'Fwd Packet Length Max' separent nettement BENIGN des attaques : les
# attaques (DoS/DDoS notamment) presentent des debits et des volumes de
# paquets atypiques. Ces variables seront de bons predicteurs pour les
# modeles supervises et non supervises.
#
# **3. Correlations fortes / redondance.** De nombreuses paires de features
# sont quasi parfaitement correlees (|r| > 0,95), par construction du
# dataset : par ex. 'Subflow Fwd Packets' ~ 'Total Fwd Packets',
# 'Fwd Header Length' ~ 'Fwd Header Length.1', les paires
# Mean/Avg Segment Size et Packet Length Std/Variance. Cette redondance
# justifiera plus tard une selection / reduction de features.
#
# **4. Signatures par famille d'attaque.** La heatmap des moyennes
# standardisees revele que chaque famille active un sous-ensemble de
# features distinct : les attaques par debit (DoS/DDoS) se distinguent sur
# les compteurs de paquets et de bytes, PortScan sur les flags et la duree
# de flux, BruteForce/WebAttack sur les longueurs de paquets. Ces
# signatures rendent la detection explicable plausible (top features par
# alerte).
#
# **5. Donnees propres et exploitables.** Le dataset gele ne contient plus
# aucun NaN ni Infinity et toutes les features sont numeriques, ce qui
# permet de passer directement a la modelisation (baseline de regles,
# Isolation Forest puis Random Forest).

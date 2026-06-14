# %% [markdown]
# # Phase 1 - Nettoyage du jeu de donnees CICIDS2017
#
# Detection explicable d'anomalies reseau.
# Ce notebook charge les 8 CSV bruts, nettoie les noms de colonnes,
# normalise les labels, force les features en numerique, supprime les
# Infinity/NaN et les doublons, allege la memoire, cree les colonnes
# cibles puis sauvegarde un fichier parquet propre.

# %% [markdown]
# ## 1. Imports

# %%
# 1. Imports
import glob
import pandas as pd
import numpy as np

# Chemin absolu de la racine du projet
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# %% [markdown]
# ## 2. Chargement des 8 CSV et concatenation

# %%
# 2. Chargement des 8 CSV bruts (boucle glob) avec encoding latin-1
fichiers = sorted(glob.glob(BASE + "/data/raw/*.csv"))
print(f"Nombre de fichiers CSV trouves : {len(fichiers)}")

dfs = []
for f in fichiers:
    df_tmp = pd.read_csv(f, encoding="latin-1", low_memory=False)
    print(f"{f.split('/')[-1]:55s} -> {df_tmp.shape[0]} lignes, {df_tmp.shape[1]} colonnes")
    dfs.append(df_tmp)

# Concatenation en un seul DataFrame
df = pd.concat(dfs, ignore_index=True)
print(f"\nShape totale apres concatenation : {df.shape}")

# %% [markdown]
# ## 3. Nettoyage des noms de colonnes
#
# Piege CICIDS2017 : 65 colonnes ont des espaces en debut/fin de nom.

# %%
# 3. Nettoyage des noms de colonnes (suppression des espaces)
print(f"Colonnes avec espaces avant nettoyage : {sum(1 for c in df.columns if c != c.strip())}")
df.columns = df.columns.str.strip()
print(f"Colonnes avec espaces apres nettoyage : {sum(1 for c in df.columns if c != c.strip())}")
print(f"Nom de la colonne cible : {repr('Label')} present -> {'Label' in df.columns}")

# %% [markdown]
# ## 4. Normalisation du label
#
# On retire les espaces, on remplace les caracteres non-ASCII (mojibake des
# 'Web Attack') par '-' puis on retire de nouveau les espaces.

# %%
# 4. Normalisation du label : strip + remplacement des caracteres non-ASCII par '-'
print("Labels bruts (avant normalisation) :")
print(df["Label"].value_counts())

df["Label"] = df["Label"].str.strip()
df["Label"] = df["Label"].str.replace(r"[^\x00-\x7F]+", "-", regex=True)
df["Label"] = df["Label"].str.strip()

print("\nLabels apres normalisation :")
print(df["Label"].value_counts())

# %% [markdown]
# ## 5. Conversion des features en numerique et gestion des Infinity
#
# Toutes les colonnes sauf Label sont forcees en numerique, puis les
# valeurs +/-Infinity sont remplacees par NaN.

# %%
# 5. Forcer les colonnes de features en numerique (tout sauf Label)
features = [c for c in df.columns if c != "Label"]
print(f"Nombre de colonnes de features : {len(features)}")

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remplacement des +/-Infinity par NaN
df = df.replace([np.inf, -np.inf], np.nan)
print(f"Nombre total de NaN apres conversion : {df.isnull().sum().sum()}")

# %% [markdown]
# ## 6. Suppression des NaN puis des doublons

# %%
# 6. Suppression des lignes avec NaN puis des doublons
lignes_avant = df.shape[0]
print(f"Lignes avant nettoyage : {lignes_avant}")

# Suppression des lignes contenant des NaN
df = df.dropna()
lignes_apres_nan = df.shape[0]
print(f"Lignes supprimees (NaN/Inf) : {lignes_avant - lignes_apres_nan}")

# Suppression des doublons
df = df.drop_duplicates()
lignes_apres_dupes = df.shape[0]
print(f"Lignes supprimees (doublons) : {lignes_apres_nan - lignes_apres_dupes}")
print(f"Lignes restantes : {lignes_apres_dupes}")

# %% [markdown]
# ## 7. Downcast float64 -> float32 (allegement memoire)

# %%
# 7. Downcast des colonnes float64 vers float32 pour reduire la memoire
memoire_avant = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memoire avant downcast : {memoire_avant:.4f} Mo")

for col in df.select_dtypes("float64").columns:
    df[col] = df[col].astype("float32")

memoire_apres = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memoire apres downcast : {memoire_apres:.4f} Mo")

# %% [markdown]
# ## 8. Creation des colonnes cibles Label_binary et Label_group

# %%
# 8. Creation de Label_binary (0 = BENIGN, 1 = attaque)
df["Label_binary"] = np.where(df["Label"] == "BENIGN", 0, 1)

# Mapping des familles d'attaques pour Label_group
mapping_group = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Application du mapping (les labels 'Web Attack ...' restent NaN puis seront combles)
df["Label_group"] = df["Label"].map(mapping_group)

# Fallback : tout label commencant par 'Web Attack' devient WebAttack
df.loc[df["Label"].str.startswith("Web Attack"), "Label_group"] = "WebAttack"

print(f"Labels non mappes restants : {df['Label_group'].isnull().sum()}")

# %% [markdown]
# ## 9. Recapitulatif

# %%
# 9. Recapitulatif final
print(f"Shape finale du DataFrame : {df.shape}")

print("\n=== Repartition de Label ===")
print(df["Label"].value_counts())

print("\n=== Repartition de Label_binary (0 = BENIGN, 1 = attaque) ===")
print(df["Label_binary"].value_counts())

print("\n=== Repartition de Label_group (familles d'attaques) ===")
print(df["Label_group"].value_counts())

# Verification absence de NaN / Inf
print(f"\nNombre total de NaN restants : {df.isnull().sum().sum()}")

# %% [markdown]
# ## 10. Sauvegarde au format parquet

# %%
# 10. Sauvegarde du DataFrame nettoye en parquet
chemin_sortie = BASE + "/data/processed/cicids_clean.parquet"
df.to_parquet(chemin_sortie, index=False)
print(f"Fichier sauvegarde : {chemin_sortie}")
print(f"Colonnes sauvegardees : {df.shape[1]}")
print("Sauvegarde terminee.")

# %% [markdown]
# # Phase 1 - CICIDS2017 data cleaning
#
# Explainable network anomaly detection.
# Load the 8 raw CSVs, clean column names, normalize labels, force features
# to numeric, drop Infinity/NaN and duplicates, shrink memory, build the
# target columns, then save a clean parquet file.

# %% [markdown]
# ## Imports

# %%
import glob
import pandas as pd
import numpy as np

# absolute project root
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# %% [markdown]
# ## Load the 8 raw CSVs and concatenate

# %%
# load data
files = sorted(glob.glob(BASE + "/data/raw/*.csv"))
print(f"CSV files found: {len(files)}")

dfs = []
for f in files:
    df_tmp = pd.read_csv(f, encoding="latin-1", low_memory=False)
    print(f"{f.split('/')[-1]:55s} -> {df_tmp.shape[0]} rows, {df_tmp.shape[1]} cols")
    dfs.append(df_tmp)

# concatenate into a single DataFrame
df = pd.concat(dfs, ignore_index=True)
print(f"\nShape after concatenation: {df.shape}")

# %% [markdown]
# ## Clean column names
#
# CICIDS2017 gotcha: many columns have leading/trailing spaces.

# %%
# strip column names
print(f"Columns with spaces before: {sum(1 for c in df.columns if c != c.strip())}")
df.columns = df.columns.str.strip()
print(f"Columns with spaces after: {sum(1 for c in df.columns if c != c.strip())}")
print(f"Target column 'Label' present -> {'Label' in df.columns}")

# %% [markdown]
# ## Normalize the label
#
# Strip spaces, replace non-ASCII characters (mojibake in 'Web Attack')
# with '-', then strip again.

# %%
# normalize label
print("Raw labels:")
print(df["Label"].value_counts())

df["Label"] = df["Label"].str.strip()
df["Label"] = df["Label"].str.replace(r"[^\x00-\x7F]+", "-", regex=True)
df["Label"] = df["Label"].str.strip()

print("\nLabels after normalization:")
print(df["Label"].value_counts())

# %% [markdown]
# ## Convert features to numeric and handle Infinity
#
# Every column except Label is forced to numeric, then +/-Infinity is
# replaced by NaN.

# %%
# force feature columns to numeric (everything except Label)
features = [c for c in df.columns if c != "Label"]
print(f"Feature columns: {len(features)}")

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# replace +/-Infinity with NaN
df = df.replace([np.inf, -np.inf], np.nan)
print(f"Total NaN after conversion: {df.isnull().sum().sum()}")

# %% [markdown]
# ## Drop NaN then duplicates

# %%
# drop NaN rows then duplicates
rows_before = df.shape[0]
print(f"Rows before cleaning: {rows_before}")

df = df.dropna()
rows_after_nan = df.shape[0]
print(f"Rows dropped (NaN/Inf): {rows_before - rows_after_nan}")

df = df.drop_duplicates()
rows_after_dupes = df.shape[0]
print(f"Rows dropped (duplicates): {rows_after_nan - rows_after_dupes}")
print(f"Rows remaining: {rows_after_dupes}")

# %% [markdown]
# ## Downcast float64 -> float32 (shrink memory)

# %%
# downcast float64 columns to float32
mem_before = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory before downcast: {mem_before:.4f} MB")

for col in df.select_dtypes("float64").columns:
    df[col] = df[col].astype("float32")

mem_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory after downcast: {mem_after:.4f} MB")

# %% [markdown]
# ## Build the target columns Label_binary and Label_group

# %%
# Label_binary (0 = BENIGN, 1 = attack)
df["Label_binary"] = np.where(df["Label"] == "BENIGN", 0, 1)

# attack family mapping for Label_group
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

# apply the mapping ('Web Attack ...' labels stay NaN for now)
df["Label_group"] = df["Label"].map(mapping_group)

# fallback: any label starting with 'Web Attack' becomes WebAttack
df.loc[df["Label"].str.startswith("Web Attack"), "Label_group"] = "WebAttack"

print(f"Unmapped labels remaining: {df['Label_group'].isnull().sum()}")

# %% [markdown]
# ## Summary

# %%
# final summary
print(f"Final DataFrame shape: {df.shape}")

print("\n=== Label distribution ===")
print(df["Label"].value_counts())

print("\n=== Label_binary distribution (0 = BENIGN, 1 = attack) ===")
print(df["Label_binary"].value_counts())

print("\n=== Label_group distribution (attack families) ===")
print(df["Label_group"].value_counts())

# check no NaN / Inf remain
print(f"\nTotal NaN remaining: {df.isnull().sum().sum()}")

# %% [markdown]
# ## Save to parquet

# %%
# save the clean DataFrame to parquet
out_path = BASE + "/data/processed/cicids_clean.parquet"
df.to_parquet(out_path, index=False)
print(f"File saved: {out_path}")
print(f"Columns saved: {df.shape[1]}")
print("Save complete.")

# %% [markdown]
# # Phase 3 - Modelisation : bake-off de modeles supervises (CICIDS2017)
#
# Ce notebook prepare les donnees, construit deux baselines interpretables
# (regles statiques + Isolation Forest), puis lance un BAKE-OFF de 7 modeles
# supervises compares par validation croisee (10-fold) sur un sous-echantillon
# stratifie. On affine chaque modele par une petite grille (GridSearchCV) et on
# selectionne le meilleur au F1. Le gagnant est re-entraine sur TOUT le train et
# sauvegarde sous forme de pipeline (StandardScaler + modele) consommant des
# features BRUTES. On produit aussi tous les artefacts consommes par
# l'evaluation (Phase 4) et le dashboard Streamlit (Phase 5).

# %% [markdown]
# ## 1. Imports et chargement du parquet gele

# %%
# 1. Imports
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

# Chemin absolu de la racine du projet
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"

# Chargement du jeu de donnees nettoye (Phase 1)
df = pd.read_parquet(BASE + "/data/processed/cicids_clean.parquet")
print(f"Shape du DataFrame charge : {df.shape}")
print(df["Label_binary"].value_counts())

# %% [markdown]
# ## 2. Definition des features et de la cible
#
# Les features sont toutes les colonnes numeriques, c'est-a-dire toutes
# les colonnes SAUF les trois colonnes de labels. La cible est binaire :
# `Label_binary` (0 = BENIGN, 1 = attaque).

# %%
# 2. Features = toutes les colonnes sauf les labels ; cible = Label_binary
colonnes_labels = ["Label", "Label_binary", "Label_group"]
features = [c for c in df.columns if c not in colonnes_labels]
target = "Label_binary"
print(f"Nombre de features candidates : {len(features)}")
print(f"Colonne cible : {target}")

# %% [markdown]
# ## 3. Suppression des features a variance nulle
#
# Les colonnes constantes (ecart-type nul) n'apportent aucune information
# au modele et sont donc retirees de la liste des features. On compte
# combien sont supprimees puis on sauvegarde la liste finale ordonnee.

# %%
# 3. Suppression des features constantes (variance nulle)
ecarts_types = df[features].std()
features_constantes = ecarts_types[ecarts_types == 0].index.tolist()
print(f"Nombre de features constantes (variance nulle) supprimees : {len(features_constantes)}")
print(f"Features supprimees : {features_constantes}")

# Liste finale des features conservees (ordre stable)
features = [c for c in features if c not in features_constantes]
print(f"Nombre de features conservees : {len(features)}")

# Sauvegarde de la liste ordonnee des features
joblib.dump(features, BASE + "/models/features.pkl")
print("Liste des features sauvegardee -> models/features.pkl")

# %% [markdown]
# ## 4. Split train / test stratifie
#
# Decoupage 70 % train / 30 % test, stratifie sur `Label_binary` pour
# conserver la meme proportion BENIGN/attaque dans les deux jeux.

# %%
# 4. Split train/test stratifie sur la cible binaire
df_train, df_test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Label_binary"]
)

# Construction des matrices de features et des cibles
X_train = df_train[features]
X_test = df_test[features]
y_train = df_train["Label_binary"]
y_test = df_test["Label_binary"]

print(f"Train : {X_train.shape[0]} lignes")
print(f"Test  : {X_test.shape[0]} lignes")
print(f"Proportion d'attaques (train) : {y_train.mean():.4f}")
print(f"Proportion d'attaques (test)  : {y_test.mean():.4f}")

# %% [markdown]
# ## 5. Standardisation des features
#
# Le `StandardScaler` est ajuste UNIQUEMENT sur le train puis applique au
# train et au test. Ce scaler gele sert a la baseline Isolation Forest et
# reste un artefact du contrat (Phase 4). Les pipelines du bake-off, eux,
# embarquent leur PROPRE StandardScaler et consomment des features BRUTES.

# %%
# 5. Standardisation : fit sur le train, transform sur train et test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Moyenne moyenne des features (train standardise) : {X_train_s.mean():.4f}")
print(f"Ecart-type moyen des features (train standardise) : {X_train_s.std():.4f}")

# Sauvegarde du scaler
joblib.dump(scaler, BASE + "/models/scaler.pkl")
print("Scaler sauvegarde -> models/scaler.pkl")

# %% [markdown]
# ## 6. Baseline de regles statiques (interpretable)
#
# Baseline VOLONTAIREMENT simpliste : quelques regles a seuils fixes sur des
# features de volume/vitesse BRUTES (non mises a l'echelle). Les seuils sont
# choisis proches des 95e/99e percentiles du trafic BENIGN. Une ligne est
# flaggee comme attaque si AU MOINS une regle se declenche (OR logique).
# On attend un taux de faux positifs eleve : c'est le point de comparaison
# face aux modeles appris.

# %%
# 6a. Inspection rapide de quelques features de volume/vitesse sur le BENIGN
benign_train = X_train[y_train == 0]
for f in ["Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets",
          "Fwd Packet Length Max", "Destination Port", "Bwd Packets/s",
          "Average Packet Size", "Flow Duration"]:
    p95 = benign_train[f].quantile(0.95)
    p99 = benign_train[f].quantile(0.99)
    print(f"{f:25s} BENIGN p95={p95:.2f}  p99={p99:.2f}")

# %%
# 6b. Definition des seuils (proches des p95/p99 du BENIGN) et des 5 regles
seuil_avg_packet_size = 496.0      # taille moyenne de paquet elevee (floods DoS/DDoS a payload)
seuil_bwd_packets_s = 58823.0      # debit de paquets retour eleve (p99 BENIGN)
seuil_flow_packets_s = 500000.0    # debit de paquets total tres eleve (p95 BENIGN)
seuil_fwd_packets_court = 3        # tres peu de paquets aller...
seuil_flow_duration_long = 1000000.0  # ...mais flux long (connexions sondes/lentes)
seuil_flow_bytes_s = 12000000.0    # debit d'octets tres eleve (p99 BENIGN)

# Regles individuelles sur les features BRUTES (X_test)
regle_avg_size = X_test["Average Packet Size"] > seuil_avg_packet_size
regle_bwd_rate = X_test["Bwd Packets/s"] > seuil_bwd_packets_s
regle_flow_rate = X_test["Flow Packets/s"] > seuil_flow_packets_s
regle_court_long = (X_test["Total Fwd Packets"] <= seuil_fwd_packets_court) & (X_test["Flow Duration"] > seuil_flow_duration_long)
regle_bytes_rate = X_test["Flow Bytes/s"] > seuil_flow_bytes_s

# OR logique des 5 regles -> prediction de la baseline
y_pred_rule = (regle_avg_size | regle_bwd_rate | regle_flow_rate | regle_court_long | regle_bytes_rate).astype(int)

# 6c. Metriques de la baseline de regles sur le test
recall_rule = recall_score(y_test, y_pred_rule)
cm_rule = confusion_matrix(y_test, y_pred_rule)
# Faux positifs = BENIGN (y=0) predits attaque (1) -> cellule [0, 1]
fp_rule = int(cm_rule[0, 1])
print(f"\nBaseline de regles - Recall (attaque) : {recall_rule:.4f}")
print(f"Baseline de regles - Faux positifs (BENIGN flagges attaque) : {fp_rule}")
print(f"Baseline de regles - Taux de faux positifs : {fp_rule / (y_test == 0).sum():.4f}")
print("\nMatrice de confusion (regles) :")
print(cm_rule)

# %%
# 6d. Sauvegarde des regles et seuils dans un fichier JSON lisible
rule_baseline = {
    "description": "Baseline de regles statiques sur features brutes. Une ligne est "
                   "flaggee comme attaque si AU MOINS une regle se declenche (OR logique). "
                   "Seuils choisis proches des 95e/99e percentiles du trafic BENIGN du train. "
                   "Baseline volontairement simpliste, taux de faux positifs eleve attendu.",
    "logique": "OR",
    "regles": [
        {"feature": "Average Packet Size", "operateur": ">", "seuil": seuil_avg_packet_size,
         "justification": "Taille moyenne de paquet elevee typique des floods DoS/DDoS a payload."},
        {"feature": "Bwd Packets/s", "operateur": ">", "seuil": seuil_bwd_packets_s,
         "justification": "Debit de paquets retour eleve (99e percentile du BENIGN)."},
        {"feature": "Flow Packets/s", "operateur": ">", "seuil": seuil_flow_packets_s,
         "justification": "Debit de paquets total tres eleve (95e percentile du BENIGN)."},
        {"feature_1": "Total Fwd Packets", "operateur_1": "<=", "seuil_1": seuil_fwd_packets_court,
         "feature_2": "Flow Duration", "operateur_2": ">", "seuil_2": seuil_flow_duration_long,
         "justification": "Tres peu de paquets aller mais flux long : connexions de sonde/lentes."},
        {"feature": "Flow Bytes/s", "operateur": ">", "seuil": seuil_flow_bytes_s,
         "justification": "Debit d'octets tres eleve (99e percentile du BENIGN)."},
    ],
    "recall_attaque_test": round(float(recall_rule), 4),
    "faux_positifs_test": fp_rule,
}
with open(BASE + "/models/rule_baseline.json", "w") as fichier:
    json.dump(rule_baseline, fichier, indent=2, ensure_ascii=False)
print("Regles sauvegardees -> models/rule_baseline.json")

# %% [markdown]
# ## 7. Isolation Forest (apprentissage du comportement normal)
#
# L'Isolation Forest apprend la structure du trafic NORMAL : on l'entraine
# uniquement sur le BENIGN du train (features mises a l'echelle). A la
# prediction, il renvoie -1 pour une anomalie et 1 pour un point normal ;
# on convertit -1 -> 1 (attaque) et 1 -> 0 (BENIGN).

# %%
# 7. Isolation Forest entraine sur le BENIGN du train uniquement
X_train_s_benign = X_train_s[y_train.values == 0]
print(f"Lignes BENIGN du train pour l'entrainement : {X_train_s_benign.shape[0]}")

iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_train_s_benign)

# Prediction sur le test : -1 = anomalie -> 1, 1 = normal -> 0
iso_pred_raw = iso.predict(X_test_s)
y_pred_iso = np.where(iso_pred_raw == -1, 1, 0)  # -1 anomalie -> attaque (1), 1 normal -> BENIGN (0)

# Metriques
recall_iso = recall_score(y_test, y_pred_iso)
cm_iso = confusion_matrix(y_test, y_pred_iso)
fp_iso = int(cm_iso[0, 1])
print(f"\nIsolation Forest - Recall (attaque) : {recall_iso:.4f}")
print(f"Isolation Forest - Faux positifs : {fp_iso}")
print("Matrice de confusion (Isolation Forest) :")
print(cm_iso)

# Sauvegarde du modele
joblib.dump(iso, BASE + "/models/isolation_forest.pkl")
print("Isolation Forest sauvegarde -> models/isolation_forest.pkl")

# %% [markdown]
# ## 8. Bake-off de 7 modeles supervises (selection par validation croisee)
#
# On compare 7 modeles supervises et on selectionne le meilleur au F1 (classe
# attaque). Chaque modele est enveloppe dans `make_pipeline(StandardScaler(),
# modele)` pour une entree BRUTE uniforme (le scaler interne re-met a l'echelle ;
# il est inoffensif pour les arbres). La comparaison se fait par validation
# croisee stratifiee 10-fold sur un SOUS-ECHANTILLON du train (pour la vitesse),
# puis chaque modele est affine par une petite grille (GridSearchCV).

# %% [markdown]
# ### 8.1. Sous-echantillon stratifie du train (~150 000 lignes)
#
# La comparaison croisee sur 1,76 M de lignes x 7 modeles x 10 folds serait trop
# longue. On tire un sous-echantillon stratifie d'environ 150 000 lignes du TRAIN,
# sur les features BRUTES `df_train[features]` (les pipelines re-scalent en interne).

# %%
# 1. Sous-echantillon stratifie d'environ 150 000 lignes du train (features BRUTES)
n_sub = 150000
frac_sub = n_sub / len(df_train)
df_sub = df_train.groupby("Label_binary", group_keys=False).sample(frac=frac_sub, random_state=42)
X_sub = df_sub[features]                 # features BRUTES (les pipelines re-scalent)
y_sub = df_sub["Label_binary"]
print(f"Sous-echantillon : {X_sub.shape[0]} lignes (sur {len(df_train)} du train)")
print(f"Proportion d'attaques (sous-echantillon) : {y_sub.mean():.4f}")

# %% [markdown]
# ### 8.2. Dictionnaire des 7 pipelines
#
# Chaque pipeline = `make_pipeline(StandardScaler(), modele)`. On garde les
# `n_jobs` des modeles par defaut pour eviter le sur-parallelisme imbrique : la
# parallelisation se fera au niveau de la validation croisee (`n_jobs=-1`).

# %%
# 2. Dictionnaire des 7 pipelines (StandardScaler + modele)
pipelines = {
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(C=0.5, random_state=42)),
    "DecisionTree": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3, random_state=42)),
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    "GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
    "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
    "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}
print(f"Nombre de modeles dans le panel : {len(pipelines)}")
for nom in pipelines:
    print(f"  - {nom}")

# %% [markdown]
# ### 8.3. Comparaison par validation croisee (cross_val_score, F1)
#
# Pour chaque modele : `cross_val_score` en 10-fold stratifie (shuffle,
# random_state=42), scoring `f1`, `n_jobs=-1`. On affiche le F1 moyen +/- ecart-type.

# %%
# 3. Comparaison croisee de chaque modele (F1 moyen +/- std)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_f1_mean = {}
cv_f1_std = {}
print("=== Comparaison croisee (10-fold, F1 classe attaque) ===")
for nom, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="f1", n_jobs=-1)
    cv_f1_mean[nom] = float(scores.mean())
    cv_f1_std[nom] = float(scores.std())
    print(f"{nom:20s} F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

# %% [markdown]
# ### 8.4. Tuning par GridSearchCV (petite grille par modele)
#
# Meme schema de validation croisee, scoring `f1`, `n_jobs=-1`. Petite grille par
# modele. On recupere `best_params_` et `best_score_` (F1 CV affine) de chacun.
# La grille cible le modele dans le pipeline : les cles sont prefixees par le nom
# d'etape minuscule genere par `make_pipeline` (ex. `randomforestclassifier__...`).

# %%
# 4. Petite grille d'hyperparametres par modele (cles = etape pipeline __ param)
grilles = {
    "LinearSVC": {"linearsvc__C": [0.5, 1, 1.5, 2, 2.5, 3]},
    "DecisionTree": {"decisiontreeclassifier__max_depth": [3, 5, 10, 15, 20, 30]},
    "LogisticRegression": {"logisticregression__C": [0.5, 1, 5, 10, 20, 30]},
    "GaussianNB": {"gaussiannb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]},
    "RandomForest": {"randomforestclassifier__max_depth": [10, 20, 30, 40, None], "randomforestclassifier__n_estimators": [100, 300]},
    "XGBoost": {"xgbclassifier__max_depth": [4, 6, 8, 10, 12], "xgbclassifier__learning_rate": [0.1, 0.3, 0.5], "xgbclassifier__n_estimators": [100, 300]},
    "KNN": {"kneighborsclassifier__n_neighbors": [1, 2, 3, 4, 5, 7]},
}

# Tuning : pour chaque modele on lance GridSearchCV et on garde best_params_/best_score_
tuned_cv_f1 = {}
best_params = {}
print("=== Tuning par GridSearchCV (10-fold, F1) ===")
for nom, pipe in pipelines.items():
    grille = GridSearchCV(pipe, grilles[nom], cv=cv, scoring="f1", n_jobs=-1)
    grille.fit(X_sub, y_sub)
    tuned_cv_f1[nom] = float(grille.best_score_)
    best_params[nom] = grille.best_params_
    print(f"{nom:20s} F1 tune = {grille.best_score_:.4f}  best_params = {grille.best_params_}")

# %% [markdown]
# ### 8.5. Sauvegarde du tableau de resultats (cv_results.json)
#
# Pour chaque modele : `cv_f1_mean`, `cv_f1_std`, `tuned_cv_f1`, `best_params`,
# plus un drapeau `selected` pour le gagnant.

# %%
# 5. Selection du gagnant (plus haut F1 CV affine) et sauvegarde de cv_results.json
nom_gagnant = max(tuned_cv_f1, key=tuned_cv_f1.get)
print(f"Modele gagnant (plus haut F1 CV affine) : {nom_gagnant}  (F1 = {tuned_cv_f1[nom_gagnant]:.4f})")

cv_results = {}
for nom in pipelines:
    cv_results[nom] = {
        "cv_f1_mean": round(cv_f1_mean[nom], 4),
        "cv_f1_std": round(cv_f1_std[nom], 4),
        "tuned_cv_f1": round(tuned_cv_f1[nom], 4),
        "best_params": best_params[nom],
        "selected": nom == nom_gagnant,
    }
with open(BASE + "/models/cv_results.json", "w") as fichier:
    json.dump(cv_results, fichier, indent=2, ensure_ascii=False)
print("Resultats de la validation croisee sauvegardes -> models/cv_results.json")

# %% [markdown]
# ### 8.6. Re-entrainement du gagnant sur TOUT le train BRUT
#
# On reconstruit le pipeline gagnant avec ses meilleurs hyperparametres puis on
# le RE-ENTRAINE sur l'integralite du train (`df_train[features]` BRUT, ~1,76 M
# lignes). Pour XGBoost / RandomForest on met `n_jobs=-1` au re-entrainement final.

# %%
# 6. Reconstruction du modele gagnant avec ses best_params + re-entrainement sur tout le train
# best_params est prefixe par l'etape pipeline : on retire le prefixe pour instancier le modele
params_gagnant = {cle.split("__", 1)[1]: valeur for cle, valeur in best_params[nom_gagnant].items()}
print(f"Hyperparametres du gagnant ({nom_gagnant}) : {params_gagnant}")

# Instanciation du modele gagnant avec ses meilleurs hyperparametres
if nom_gagnant == "LinearSVC":
    modele_gagnant = LinearSVC(random_state=42, **params_gagnant)
elif nom_gagnant == "DecisionTree":
    modele_gagnant = DecisionTreeClassifier(random_state=42, **params_gagnant)
elif nom_gagnant == "LogisticRegression":
    modele_gagnant = LogisticRegression(max_iter=1000, random_state=42, **params_gagnant)
elif nom_gagnant == "GaussianNB":
    modele_gagnant = GaussianNB(**params_gagnant)
elif nom_gagnant == "RandomForest":
    modele_gagnant = RandomForestClassifier(n_jobs=-1, random_state=42, **params_gagnant)
elif nom_gagnant == "XGBoost":
    modele_gagnant = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42, **params_gagnant)
else:  # KNN
    modele_gagnant = KNeighborsClassifier(**params_gagnant)

# Pipeline gagnant : StandardScaler + modele gagnant (entree BRUTE)
best_model = make_pipeline(StandardScaler(), modele_gagnant)
best_model.fit(df_train[features], y_train)
print(f"Pipeline gagnant re-entraine sur tout le train : {df_train.shape[0]} lignes")

# %% [markdown]
# ### 8.7. Garantie de predict_proba (CalibratedClassifierCV si besoin)
#
# `best_model.pkl` DOIT exposer `predict_proba` (score de risque). Si le gagnant
# ne l'a pas (LinearSVC), on enveloppe le pipeline gagnant dans
# `CalibratedClassifierCV(..., cv=3)` et on le re-entraine.

# %%
# 7. Si le gagnant n'a pas predict_proba, on le calibre (CalibratedClassifierCV cv=3)
has_proba = hasattr(best_model, "predict_proba")
if not has_proba:
    print(f"{nom_gagnant} n'a pas predict_proba -> calibration (CalibratedClassifierCV cv=3)")
    best_model = CalibratedClassifierCV(make_pipeline(StandardScaler(), modele_gagnant), cv=3)
    best_model.fit(df_train[features], y_train)
    has_proba = hasattr(best_model, "predict_proba")
print(f"Le modele gagnant expose predict_proba : {has_proba}")

# %% [markdown]
# ### 8.8. Sauvegarde du modele gagnant et de ses metadonnees
#
# On sauvegarde `best_model.pkl` (pipeline gagnant) et `best_model_meta.json`
# (nom, best_params, F1 CV moyen/std, has_proba, retrained_on_full).

# %%
# 8. Sauvegarde de best_model.pkl + best_model_meta.json
joblib.dump(best_model, BASE + "/models/best_model.pkl")
print("Modele gagnant sauvegarde -> models/best_model.pkl")

best_model_meta = {
    "name": nom_gagnant,
    "best_params": best_params[nom_gagnant],
    "cv_f1_mean": round(cv_f1_mean[nom_gagnant], 4),
    "cv_f1_std": round(cv_f1_std[nom_gagnant], 4),
    "has_proba": True,
    "retrained_on_full": True,
}
with open(BASE + "/models/best_model_meta.json", "w") as fichier:
    json.dump(best_model_meta, fichier, indent=2, ensure_ascii=False)
print("Metadonnees du gagnant sauvegardees -> models/best_model_meta.json")
print(json.dumps(best_model_meta, indent=2, ensure_ascii=False))

# %% [markdown]
# ### 8.9. Verification rapide du gagnant sur le TEST (features BRUTES)
#
# Convention de prediction : `X = df_test[features]` (BRUT) ;
# `best_model.predict(X)` / `best_model.predict_proba(X)[:, 1]` (= score de risque).
# On affiche accuracy, F1, et la matrice de confusion sur le test.

# %%
# 9. Verification : recharge best_model.pkl puis predit sur le test BRUT
best_model_recharge = joblib.load(BASE + "/models/best_model.pkl")
X_test_brut = df_test[features]                      # features BRUTES (le pipeline re-scale)
y_pred_best = best_model_recharge.predict(X_test_brut)
proba_best = best_model_recharge.predict_proba(X_test_brut)[:, 1]   # score de risque

acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
cm_best = confusion_matrix(y_test, y_pred_best)
print(f"Modele gagnant ({nom_gagnant}) - Accuracy (test) : {acc_best:.4f}")
print(f"Modele gagnant ({nom_gagnant}) - F1 attaque (test) : {f1_best:.4f}")
print(f"Apercu des scores de risque (5 premiers) : {np.round(proba_best[:5], 4)}")
print("\n=== Rapport de classification (modele gagnant) ===")
print(classification_report(y_test, y_pred_best, target_names=["BENIGN", "Attaque"], digits=4))
print("Matrice de confusion (modele gagnant) :")
print(cm_best)

# %% [markdown]
# ## 9. Random Forest "historique" (conserve pour la Phase 4)
#
# La Phase 4 (evaluation) consomme `random_forest.pkl` via le scaler gele. On
# conserve donc ce Random Forest dedie (entree DEJA mise a l'echelle), distinct
# du `best_model.pkl` du bake-off (entree BRUTE). Il garde `class_weight=
# 'balanced'` et `max_depth=20` pour limiter la taille du modele et le
# surapprentissage tout en separant bien les familles d'attaques.

# %%
# 9. Random Forest historique (entree mise a l'echelle) pour la Phase 4
rf = RandomForestClassifier(
    n_estimators=100, max_depth=20, class_weight="balanced",
    n_jobs=-1, random_state=42
)
rf.fit(X_train_s, y_train)

# Prediction sur le test (features mises a l'echelle)
y_pred_rf = rf.predict(X_test_s)

# Metriques
acc_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
fp_rf = int(cm_rf[0, 1])
print(f"Random Forest historique - Accuracy : {acc_rf:.4f}")
print(f"Random Forest historique - Recall (attaque) : {recall_rf:.4f}")
print(f"Random Forest historique - Faux positifs : {fp_rf}")
print("Matrice de confusion (Random Forest historique) :")
print(cm_rf)

# Sauvegarde du modele
joblib.dump(rf, BASE + "/models/random_forest.pkl")
print("Random Forest sauvegarde -> models/random_forest.pkl")

# %% [markdown]
# ## 10. Explicabilite : moyennes des features sur le BENIGN
#
# On sauvegarde les moyennes (echelle BRUTE) de chaque feature sur le BENIGN
# du train. Le dashboard s'en sert pour expliquer "pourquoi cette alerte" en
# comparant la valeur d'une ligne suspecte au comportement normal moyen.

# %%
# 10. Moyennes BENIGN (echelle brute) pour l'explicabilite
benign_means = X_train[y_train == 0].mean()
print("Apercu des moyennes BENIGN (echelle brute) :")
print(benign_means.head())
joblib.dump(benign_means, BASE + "/models/benign_means.pkl")
print("Moyennes BENIGN sauvegardees -> models/benign_means.pkl")

# %% [markdown]
# ## 11. Sauvegarde du jeu de test et d'un echantillon d'application
#
# Le jeu de test complet (features BRUTES + labels) est sauvegarde pour la
# Phase 4 (evaluation). Un echantillon stratifie d'environ 3000 lignes par
# famille d'attaque (`Label_group`) est sauvegarde pour le dashboard.

# %%
# 11a. Sauvegarde du jeu de test complet (features brutes + labels)
colonnes_test = features + ["Label", "Label_binary", "Label_group"]
df_test_out = df_test[colonnes_test]
df_test_out.to_parquet(BASE + "/data/processed/test_set.parquet", index=False)
print(f"Jeu de test sauvegarde -> data/processed/test_set.parquet  (shape {df_test_out.shape})")

# %%
# 11b. Echantillon stratifie ~3000 lignes par famille (min par groupe), random_state=42
n_cible = 3000
n_groupes = df_test_out["Label_group"].nunique()
n_par_groupe = max(1, n_cible // n_groupes)
# On prend au plus n_par_groupe lignes par groupe (sans depasser la taille du groupe)
# Boucle simple sur les familles puis concatenation (conserve toutes les colonnes)
morceaux = []
for groupe, lignes_groupe in df_test_out.groupby("Label_group"):
    n_prendre = min(len(lignes_groupe), n_par_groupe)
    morceaux.append(lignes_groupe.sample(n=n_prendre, random_state=42))
app_sample = pd.concat(morceaux, ignore_index=True)
print(f"Echantillon d'application : {app_sample.shape[0]} lignes")
print(app_sample["Label_group"].value_counts())
app_sample.to_parquet(BASE + "/data/processed/app_sample.parquet", index=False)
print("Echantillon sauvegarde -> data/processed/app_sample.parquet")

# %% [markdown]
# ## 12. Recapitulatif du bake-off et du gagnant
#
# Tableau de comparaison des 7 modeles (F1 CV moyen +/- std et F1 CV affine),
# puis rappel du gagnant et de ses performances sur le test.

# %%
# 12. Recapitulatif : tableau de comparaison des 7 modeles + gagnant
print("=== Bake-off : comparaison des 7 modeles (validation croisee 10-fold, F1) ===")
print(f"{'Modele':20s} {'F1 CV':>10s} {'+/- std':>10s} {'F1 tune':>10s} {'gagnant':>9s}")
for nom in pipelines:
    marque = "  <===" if nom == nom_gagnant else ""
    print(f"{nom:20s} {cv_f1_mean[nom]:>10.4f} {cv_f1_std[nom]:>10.4f} {tuned_cv_f1[nom]:>10.4f} {marque:>9s}")

print(f"\nModele gagnant : {nom_gagnant}")
print(f"F1 CV (sous-echantillon) : {cv_f1_mean[nom_gagnant]:.4f} +/- {cv_f1_std[nom_gagnant]:.4f}")
print(f"F1 CV affine (GridSearch) : {tuned_cv_f1[nom_gagnant]:.4f}")
print(f"Accuracy sur le test : {acc_best:.4f}")
print(f"F1 (attaque) sur le test : {f1_best:.4f}")

# %% [markdown]
# ## 13. Verification du contrat d'artefacts
#
# On verifie que TOUS les fichiers attendus (existants conserves + nouveaux du
# bake-off) existent.

# %%
# 13. Verification de l'existence de tous les artefacts du contrat
artefacts = [
    # Artefacts existants conserves
    BASE + "/models/features.pkl",
    BASE + "/models/scaler.pkl",
    BASE + "/models/benign_means.pkl",
    BASE + "/models/isolation_forest.pkl",
    BASE + "/models/rule_baseline.json",
    BASE + "/models/random_forest.pkl",
    BASE + "/data/processed/test_set.parquet",
    BASE + "/data/processed/app_sample.parquet",
    # Nouveaux artefacts du bake-off
    BASE + "/models/best_model.pkl",
    BASE + "/models/best_model_meta.json",
    BASE + "/models/cv_results.json",
]
for chemin in artefacts:
    existe = os.path.exists(chemin)
    taille = os.path.getsize(chemin) / 1024**2 if existe else 0.0
    print(f"{'OK ' if existe else 'MANQUANT '} {chemin}  ({taille:.4f} Mo)")

tous_presents = all(os.path.exists(c) for c in artefacts)
print(f"\nTous les artefacts du contrat sont presents : {tous_presents}")

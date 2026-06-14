# Dashboard SOC — Détection explicable d'anomalies (CICIDS2017)
# Phase 5 : application Streamlit pour l'analyste.
# Style notebook : code plat, commentaires en français, étapes numérotées.
# Exception autorisée : quelques fonctions minimales pour le cache de chargement.

# Imports
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Chemins absolus des artefacts (contrat de la Phase 3)
BASE = "/home/samouraifox/Work/stuff/S6/Notes_Ai/projet-final/model-building/Detection-explicable-danomalies"
SCALER_PATH = BASE + "/models/scaler.pkl"
BEST_MODEL_PATH = BASE + "/models/best_model.pkl"
BEST_META_PATH = BASE + "/models/best_model_meta.json"
ISO_PATH = BASE + "/models/isolation_forest.pkl"
FEATURES_PATH = BASE + "/models/features.pkl"
BENIGN_MEANS_PATH = BASE + "/models/benign_means.pkl"
RULES_PATH = BASE + "/models/rule_baseline.json"
SAMPLE_PATH = BASE + "/data/processed/app_sample.parquet"
FEEDBACK_PATH = BASE + "/app/feedback.csv"


# 1. Chargement (en cache) des modèles et artefacts
@st.cache_resource
def charger_modeles():
    # Modèle gagnant (pipeline scaler interne) + scaler baseline iso + métadonnées
    scaler = joblib.load(SCALER_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)
    with open(BEST_META_PATH, "r") as f:
        best_meta = json.load(f)
    iso = joblib.load(ISO_PATH)
    features = joblib.load(FEATURES_PATH)
    benign_means = joblib.load(BENIGN_MEANS_PATH)
    with open(RULES_PATH, "r") as f:
        rules = json.load(f)
    return scaler, best_model, best_meta, iso, features, benign_means, rules


@st.cache_data
def charger_echantillon():
    # Échantillon stratifié du test (features brutes + labels)
    df = pd.read_parquet(SAMPLE_PATH)
    return df


# Petite fonction utilitaire : verdict des règles statiques sur une ligne BRUTE
def verdict_regles(row, rules):
    # Logique OU : une seule règle déclenchée suffit pour flagger en attaque
    detail = []
    declenche = False
    # 2. Parcours des règles décrites dans rule_baseline.json
    for r in rules["regles"]:
        if "feature" in r:
            # Règle simple : feature OP seuil
            valeur = row[r["feature"]]
            if r["operateur"] == ">":
                ok = valeur > r["seuil"]
            else:
                ok = valeur < r["seuil"]
            nom = f"{r['feature']} {r['operateur']} {r['seuil']:.0f}"
        else:
            # Règle composée : feature_1 OP seuil_1 ET feature_2 OP seuil_2
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
            nom = (
                f"{r['feature_1']} {r['operateur_1']} {r['seuil_1']:.0f} ET "
                f"{r['feature_2']} {r['operateur_2']} {r['seuil_2']:.0f}"
            )
        if ok:
            declenche = True
        detail.append({"Règle": nom, "Déclenchée": "Oui" if ok else "Non"})
    return declenche, detail


# Chargement effectif
scaler, best_model, best_meta, iso, features, benign_means, rules = charger_modeles()
df = charger_echantillon()

# Nom du modèle gagnant (issu du bake-off de la Phase 3)
nom_modele = best_meta["name"]

# Configuration de la page
st.set_page_config(page_title="SOC — Détection d'anomalies CICIDS2017", layout="wide")

# 3. Titre et courte introduction
st.title("Détection explicable d'anomalies réseau — Tableau de bord SOC")
st.markdown(
    f"Cet outil aide l'analyste à **trier les flux réseau** du jeu CICIDS2017. "
    f"Pour chaque flux on compare trois approches (règles statiques, Isolation Forest, "
    f"**{nom_modele}** retenu en Phase 3), on affiche un **score de risque** et on "
    f"explique **pourquoi** une alerte se déclenche. Les vrais labels sont montrés à "
    f"titre indicatif seulement."
)
# Modèle retenu par le bake-off (F1 de validation croisée)
st.success(
    f"Modèle retenu : **{nom_modele}** "
    f"(F1 CV = {best_meta['cv_f1_mean']:.4f} ± {best_meta['cv_f1_std']:.4f})"
)

# 4. Barre latérale : sélection du flux à inspecter
st.sidebar.header("Sélection du flux")

# Filtre optionnel par famille (Label_group)
familles = ["Toutes"] + sorted(df["Label_group"].unique().tolist())
famille = st.sidebar.selectbox("Filtrer par famille (Label_group)", familles)

# Sous-ensemble des index disponibles selon le filtre
if famille == "Toutes":
    index_dispo = df.index.tolist()
else:
    index_dispo = df[df["Label_group"] == famille].index.tolist()

# Initialisation de l'index courant en mémoire de session
if "flux_idx" not in st.session_state:
    st.session_state.flux_idx = index_dispo[0]

# Bouton flux aléatoire : tire un index dans le sous-ensemble filtré
if st.sidebar.button("Flux aléatoire"):
    st.session_state.flux_idx = int(np.random.choice(index_dispo))

# Si l'index courant n'est plus dans le filtre, on le ramène au premier dispo
if st.session_state.flux_idx not in index_dispo:
    st.session_state.flux_idx = index_dispo[0]

# Sélecteur de l'index du flux dans le sous-ensemble filtré
flux_idx = st.sidebar.selectbox(
    "Index du flux à inspecter",
    index_dispo,
    index=index_dispo.index(st.session_state.flux_idx),
)
st.session_state.flux_idx = flux_idx

st.sidebar.caption(f"{len(index_dispo)} flux disponibles dans la sélection.")

# 5. Calculs pour le flux choisi
# Ligne brute sélectionnée
row = df.loc[flux_idx]

# Features BRUTES : le gagnant est un pipeline qui scale en interne
X = df.loc[[flux_idx], features]

# Verdict du modèle gagnant + score de risque (probabilité d'attaque en %)
# On passe les features BRUTES, surtout PAS scaler.transform avant
best_pred = int(best_model.predict(X)[0])  # 0 = BENIGN, 1 = attaque
score_risque = float(best_model.predict_proba(X)[:, 1][0]) * 100.0

# Baseline Isolation Forest : elle, utilise le scaler externe sur le BRUT
Xs = scaler.transform(X)
# Score Isolation Forest : -1 = anomalie -> 1, 1 = normal -> 0
iso_raw = int(iso.predict(Xs)[0])
iso_flag = 1 if iso_raw == -1 else 0
# Score d'anomalie continu (plus c'est négatif, plus c'est anormal)
iso_score = float(iso.score_samples(Xs)[0])

# Verdict des règles statiques (sur features BRUTES)
regle_flag, regle_detail = verdict_regles(row, rules)

# Vraie famille du flux (à titre indicatif)
vraie_famille = row["Label_group"]
vrai_label = row["Label"]

# 6. Panneau VERDICT
st.header(f"Verdict du modèle principal ({nom_modele})")
col_a, col_b, col_c = st.columns([2, 2, 2])

with col_a:
    # Badge clair vert/rouge selon le verdict du gagnant
    if best_pred == 1:
        st.error("ANOMALIE — flux suspect")
    else:
        st.success("NORMAL — flux bénin")

with col_b:
    # Score de risque affiché en métrique + barre de progression
    st.metric("Score de risque (proba attaque)", f"{score_risque:.2f} %")
    st.progress(min(int(round(score_risque)), 100))

with col_c:
    # Vraie famille à titre indicatif (ne sert pas à la décision)
    st.metric("Vraie famille (indicatif)", vraie_famille)
    st.caption(f"Label détaillé : {vrai_label}")

# 7. Panneau COMPARAISON des 3 approches
st.header("Comparaison des trois approches")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Règles statiques")
    if regle_flag:
        st.error("ANOMALIE")
    else:
        st.success("NORMAL")
    st.caption("Logique OU sur seuils bruts (baseline simpliste).")

with col2:
    st.subheader("Isolation Forest")
    if iso_flag == 1:
        st.error("ANOMALIE")
    else:
        st.success("NORMAL")
    st.caption(f"Score d'anomalie : {iso_score:.4f} (plus bas = plus anormal).")

with col3:
    st.subheader(nom_modele)
    if best_pred == 1:
        st.error("ANOMALIE")
    else:
        st.success("NORMAL")
    st.caption(f"Score de risque : {score_risque:.2f} %.")

# Détail des règles déclenchées
with st.expander("Détail des règles statiques"):
    df_regles = pd.DataFrame(regle_detail)
    st.dataframe(df_regles, hide_index=True, use_container_width=True)

# 8. Panneau POURQUOI CETTE ALERTE
st.header("Pourquoi cette alerte ? (explicabilité)")
st.markdown(
    f"On affiche les features les plus importantes pour **{nom_modele}**, en comparant "
    f"la **valeur du flux** à la **moyenne du trafic BENIGN** d'entraînement. "
    f"Les features dont la valeur s'écarte fortement de la moyenne bénigne sont surlignées."
)

# Nombre de features à expliquer (curseur)
top_n = st.slider("Nombre de features à expliquer", 5, 20, 10)

# Importances du gagnant : on récupère le modèle final (dernier step du pipeline)
modele_final = list(best_model.named_steps.values())[-1]
if hasattr(modele_final, "feature_importances_"):
    # Arbres / gradient boosting : importances natives
    poids = np.asarray(modele_final.feature_importances_)
    libelle_importance = "Importance"
elif hasattr(modele_final, "coef_"):
    # Modèle linéaire : on prend la valeur absolue des coefficients
    poids = np.abs(np.ravel(modele_final.coef_))
    libelle_importance = "Importance (|coef|)"
else:
    # Repli : importances uniformes (aucune info de poids exploitable)
    poids = np.ones(len(features))
    libelle_importance = "Importance (uniforme)"

# Importances triées, on garde le top N
importances = pd.Series(poids, index=features)
top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

# Construction du tableau de comparaison valeur flux vs moyenne BENIGN
lignes = []
for f in top_features:
    val = float(row[f])
    moy = float(benign_means[f])
    # Écart relatif (en multiples de la moyenne bénigne), évite la division par zéro
    if moy != 0:
        ecart = (val - moy) / abs(moy)
    else:
        ecart = 0.0 if val == 0 else np.inf
    # On marque comme anormal un écart de plus de 100 % par rapport au BENIGN
    anormal = abs(ecart) > 1.0
    lignes.append(
        {
            "Feature": f,
            libelle_importance: float(importances[f]),
            "Valeur du flux": val,
            "Moyenne BENIGN": moy,
            "Écart relatif": ecart,
            "Anormal": "Oui" if anormal else "Non",
        }
    )

df_explain = pd.DataFrame(lignes)


# Surlignage des lignes anormales (fond rouge clair)
def surligner_anormal(ligne):
    if ligne["Anormal"] == "Oui":
        return ["background-color: #ffcccc"] * len(ligne)
    return [""] * len(ligne)


# Tableau stylé avec formatage numérique
style_explain = (
    df_explain.style.apply(surligner_anormal, axis=1).format(
        {
            libelle_importance: "{:.4f}",
            "Valeur du flux": "{:.2f}",
            "Moyenne BENIGN": "{:.2f}",
            "Écart relatif": "{:+.2f}",
        }
    )
)
st.dataframe(style_explain, hide_index=True, use_container_width=True)

# Graphique en barres : comparaison valeur flux vs moyenne BENIGN
st.subheader("Comparaison graphique : flux vs moyenne BENIGN")
df_bar = df_explain.set_index("Feature")[["Valeur du flux", "Moyenne BENIGN"]]
st.bar_chart(df_bar)

# 9. Panneau RETOUR ANALYSTE
st.header("Retour de l'analyste")
st.markdown(
    f"Confirmez le verdict de **{nom_modele}** pour ce flux. Le retour est enregistré "
    f"dans `app/feedback.csv` et pourra servir à améliorer le modèle."
)

col_vp, col_fp = st.columns(2)
clic_vp = col_vp.button("Vrai positif (alerte confirmée)")
clic_fp = col_fp.button("Faux positif (fausse alerte)")

# Enregistrement du retour si l'un des deux boutons est cliqué
if clic_vp or clic_fp:
    retour = "Vrai positif" if clic_vp else "Faux positif"
    # Ligne à ajouter au journal de feedback
    nouvelle_ligne = pd.DataFrame(
        [
            {
                "index_flux": int(flux_idx),
                "verdict_modele": "ANOMALIE" if best_pred == 1 else "NORMAL",
                "score_risque": round(score_risque, 4),
                "retour_analyste": retour,
            }
        ]
    )
    # Création de l'en-tête si le fichier n'existe pas encore, sinon ajout
    if os.path.exists(FEEDBACK_PATH):
        nouvelle_ligne.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        nouvelle_ligne.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)
    st.success(
        f"Retour enregistré : flux {flux_idx} -> {retour} "
        f"(verdict {nom_modele} : {'ANOMALIE' if best_pred == 1 else 'NORMAL'}, "
        f"score {score_risque:.2f} %)."
    )

# Affichage du journal de feedback existant
if os.path.exists(FEEDBACK_PATH):
    with st.expander("Voir le journal des retours enregistrés"):
        df_feedback = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(df_feedback, hide_index=True, use_container_width=True)

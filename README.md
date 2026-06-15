# Détection explicable d'anomalies (CICIDS2017)

Détection explicable d'anomalies dans les logs d'infrastructure bancaire par apprentissage
supervisé — mini-projet IA S6.

*Explainable anomaly detection in network/banking infrastructure logs.*

- **Dataset :** CICIDS2017 (8 fichiers CSV, ~2,8 M flux réseau).
- **Modèles :** comparaison (bake-off) de **7 classifieurs supervisés** (LinearSVC, Decision Tree,
  Logistic Regression, Naive Bayes, Random Forest, XGBoost, KNN) par **validation croisée 10-fold**.
  On affiche **toutes les métriques** (accuracy, précision, recall, F1) et **une matrice de confusion
  pour chaque modèle**, puis on **garde le meilleur** (sélection sur le F1) : **XGBoost**. Une
  baseline de règles statiques sert de **référence** pour la réduction des faux positifs.
- **Interface :** dashboard Streamlit (verdict, score de risque, top features « pourquoi cette
  alerte », comparaison avec la baseline de règles, bouton de retour analyste).
- **Défendabilité :** INT-001 (détection d'anomalies réseau sur dataset public),
  INT-003 (réduction des faux positifs SOC + explication des features), INT-004 (structure de démo).

---

## Résultats (jeu de test : 756 240 flux)

Modèle gardé : **XGBoost** (`max_depth=8, learning_rate=0.1, n_estimators=300`), sélectionné sur le
F1 de validation croisée (10-fold) puis ré-entraîné sur tout le train (1,76 M lignes).

| Modèle | Recall (attaque) | Précision | F1 | Accuracy | Faux positifs | Réduction FP vs règles |
|---|---|---|---|---|---|---|
| Règles statiques | 0.6142 | 0.4455 | 0.5164 | 0.8057 | 97 654 | — (baseline) |
| **XGBoost (gardé)** | **0.9987** | **0.9966** | **0.9976** | **0.9992** | **438** | **−99.55 %** |

Comparaison du bake-off (F1 de validation croisée 10-fold, 7 modèles) :
XGBoost 0.9960 > Random Forest 0.9937 > Decision Tree 0.9928 > KNN 0.9643 >
Logistic Regression 0.8694 > LinearSVC 0.8509 > Naive Bayes 0.3616.

> ⚠️ **Pourquoi sélectionner sur le F1 et pas sur le recall seul ?** Naive Bayes atteint un recall de
> 0.9942 (il attrape presque toutes les attaques) mais une précision de seulement 0.218 (≈ 78 % de
> ses alertes sont fausses). Sélectionner sur le recall seul favoriserait ce genre de modèle « qui
> crie au loup ». Le F1 équilibre recall et précision : c'est exactement l'objectif, détecter les
> attaques **et** limiter les fausses alertes.

Toutes les métriques par modèle + matrices de confusion : `models/cv_results.json`,
`reports/metrics.json`, et les figures dans `reports/figures/` (dont `12_confusion_matrices.png`,
une matrice de confusion par modèle).

---

## Installation

L'environnement de développement utilise Python 3.14 (pandas 3.0, scikit-learn 1.8, streamlit 1.56,
xgboost 3.2).

```bash
cd Detection-explicable-danomalies
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Les 8 CSV CICIDS2017 ne sont **pas** versionnés (cf. `.gitignore`). Les placer dans `data/raw/` :

```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv               (BENIGN)
├── Tuesday-WorkingHours.pcap_ISCX.csv              (FTP/SSH-Patator)
├── Wednesday-workingHours.pcap_ISCX.csv            (DoS*, Heartbleed)
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv       (Bot)
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Source utilisée : [Kaggle — Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset).

## Pipeline (notebooks, à exécuter dans l'ordre)

Chaque notebook existe en double : une source `.py` (format jupytext « percent ») et le `.ipynb`
exécuté. Ils produisent leurs artefacts dans `data/processed/`, `models/` et `reports/`.

| Notebook | Rôle | Produit |
|---|---|---|
| `01_nettoyage` | nettoyage + consolidation des 8 CSV | `data/processed/cicids_clean.parquet` |
| `02_eda` | analyse exploratoire | `reports/figures/01…05.png` |
| `03_modelisation` | prep + baseline de règles + **bake-off 7 modèles** (CV 10-fold, toutes les métriques + une matrice de confusion par modèle) → meilleur gardé | `models/best_model.pkl`, `cv_results.json`, `scaler`/`features`/`benign_means.pkl`, `rule_baseline.json`, `test_set`/`app_sample` |
| `04_evaluation` | métriques sur le test, comparaison gagnant vs règles, explicabilité | `reports/metrics.json`, `figures/06,08…12.png` |

```bash
# Ouvrir dans Jupyter et exécuter 01 → 04
jupyter lab

# …ou ré-exécuter en ligne de commande
jupyter nbconvert --to notebook --execute --inplace notebooks/01_nettoyage.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelisation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

> ⚠️ Les notebooks et `app/app.py` utilisent un chemin absolu via la constante `BASE` définie en
> haut de chaque fichier. **Si vous déplacez le dépôt, mettez à jour `BASE`** dans les 4 notebooks
> et dans `app/app.py`.

## Lancer le dashboard

```bash
streamlit run app/app.py
```

Le dashboard charge `best_model.pkl` (le modèle gardé, affiché par son nom) et
`data/processed/app_sample.parquet` : sélection d'un flux, verdict + score de risque du modèle gardé,
comparaison des deux approches (modèle gardé vs règles statiques), panneau « pourquoi cette alerte »
(valeur du flux vs moyenne BENIGN) et bouton de retour analyste (écrit dans `app/feedback.csv`).

## Structure

```
data/raw/            # 8 CSV CICIDS2017 (non versionnés)
data/processed/      # dataset gelé + jeux de test/échantillon (parquet)
notebooks/           # 01 → 04 (.py jupytext + .ipynb exécutés)
models/              # best_model.pkl (gardé) + cv_results.json + scaler + features + benign_means + rule_baseline.json
reports/figures/     # figures PNG (dont 11_bakeoff_cv_f1.png et 12_confusion_matrices.png)
reports/metrics.json # métriques : bake-off CV (7 modèles, toutes métriques) + gagnant vs règles
app/app.py           # dashboard Streamlit
CLAUDE.md            # directives de style de code (cours « Python - Partie 2 »)
plan.md              # feuille de route + avancement
```

## Style de code

Tout le code des notebooks suit le style du cours « Python - Partie 2 » : Python plat / procédural
type notebook, alias `pd`/`np`/`plt`/`sns`, **noms de variables, prints et titres de graphiques en
anglais**, peu de commentaires (courts), pas de fonctions/classes/type hints dans les notebooks,
`random_state=42`. Voir `CLAUDE.md`.

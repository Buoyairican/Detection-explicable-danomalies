# Détection explicable d'anomalies (CICIDS2017)

Détection explicable d'anomalies dans les logs d'infrastructure bancaire par apprentissage du
comportement normal — mini-projet IA S6.

*Explainable anomaly detection in network/banking infrastructure logs using a behavioral baseline.*

- **Dataset :** CICIDS2017 (8 fichiers CSV, ~2,8 M flux réseau).
- **Modèles :** bake-off de **7 classifieurs supervisés** (LinearSVC, Decision Tree, Logistic
  Regression, Naive Bayes, Random Forest, XGBoost, KNN) comparés par validation croisée — le meilleur
  est **gardé** (**XGBoost**). L'Isolation Forest (apprentissage du comportement normal) et une
  baseline de règles statiques servent de **références**.
- **Interface :** dashboard Streamlit (verdict, score de risque, top features « pourquoi cette
  alerte », comparaison avec la baseline de règles, bouton de retour analyste).
- **Défendabilité :** INT-001 (détection d'anomalies réseau sur dataset public),
  INT-003 (réduction des faux positifs SOC + explication des features), INT-004 (structure de démo).

---

## Résultats (jeu de test : 756 240 flux)

Modèle gardé par le bake-off : **XGBoost** (`max_depth=8, learning_rate=0.1, n_estimators=300`),
sélectionné sur le F1 de validation croisée (10-fold) puis ré-entraîné sur tout le train (1,76 M lignes).

| Modèle | Recall (attaque) | Accuracy | Faux positifs | Taux FP | Réduction FP vs règles |
|---|---|---|---|---|---|
| Règles statiques | 0.6142 | 0.8057 | 97 654 | 15.5 % | — (baseline) |
| Isolation Forest | 0.3994 | 0.8572 | 31 274 | 4.98 % | **−68.0 %** |
| **XGBoost (gardé)** | **0.9987** | **0.9992** | **438** | **0.07 %** | **−99.55 %** |

Comparaison du bake-off (F1 CV, 7 modèles) : XGBoost 0.9964 > Random Forest 0.9949 > Decision Tree
0.9937 > KNN 0.9739 > Logistic Regression 0.8713 > LinearSVC 0.8588 > GaussianNB 0.4264.
Détail dans `reports/metrics.json` et `models/cv_results.json` ; figures dans `reports/figures/`.

---

## Installation

L'environnement de développement utilise Python 3.14 (pandas 3.0, scikit-learn 1.8, streamlit 1.56).

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
| `03_modelisation` | prep + baselines (règles, Isolation Forest) + **bake-off 7 modèles** (CV + GridSearch) → meilleur gardé | `models/best_model.pkl`, `cv_results.json`, `*.pkl/.json`, `test_set` / `app_sample` |
| `04_evaluation` | métriques, comparaison gagnant vs baselines, explicabilité | `reports/metrics.json`, `figures/06…11.png` |

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
comparaison des 3 approches (modèle gardé vs Isolation Forest vs règles), panneau « pourquoi cette
alerte » (valeur du flux vs moyenne BENIGN) et bouton de retour analyste (écrit dans `app/feedback.csv`).

## Structure

```
data/raw/            # 8 CSV CICIDS2017 (non versionnés)
data/processed/      # dataset gelé + jeux de test/échantillon (parquet)
notebooks/           # 01 → 04 (.py jupytext + .ipynb exécutés)
models/              # best_model.pkl (gardé) + cv_results.json + isolation_forest + scaler + features + benign_means + rule_baseline.json
reports/figures/     # 11 figures PNG (titres FR, dont 11_bakeoff_cv_f1.png)
reports/metrics.json # métriques : bake-off CV + modèle gardé vs baselines
app/app.py           # dashboard Streamlit
CLAUDE.md            # directives de style de code (cours « Python - Partie 2 »)
plan.md              # feuille de route + avancement
```

## Style de code

Tout le code suit le style du cours « Python - Partie 2 » : Python plat / procédural type notebook,
alias `pd`/`np`/`plt`/`sns`, commentaires français en étapes numérotées, pas de
fonctions/classes/type hints dans les notebooks, `random_state=42`. Voir `CLAUDE.md`.

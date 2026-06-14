# Mini-projet IA — Détection explicable d'anomalies dans les logs d'infrastructure bancaire

**Dataset:** CICIDS2017
**Modèles:** Bake-off de 7 classifieurs (CV + GridSearch) → meilleur gardé : **XGBoost** ; Isolation Forest + règles statiques en référence
**Interface:** Streamlit

---

## Phase 0 — Setup
- [x] Créer le repo Git + dossiers (`/data`, `/notebooks`, `/models`, `/app`)
- [x] Créer l'environnement Python (venv + pandas, scikit-learn, streamlit, matplotlib/seaborn, joblib)
- [ ] Répartir les tâches entre les trois membres

## Phase 1 — Récupérer et comprendre les données
- [x] Télécharger CICIDS2017 (8 fichiers CSV depuis le site du CIC)
- [x] Identifier quel jour = quelle attaque, et le sens des labels
- [x] Charger et inspecter (shapes, colonnes, distribution des labels)

## Phase 2 — Nettoyage (le vrai travail)
- [x] Corriger les noms de colonnes (espaces en début de nom — piège connu)
- [x] Gérer les `Infinity` / `NaN` dans les colonnes de flux (autre piège connu)
- [x] Fusionner les 8 fichiers ou choisir un sous-ensemble ciblé
- [x] Consolider les labels (binaire normal/attaque + multiclasse regroupée)
- [x] Choisir la stratégie pour le déséquilibre des classes (class_weight / undersample / SMOTE)
- [x] Train/test split + scaling

## Phase 3 — EDA
- [x] Distributions, corrélations, top features par attaque
- [x] Sauvegarder les figures propres pour le rapport

## Phase 4 — Baseline de règles statiques
- [x] Écrire des règles simples par seuils sur quelques features
- [x] Mesurer son taux de faux positifs (pour comparer avec les 60–80 % de INT-003)

## Phase 5 — Isolation Forest (apprentissage du comportement normal)
- [x] Entraîner sur le trafic normal uniquement
- [x] Régler `contamination` et `n_estimators`
- [x] Évaluer (precision / recall / FP)

## Phase 6 — Random Forest (explicabilité)
- [x] Entraîner le classifieur supervisé
- [x] Régler les hyperparamètres
- [x] Extraire l'importance globale des features
- [x] Construire les top features par prédiction (le "pourquoi cette alerte")

## Phase 7 — Évaluation et comparaison
- [x] Matrices de confusion + tableau de métriques
- [x] Comparer les trois : règles vs Isolation Forest vs Random Forest
- [x] Positionner les résultats face à la cible de 60–80 % de FP

## Phase 8 — Dashboard Streamlit
- [x] Layout + chargement des modèles sauvegardés
- [x] Afficher le verdict + le score de risque
- [x] Afficher les top features
- [x] Ajouter le panneau de comparaison avec la baseline de règles
- [x] Ajouter le bouton de retour analyste (sauvegarde FP/TP dans un CSV)
- [x] Polish

## Phase 9 — Rapport final
- [ ] Rédiger méthodologie + résultats
- [ ] Intégrer les insights / verbatims des entretiens
- [ ] Figures + mise en forme

## Phase 10 — Soutenance
- [ ] Slides
- [ ] Répétition

---

> ⚠️ La Phase 2 bloque tout le reste. Geler le dataset nettoyé avant de commencer la modélisation, sinon le travail est refait deux fois.

---

## Avancement — build du 2026-06-14

**État : Phases 1 à 8 terminées et vérifiées**, puis **pivot vers un bake-off de modèles** : au lieu
d'un Random Forest figé, on entraîne et compare **7 classifieurs supervisés** par validation croisée
et on **garde le meilleur** (XGBoost). L'Isolation Forest et la baseline de règles sont conservés
comme **références**. Système complet et fonctionnel (notebooks exécutés + modèle gardé sauvegardé +
dashboard Streamlit testé). Restent les Phases 9–10 (rapport écrit et soutenance).

### Bake-off — comparaison par validation croisée (F1, 10-fold, sous-échantillon stratifié 150K)

| Modèle | F1 CV (tuné) | Meilleurs hyperparamètres |
|---|---|---|
| **XGBoost** ⭐ gardé | **0.9964 ± 0.0005** | `max_depth=8, learning_rate=0.1, n_estimators=300` |
| Random Forest | 0.9949 | `max_depth=None, n_estimators=300` |
| Decision Tree | 0.9937 | `max_depth=20` |
| KNN | 0.9739 | `n_neighbors=1` |
| Logistic Regression | 0.8713 | `C=20` |
| LinearSVC | 0.8588 | `C=3` |
| GaussianNB | 0.4264 | `var_smoothing=1e-4` |

Sélection : chaque modèle = `make_pipeline(StandardScaler, modèle)` ; `cross_val_score` puis
`GridSearchCV` en `StratifiedKFold(10)`, `scoring='f1'`, sur un sous-échantillon stratifié de
**150 000** lignes du train ; le gagnant (XGBoost) est **ré-entraîné sur l'intégralité du train
(1 764 558 lignes)** puis évalué sur le jeu de test tenu à l'écart (756 240 lignes).

### Résultats sur le jeu de test (756 240 flux : 628 518 BENIGN / 127 722 attaques)

| Modèle | Précision | Recall | F1 | Accuracy | Faux positifs | Taux FP |
|---|---|---|---|---|---|---|
| Règles statiques | 0.4455 | 0.6142 | 0.5164 | 0.8057 | 97 654 | 15.54 % |
| Isolation Forest | 0.6199 | 0.3994 | 0.4858 | 0.8572 | 31 274 | 4.98 % |
| **XGBoost (gardé)** | **0.9966** | **0.9987** | **0.9976** | **0.9992** | **438** | **0.07 %** |

- **Réduction des faux positifs vs baseline de règles** : XGBoost **−99.55 %** (438 vs 97 654),
  Isolation Forest **−68.0 %** (31 274 vs 97 654).
- **Cible INT-003 (60–80 % de réduction des FP)** : l'Isolation Forest tombe **dans** la fourchette ;
  XGBoost la **dépasse** largement (dépassement favorable).
- **Explicabilité** : importances globales de XGBoost (`feature_importances_`) + explication par
  alerte (valeur du flux vs moyenne du trafic BENIGN). Voir `reports/figures/09`, `10`, `11`.
- **Plafond du dataset confirmé** : grilles élargies ~3× → F1 du sommet inchangé (XGBoost 0.9964,
  RF 0.9949) ⇒ ~0.996 est un plafond des données, pas un artefact de grille tronquée. Le Decision
  Tree, lui, a réellement progressé (0.989 → 0.9937).

### Choix techniques retenus
- Nettoyage : strip des noms de colonnes, normalisation des labels `Web Attack` (mojibake),
  `±Infinity → NaN → dropna`, suppression des doublons, downcast float32 → 2 520 798 lignes gelées.
- Features : 78 candidates → **8 colonnes constantes supprimées** → **70 features**.
- Déséquilibre : split **stratifié** sur `Label_binary` (pas de SMOTE).
- Panel : `LinearSVC, DecisionTree, LogisticRegression, GaussianNB, RandomForest, XGBoost, KNN`,
  chacun dans un pipeline avec `StandardScaler`.
- Isolation Forest : `contamination=0.05`, entraîné sur le **BENIGN du train uniquement** (référence).

### Artefacts produits
- `data/processed/cicids_clean.parquet` (dataset gelé), `test_set.parquet`, `app_sample.parquet`
- `models/` : **`best_model.pkl`** (pipeline XGBoost gardé), `best_model_meta.json`, `cv_results.json`,
  `isolation_forest.pkl`, `scaler.pkl`, `features.pkl`, `benign_means.pkl`, `rule_baseline.json`
- `reports/metrics.json` + `reports/figures/01…11.png` (dont `11_bakeoff_cv_f1.png`)
- `notebooks/01_nettoyage` → `04_evaluation` (sources `.py` jupytext + `.ipynb` exécutés)
- `app/app.py` (dashboard Streamlit, charge `best_model.pkl` et affiche le modèle gardé)

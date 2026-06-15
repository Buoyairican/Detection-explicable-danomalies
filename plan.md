# Mini-projet IA — Détection explicable d'anomalies dans les logs d'infrastructure bancaire

**Dataset:** CICIDS2017
**Modèles:** Bake-off de 7 classifieurs comparés par validation croisée 10-fold (toutes les métriques + une matrice de confusion par modèle) → meilleur gardé : **XGBoost** ; baseline de règles statiques en référence
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

## Phase 5 — Isolation Forest (apprentissage du comportement normal) — _retiré dans la version finale_
- [x] ~~Entraîner sur le trafic normal uniquement~~
- [x] ~~Régler `contamination` et `n_estimators`~~
- [x] ~~Évaluer (precision / recall / FP)~~
> Remplacé par le bake-off supervisé (voir Avancement). L'Isolation Forest n'est plus dans le projet.

## Phase 6 — Modèles supervisés (bake-off + explicabilité)
- [x] Entraîner et comparer les 7 classifieurs (validation croisée 10-fold)
- [x] Afficher toutes les métriques + une matrice de confusion par modèle
- [x] Extraire l'importance globale des features du modèle gardé
- [x] Construire les top features par prédiction (le "pourquoi cette alerte")

## Phase 7 — Évaluation et comparaison
- [x] Matrices de confusion + tableau de métriques
- [x] Comparer le modèle gardé (XGBoost) vs la baseline de règles
- [x] Mesurer la réduction des faux positifs vs la baseline de règles

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

## Avancement — build du 2026-06-14 (mis à jour 2026-06-15)

**État : Phases 1 à 8 terminées et vérifiées.** Le cœur du projet est un **bake-off de 7 classifieurs
supervisés** comparés par **validation croisée 10-fold** : pour chaque modèle on affiche **toutes les
métriques** (accuracy, précision, recall, F1) et **une matrice de confusion**, puis on **garde le
meilleur** (sélection sur le F1) : **XGBoost**. Une **baseline de règles statiques** sert de
**référence** pour la réduction des faux positifs. (L'Isolation Forest de la première version a été
retiré ; le code a aussi été simplifié au niveau « débutant » : plus de GridSearchCV ni de pipelines,
juste un dict de modèles + `cross_val_score` + fit/predict + tableau de métriques.) Système complet
et fonctionnel (notebooks exécutés + modèle gardé sauvegardé + dashboard Streamlit testé). Restent
les Phases 9–10 (rapport écrit et soutenance).

### Bake-off — comparaison par validation croisée (F1, 10-fold, sous-échantillon stratifié 150K)

| Modèle | F1 CV | Recall (test*) | Précision (test*) | F1 (test*) |
|---|---|---|---|---|
| **XGBoost** ⭐ gardé | **0.9960 ± 0.0011** | 0.9976 | 0.9944 | 0.9960 |
| Random Forest | 0.9937 | 0.9954 | 0.9949 | 0.9951 |
| Decision Tree | 0.9928 | 0.9967 | 0.9927 | 0.9947 |
| KNN | 0.9643 | 0.9649 | 0.9655 | 0.9652 |
| Logistic Regression | 0.8694 | 0.8621 | 0.8817 | 0.8718 |
| LinearSVC | 0.8509 | 0.8351 | 0.8712 | 0.8528 |
| Naive Bayes | 0.3616 | 0.9942 | 0.2181 | 0.3578 |

\*test = jeu de test du bake-off (sous-échantillon). Le gagnant est ré-évalué sur le test complet ci-dessous.

**Pourquoi sélectionner sur le F1 et pas sur le recall seul :** Naive Bayes a un recall de 0.9942
mais une précision de 0.218 (≈ 78 % de fausses alertes) → sélectionner sur le recall seul
favoriserait ce genre de modèle « qui crie au loup ». Le F1 équilibre recall (attraper les attaques)
et précision (limiter les fausses alertes), ce qui correspond exactement à l'objectif. Toutes les
métriques + une matrice de confusion sont affichées pour chaque modèle.

Sélection : `cross_val_score` (cv=10, `scoring='f1'`) sur un sous-échantillon stratifié de
**150 000** lignes du train ; le gagnant (XGBoost) est **ré-entraîné sur l'intégralité du train
(1 764 558 lignes)** puis évalué sur le jeu de test complet (756 240 lignes).

### Résultats sur le jeu de test (756 240 flux : 628 518 BENIGN / 127 722 attaques)

| Modèle | Précision | Recall | F1 | Accuracy | Faux positifs | Taux FP |
|---|---|---|---|---|---|---|
| Règles statiques | 0.4455 | 0.6142 | 0.5164 | 0.8057 | 97 654 | 15.54 % |
| **XGBoost (gardé)** | **0.9966** | **0.9987** | **0.9976** | **0.9992** | **438** | **0.07 %** |

- **Réduction des faux positifs vs baseline de règles** : XGBoost **−99.55 %** (438 vs 97 654).
- **Explicabilité** : importances globales de XGBoost (`feature_importances_`) + explication par
  alerte (valeur du flux vs moyenne du trafic BENIGN). Voir `reports/figures/09`, `10`, `12`.

### Choix techniques retenus
- Nettoyage : strip des noms de colonnes, normalisation des labels `Web Attack` (mojibake),
  `±Infinity → NaN → dropna`, suppression des doublons, downcast float32 → 2 520 798 lignes gelées.
- Features : 78 candidates → **8 colonnes constantes supprimées** → **70 features**.
- Déséquilibre : split **stratifié** sur `Label_binary` (pas de SMOTE).
- Panel : `LinearSVC, DecisionTree, LogisticRegression, GaussianNB, RandomForest, XGBoost, KNN`.
  Features mises à l'échelle une fois (`StandardScaler` fit sur le train) ; le modèle gardé attend
  des features scalées (le dashboard et la Phase 4 appliquent le scaler avant de prédire).
- Sélection sur le **F1** (équilibre recall/précision) ; toutes les métriques + matrice de confusion
  affichées pour chaque modèle.

### Artefacts produits
- `data/processed/cicids_clean.parquet` (dataset gelé), `test_set.parquet`, `app_sample.parquet`
- `models/` : **`best_model.pkl`** (XGBoost gardé, entrée scalée), `best_model_meta.json`,
  `cv_results.json` (7 modèles, toutes métriques), `scaler.pkl`, `features.pkl`, `benign_means.pkl`,
  `rule_baseline.json`
- `reports/metrics.json` + `reports/figures/01…12.png` (dont `11_bakeoff_cv_f1.png` et
  `12_confusion_matrices.png`)
- `notebooks/01_nettoyage` → `04_evaluation` (sources `.py` jupytext + `.ipynb` exécutés)
- `app/app.py` (dashboard Streamlit, charge `best_model.pkl` et affiche le modèle gardé)

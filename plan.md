# Mini-projet IA — Détection explicable d'anomalies dans les logs d'infrastructure bancaire

**Dataset:** CICIDS2017
**Modèles:** Isolation Forest + Random Forest
**Interface:** Streamlit

---

## Phase 0 — Setup
- [ ] Créer le repo Git + dossiers (`/data`, `/notebooks`, `/models`, `/app`)
- [ ] Créer l'environnement Python (venv + pandas, scikit-learn, streamlit, matplotlib/seaborn, joblib)
- [ ] Répartir les tâches entre les trois membres

## Phase 1 — Récupérer et comprendre les données
- [ ] Télécharger CICIDS2017 (8 fichiers CSV depuis le site du CIC)
- [ ] Identifier quel jour = quelle attaque, et le sens des labels
- [ ] Charger et inspecter (shapes, colonnes, distribution des labels)

## Phase 2 — Nettoyage (le vrai travail)
- [ ] Corriger les noms de colonnes (espaces en début de nom — piège connu)
- [ ] Gérer les `Infinity` / `NaN` dans les colonnes de flux (autre piège connu)
- [ ] Fusionner les 8 fichiers ou choisir un sous-ensemble ciblé
- [ ] Consolider les labels (binaire normal/attaque + multiclasse regroupée)
- [ ] Choisir la stratégie pour le déséquilibre des classes (class_weight / undersample / SMOTE)
- [ ] Train/test split + scaling

## Phase 3 — EDA
- [ ] Distributions, corrélations, top features par attaque
- [ ] Sauvegarder les figures propres pour le rapport

## Phase 4 — Baseline de règles statiques
- [ ] Écrire des règles simples par seuils sur quelques features
- [ ] Mesurer son taux de faux positifs (pour comparer avec les 60–80 % de INT-003)

## Phase 5 — Isolation Forest (apprentissage du comportement normal)
- [ ] Entraîner sur le trafic normal uniquement
- [ ] Régler `contamination` et `n_estimators`
- [ ] Évaluer (precision / recall / FP)

## Phase 6 — Random Forest (explicabilité)
- [ ] Entraîner le classifieur supervisé
- [ ] Régler les hyperparamètres
- [ ] Extraire l'importance globale des features
- [ ] Construire les top features par prédiction (le "pourquoi cette alerte")

## Phase 7 — Évaluation et comparaison
- [ ] Matrices de confusion + tableau de métriques
- [ ] Comparer les trois : règles vs Isolation Forest vs Random Forest
- [ ] Positionner les résultats face à la cible de 60–80 % de FP

## Phase 8 — Dashboard Streamlit
- [ ] Layout + chargement des modèles sauvegardés
- [ ] Afficher le verdict + le score de risque
- [ ] Afficher les top features
- [ ] Ajouter le panneau de comparaison avec la baseline de règles
- [ ] Ajouter le bouton de retour analyste (sauvegarde FP/TP dans un CSV)
- [ ] Polish

## Phase 9 — Rapport final
- [ ] Rédiger méthodologie + résultats
- [ ] Intégrer les insights / verbatims des entretiens
- [ ] Figures + mise en forme

## Phase 10 — Soutenance
- [ ] Slides
- [ ] Répétition

---

> ⚠️ La Phase 2 bloque tout le reste. Geler le dataset nettoyé avant de commencer la modélisation, sinon le travail est refait deux fois.

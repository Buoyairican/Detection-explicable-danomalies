# CLAUDE.md — Détection explicable d'anomalies (CICIDS2017)

Mini-projet IA S6 : détection explicable d'anomalies dans des logs réseau (CICIDS2017),
avec Isolation Forest + Random Forest et un dashboard Streamlit.
Voir `plan.md` pour la feuille de route.

---

## RÈGLE D'OR — Style de code

Tout le code généré doit imiter **exactement** le style du cours « Python - Partie 2 »
(Pr. Aniss Moumen). Ce cours enseigne du Python **plat, procédural, type notebook** :
**pas de fonctions, pas de classes, pas de type hints, pas de docstrings.**
Imports → chargement → sélection → nettoyage → encodage → scaling → split → fit → predict → évaluation → visualisation, le tout en instructions séquentielles.

Quand on s'écarte de ce style (ex. l'app Streamlit qui impose des callbacks),
rester **minimal** et garder le même esprit : commentaires français, noms courts, aucun sur-design.

---

## Imports (toujours ces alias, jamais d'autres)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
```

`pd`, `np`, `plt`, `sns` : jamais d'autres alias. Importer uniquement ce qu'on utilise.

## Nommage

- DataFrame = **`df`**. Versions dérivées : `df_clean`, `df_numeric`, `df_encoded`.
- Features = **`X`** (majuscule), cible = **`y`** (minuscule).
- `X_train, X_test, y_train, y_test = train_test_split(...)`.
- Listes de colonnes en haut du script : `features = ['Flow Duration', ...]` et `target = 'Label'`, puis `df = df[features + [target]]`.
- Instances courtes : `scaler = StandardScaler()`, `le = LabelEncoder()`, `iso_forest = IsolationForest(...)`, `rf = RandomForestClassifier(...)`, `model = ...`.
- `snake_case` pour les variables (`lower_bound`, `outliers_iqr`, `selected_features`).
- Colonnes : garder les noms d'origine du dataset (anglais). Colonnes dérivées : suffixe explicite (`'Fare_log'`, `'Age_scaled'`, `'anomaly'`).

## Commentaires

- **En français**, avec `#` uniquement. **Aucune docstring, aucun `"""..."""`.**
- Structurer chaque script avec des **étapes numérotées** :
  ```python
  # 1. Chargement des données
  # 2. Nettoyage : noms de colonnes + Infinity/NaN
  # 3. Encodage des labels
  # 4. Standardisation
  # 5. Split train/test
  # 6. Entraînement du modèle
  # 7. Évaluation
  # 8. Visualisation
  ```
- Commentaires courts, à l'impératif (« Chargement du dataset », « Suppression des lignes manquantes »).
- Commentaire en bout de ligne pour préciser : `df['anomaly'] = iso_forest.fit_predict(X_scaled)  # -1 = outlier, 1 = normal`.

## Structure du code

- **Scripts procéduraux de haut en bas.** Pas de `def`, pas de `class`, pas de `if __name__ == "__main__"`, pas de type hints.
- Le seul motif de structuration autorisé est le **dict de modèles + boucle for** pour comparer plusieurs modèles :
  ```python
  models = {"Isolation Forest": ..., "Random Forest": ...}
  for name, model in models.items():
      ...
      print(f"\n{name}")
      print(f"Accuracy: {acc:.4f}")
  ```
- Notebooks Jupyter pour l'analyse et la modélisation ; `.py` plat pour les scripts utilitaires (téléchargement, nettoyage).

## Idiomes pandas/sklearn (réutiliser ces patterns exacts)

```python
# Chargement
df = pd.read_csv("data/raw/fichier.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

# Nettoyage des noms de colonnes (piège CICIDS2017 : espaces en début de nom)
df.columns = df.columns.str.strip()

# Infinity / NaN
df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)

# Outliers (IQR)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# Encodage
le = LabelEncoder()
df['Label_encoded'] = le.fit_transform(df['Label'])

# Standardisation (fit sur train, transform sur test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

- `random_state=42` **partout** (split, IsolationForest, RandomForest).
- `test_size=0.3` par défaut ; `stratify=y` quand les classes sont déséquilibrées.

## Visualisation

- **seaborn (`sns`) en principal**, par-dessus matplotlib. Mélanger librement.
- **Titres et axes en français** (avec accents) :
  ```python
  plt.figure(figsize=(8, 6))
  sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
  plt.title("Matrice de corrélation entre variables numériques")
  plt.show()
  ```
- Fonctions usuelles : `sns.histplot(..., kde=True)`, `sns.countplot(...)`, `sns.boxplot(...)`, `sns.scatterplot(...)`, `sns.heatmap(...)`.
- Subplots : `fig, axs = plt.subplots(1, 3, figsize=(15, 4))` puis `ax=axs[i]`, finir par `plt.tight_layout()` puis `plt.show()`.
- Chaque bloc de graphique se termine par `plt.show()`. Config globale possible : `sns.set(style="whitegrid")`.

## ML / sklearn

- Modèles instanciés avec params par défaut sauf nécessité : `RandomForestClassifier(random_state=42)`, `IsolationForest(contamination=0.05, random_state=42)`.
- `model.fit(X_train, y_train)` → `y_pred = model.predict(X_test)`. Anomalie : `iso_forest.fit_predict(X_scaled)`.
- Métriques classif : `accuracy_score`, `classification_report`, `confusion_matrix`.
- Sauvegarde des modèles : `joblib.dump(rf, "models/random_forest.pkl")` / `joblib.load(...)`.

## Affichage des résultats

- **f-strings partout**, précision `:.4f` :
  ```python
  print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
  ```
- `print()` pour tout afficher (style script, ne pas compter sur l'auto-display du notebook).
- En-têtes de section en français : `print("=== Évaluation des modèles ===")`.

## À NE PAS faire

- ❌ Pas de fonctions / classes / type hints / docstrings (sauf strict besoin Streamlit).
- ❌ Pas de commentaires en anglais — tout en français.
- ❌ Pas de `try/except` ni de validation d'entrée (scripts happy-path).
- ❌ Pas d'alias d'import exotiques, pas de `Pipeline([...])` nommé ni `ColumnTransformer` — utiliser `make_pipeline(...)` si besoin.
- ❌ Pas de sur-ingénierie (config, logging, abstractions) : rester au niveau du cours.

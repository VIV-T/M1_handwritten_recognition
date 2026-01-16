# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:52:24 2025

@author: veroe
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Script de classification Random Forest sur features extraites d'images

Auteur: veroe
Date: 31/12/2025
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Charger les features et labels ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "features_GEO")

X = np.load(os.path.join(DATA_DIR, "features_GEO.npy"))
y = np.load(os.path.join(DATA_DIR, "labels_GEO.npy"), allow_pickle=True)


#X = np.load("features_all.npy")
#y = np.load("labels_all.npy", allow_pickle=True)  # obligatoire si y contient des objets/strings

print("X shape:", X.shape)
print("y shape:", y.shape)

# --- 2. Nettoyage des labels ---
mask = [label is not None for label in y]  # on ignore les None
X = X[mask]
y = np.array([label for label in y if label is not None])

print("Après nettoyage:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# --- 3. Normalisation ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Séparation train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Entraînement du modèle Random Forest ---
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)

# --- 6. Évaluation ---
y_pred = clf.predict(X_test)
print("\nAccuracy sur le set test:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# --- 7. Sauvegarde du modèle et du scaler (optionnel) ---
import joblib
joblib.dump(clf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModèle et scaler sauvegardés.")

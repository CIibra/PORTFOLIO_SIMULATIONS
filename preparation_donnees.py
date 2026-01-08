"""
Script : preparation_donnees.py
Auteur : Ib Cisse
Objectif : Nettoyer, préparer et standardiser un jeu de données client pour analyse ou modélisation.

Étapes réalisées :
1. Chargement des données brutes depuis un fichier CSV
2. Nettoyage :
   - Suppression des doublons
   - Suppression des valeurs manquantes
   - Filtrage des lignes incohérentes (revenu <= 0)
3. Standardisation :
   - Mise en minuscules des noms
   - Uniformisation des adresses (strip, remplacement)
   - Formatage des dates
4. Création de variables dérivées :
   - Âge à partir de la date de naissance
   - Catégorie de revenu
5. Export du jeu de données nettoyé vers un nouveau fichier CSV
"""

import pandas as pd
from datetime import datetime

# 1. Chargement
df = pd.read_csv("clients_bruts.csv")

# 2. Nettoyage
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = df[df["revenu"] > 0]

# 3. Standardisation
df["nom"] = df["nom"].str.lower().str.strip()
df["adresse"] = df["adresse"].str.replace(" av ", " avenue ").str.strip()
df["date_naissance"] = pd.to_datetime(df["date_naissance"], errors="coerce")

# 4. Variables dérivées
today = pd.Timestamp("today")
df["age"] = (today - df["date_naissance"]).dt.days // 365
df["categorie_revenu"] = pd.cut(df["revenu"],
    bins=[0, 20000, 50000, 100000, float("inf")],
    labels=["faible", "moyen", "élevé", "très élevé"]
)

# 5. Export
df.to_csv("clients_nettoyes.csv", index=False)

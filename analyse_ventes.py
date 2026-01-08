# ------------------------------------------------------------
# Analyse descriptive des ventes numériques
# Génère 5 graphes PNG pour rapport et portfolio
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("ventes.csv")

# ------------------------------------------------------------
# Étape 1 : Graphique global (CA total vs Marge totale)
# ------------------------------------------------------------
ca_total = df["Chiffre d'affaires"].sum()
marge_totale = df["Marge"].sum()

plt.figure(figsize=(6,4))
plt.bar(["CA total", "Marge totale"], [ca_total, marge_totale], color=["skyblue","orange"])
plt.title("Chiffre d'affaires et marge globale")
plt.ylabel("Montant (€)")
plt.savefig("ca_global.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Étape 2 : Graphique par région
# ------------------------------------------------------------
stats_region = df.groupby("Région")["Chiffre d'affaires"].sum().sort_values()

plt.figure(figsize=(8,5))
stats_region.plot(kind="bar", color="green")
plt.title("Chiffre d'affaires par région")
plt.ylabel("CA (€)")
plt.savefig("ca_par_region.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Étape 3 : Graphique par produit
# ------------------------------------------------------------
stats_produit = df.groupby("Produit")["Chiffre d'affaires"].sum().sort_values()

plt.figure(figsize=(10,6))
stats_produit.plot(kind="barh", color="purple")
plt.title("Chiffre d'affaires par produit")
plt.xlabel("CA (€)")
plt.savefig("ca_par_produit.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Étape 4 : Graphique par équipe
# ------------------------------------------------------------
stats_equipe = df.groupby("Équipe")["Chiffre d'affaires"].sum()

plt.figure(figsize=(6,4))
stats_equipe.plot(kind="bar", color="red")
plt.title("Chiffre d'affaires par équipe")
plt.ylabel("CA (€)")
plt.savefig("ca_par_equipe.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Étape 5 : Graphique temporel (CA par jour)
# ------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
stats_jour = df.groupby("Date")["Chiffre d'affaires"].sum()

plt.figure(figsize=(8,5))
stats_jour.plot(marker="o", linestyle="-", color="blue")
plt.title("Évolution du chiffre d'affaires par jour")
plt.ylabel("CA (€)")
plt.xlabel("Date")
plt.savefig("evolution_ca.png", dpi=300)
plt.close()

print("✅ Graphes générés : ca_global.png, ca_par_region.png, ca_par_produit.png, ca_par_equipe.png, evolution_ca.png")

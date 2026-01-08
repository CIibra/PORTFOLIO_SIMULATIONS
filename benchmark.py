# ------------------------------------------------------------
# etape_benchmark.py
# Étape 3 : Benchmark interne des ventes numériques
# Objectif : comparer les performances entre équipes et régions
# Génère des graphes PNG pour rapport et portfolio
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------------
# 0) Chargement et préparation
# -------------------------------
DATA_FILE = "ventes.csv"
OUT_DIR = Path("out_benchmark")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# -------------------------------
# 1) CA et marge par équipe
# -------------------------------
stats_equipe = df.groupby("Équipe")[["Chiffre d'affaires","Marge"]].sum().reset_index()

plt.figure(figsize=(7,5))
stats_equipe.set_index("Équipe").plot(kind="bar", stacked=False, color=["#2563eb","#f59e0b"])
plt.title("Benchmark interne : CA et marge par équipe")
plt.ylabel("Montant (€)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "benchmark_ca_marge_equipe.png", dpi=300)
plt.close()

# -------------------------------
# 2) CA par région et équipe (heatmap)
# -------------------------------
pivot_region_equipe = df.pivot_table(
    index="Région", columns="Équipe", values="Chiffre d'affaires", aggfunc="sum"
)

plt.figure(figsize=(8,6))
sns.heatmap(pivot_region_equipe, annot=True, fmt=".0f", cmap="Blues")
plt.title("Benchmark interne : CA par région et équipe")
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_region_equipe.png", dpi=300)
plt.close()

# -------------------------------
# 3) Parts de marché par catégorie
# -------------------------------
stats_categorie = df.groupby("Catégorie")["Chiffre d'affaires"].sum().reset_index()

plt.figure(figsize=(7,7))
plt.pie(stats_categorie["Chiffre d'affaires"], labels=stats_categorie["Catégorie"],
        autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
plt.title("Répartition du CA par catégorie de produit")
plt.tight_layout()
plt.savefig(OUT_DIR / "part_categorie.png", dpi=300)
plt.close()

# -------------------------------
# 4) Ratio marge/CA par équipe
# -------------------------------
stats_equipe["Ratio_marge"] = stats_equipe["Marge"] / stats_equipe["Chiffre d'affaires"]

plt.figure(figsize=(7,5))
sns.barplot(x="Équipe", y="Ratio_marge", data=stats_equipe, palette="viridis")
plt.title("Ratio marge/CA par équipe")
plt.ylabel("Ratio")
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_marge_equipe.png", dpi=300)
plt.close()

print("✅ Graphes générés dans:", OUT_DIR.resolve())
print(" - benchmark_ca_marge_equipe.png")
print(" - heatmap_region_equipe.png")
print(" - part_categorie.png")
print(" - ratio_marge_equipe.png")

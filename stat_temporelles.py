# ------------------------------------------------------------
# etape_temporelle.py
# Statistiques temporelles des ventes (Étape 2 - temporel)
# Génère des graphes PNG et un résumé TXT pour le rapport/portfolio
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# 0) Chargement et préparation
# -------------------------------
DATA_FILE = "ventes.csv"
OUT_DIR = Path("out_temporel")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# Vérifications minimales
required_cols = {"Date", "Chiffre d'affaires"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans {DATA_FILE}: {missing}")

# -------------------------------
# 1) CA par jour (agrégation)
# -------------------------------
daily = (
    df.groupby("Date", as_index=False)["Chiffre d'affaires"]
      .sum()
      .sort_values("Date")
      .rename(columns={"Chiffre d'affaires": "CA"})
)

# -------------------------------
# 2) Moyenne mobile (fenêtre 3 jours)
# -------------------------------
daily["CA_MA3"] = daily["CA"].rolling(window=3, min_periods=1).mean()

# -------------------------------
# 3) Détection simple des pics/creux
# Méthode: seuils basés sur moyenne +/- écart-type
# -------------------------------
mean_ca = daily["CA"].mean()
std_ca = daily["CA"].std(ddof=1) if len(daily) > 1 else 0.0
high_thresh = mean_ca + std_ca
low_thresh  = mean_ca - std_ca

daily["Pic"]  = daily["CA"] >= high_thresh
daily["Creux"] = daily["CA"] <= low_thresh

# -------------------------------
# 4) Jour de la semaine (analyse)
# -------------------------------
daily["Jour_semaine"] = daily["Date"].dt.day_name(locale="fr_FR") if hasattr(daily["Date"].dt, "day_name") else daily["Date"].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
# Essayer de mapper en français si disponible
map_fr = {
    "Monday":"Lundi","Tuesday":"Mardi","Wednesday":"Mercredi",
    "Thursday":"Jeudi","Friday":"Vendredi","Saturday":"Samedi","Sunday":"Dimanche"
}
daily["Jour_semaine_fr"] = daily["Jour_semaine"].map(map_fr).fillna(daily["Jour_semaine"])

weekday_stats = (
    daily.groupby("Jour_semaine_fr", as_index=False)["CA"]
         .mean()
)
# Ordre des jours en français
weekday_order_fr = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
weekday_stats["order"] = weekday_stats["Jour_semaine_fr"].apply(lambda x: weekday_order_fr.index(x) if x in weekday_order_fr else 999)
weekday_stats = weekday_stats.sort_values("order")

# -------------------------------
# 5) Graphiques
# -------------------------------

# 5a) Évolution du CA par jour
plt.figure(figsize=(9,5))
plt.plot(daily["Date"], daily["CA"], marker="o", linewidth=2, color="#2563eb")  # bleu
plt.title("Évolution du chiffre d'affaires par jour")
plt.xlabel("Date")
plt.ylabel("CA (€)")
plt.grid(alpha=0.25)
for _, row in daily.iterrows():
    if row["Pic"]:
        plt.scatter(row["Date"], row["CA"], color="#16a34a", s=60, zorder=3)  # vert pour pic
    elif row["Creux"]:
        plt.scatter(row["Date"], row["CA"], color="#dc2626", s=60, zorder=3)  # rouge pour creux
plt.tight_layout()
plt.savefig(OUT_DIR / "evolution_ca.png", dpi=300)
plt.close()

# 5b) CA + moyenne mobile 3 jours
plt.figure(figsize=(9,5))
plt.plot(daily["Date"], daily["CA"], marker="o", linewidth=1.5, color="#0ea5e9", label="CA quotidien")
plt.plot(daily["Date"], daily["CA_MA3"], linewidth=2.5, color="#f59e0b", label="Moyenne mobile (3 jours)")
plt.title("CA par jour et moyenne mobile (3 jours)")
plt.xlabel("Date")
plt.ylabel("CA (€)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "evolution_ca_ma3.png", dpi=300)
plt.close()

# 5c) CA moyen par jour de la semaine
plt.figure(figsize=(8,4.5))
plt.bar(weekday_stats["Jour_semaine_fr"], weekday_stats["CA"], color="#10b981")  # vert
plt.title("CA moyen par jour de la semaine")
plt.xlabel("Jour")
plt.ylabel("CA moyen (€)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "ca_par_jour_semaine.png", dpi=300)
plt.close()

# -------------------------------
# 6) Résumé TXT (pour rapport)
# -------------------------------
resume_lines = []
resume_lines.append("=== Résumé temporel ===")
resume_lines.append(f"Nombre de jours observés: {len(daily)}")
resume_lines.append(f"CA moyen/jour: {mean_ca:,.0f} €")
resume_lines.append(f"Écart-type CA: {std_ca:,.0f} €")
resume_lines.append(f"Seuil pic (>=): {high_thresh:,.0f} €")
resume_lines.append(f"Seuil creux (<=): {low_thresh:,.0f} €")

# Jours pic/creux
pics = daily[daily["Pic"]][["Date","CA"]]
creux = daily[daily["Creux"]][["Date","CA"]]
if not pics.empty:
    resume_lines.append("Jours 'pic' détectés:")
    for _, r in pics.iterrows():
        resume_lines.append(f" - {r['Date'].date()} : {r['CA']:,.0f} €")
else:
    resume_lines.append("Aucun 'pic' détecté.")

if not creux.empty:
    resume_lines.append("Jours 'creux' détectés:")
    for _, r in creux.iterrows():
        resume_lines.append(f" - {r['Date'].date()} : {r['CA']:,.0f} €")
else:
    resume_lines.append("Aucun 'creux' détecté.")

# Top jour de semaine
top_weekday = weekday_stats.sort_values("CA", ascending=False).head(1)
if not top_weekday.empty:
    jw, val = top_weekday.iloc[0]["Jour_semaine_fr"], top_weekday.iloc[0]["CA"]
    resume_lines.append(f"Jour de semaine le plus performant (moyenne): {jw} ({val:,.0f} €)")
else:
    resume_lines.append("Impossible de déterminer le jour le plus performant.")

(OUT_DIR / "resume_temporel.txt").write_text("\n".join(resume_lines), encoding="utf-8")

print("✅ Visuels générés dans:", OUT_DIR.resolve())
print(" - evolution_ca.png")
print(" - evolution_ca_ma3.png")
print(" - ca_par_jour_semaine.png")
print("📄 Résumé:", (OUT_DIR / "resume_temporel.txt").resolve())

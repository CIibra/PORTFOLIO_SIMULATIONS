# ------------------------------------------------------------
# etape_insights.py
# Étape 4 : Insights et recommandations
# Objectif : générer un rapport PDF avec les graphes et un résumé
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path

# -------------------------------
# 0) Chargement et préparation
# -------------------------------
DATA_FILE = "ventes.csv"
OUT_DIR = Path("out_insights")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# -------------------------------
# 1) Constats clés (calculs simples)
# -------------------------------
ca_total = df["Chiffre d'affaires"].sum()
marge_total = df["Marge"].sum()
top_region = df.groupby("Région")["Chiffre d'affaires"].sum().idxmax()
top_produit = df.groupby("Produit")["Chiffre d'affaires"].sum().idxmax()
top_equipe = df.groupby("Équipe")["Chiffre d'affaires"].sum().idxmax()

insights = [
    f"CA total observé : {ca_total:,.0f} EUR",
    f"Marge totale : {marge_total:,.0f} EUR",
    f"Région la plus performante : {top_region}",
    f"Produit best-seller : {top_produit}",
    f"Équipe leader en CA : {top_equipe}",
]

recommandations = [
    "Renforcer les équipes dans les régions à faible CA (Nice, Toulouse).",
    "Investir davantage dans les produits best-sellers (Galaxy Tab S9, MacBook Air M2).",
    "Développer les segments accessoires audio et montres connectées, en croissance.",
    "Optimiser la marge en ajustant le portefeuille produit (premium vs volume).",
]

# -------------------------------
# 2) Génération du rapport PDF
# -------------------------------
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Rapport d'analyse des ventes numériques", ln=True, align="C")

pdf.set_font("Arial", "", 12)
pdf.cell(0, 10, "Étape 4 : Insights et recommandations", ln=True)

pdf.ln(5)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Constats clés :", ln=True)
pdf.set_font("Arial", "", 12)
for line in insights:
    pdf.multi_cell(0, 8, f"- {line}")

pdf.ln(5)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Recommandations :", ln=True)
pdf.set_font("Arial", "", 12)
for line in recommandations:
    pdf.multi_cell(0, 8, f"- {line}")

pdf.ln(10)
pdf.set_font("Arial", "I", 10)
pdf.multi_cell(0, 8, "Ce rapport inclut les visuels générés aux étapes précédentes (benchmark, temporel, descriptif).")

# Sauvegarde
pdf.output(str(OUT_DIR / "rapport_insights.pdf"))

print("✅ Rapport PDF généré :", (OUT_DIR / "rapport_insights.pdf").resolve())

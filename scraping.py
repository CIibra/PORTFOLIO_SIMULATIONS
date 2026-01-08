"""
Script : scraping.py
Auteur : CISSE ibrahim
Scraping en temps réel des séismes mondiaux
-------------------------------------------
Ce script interroge l'API publique de l'USGS toutes les 30 secondes pour récupérer les séismes
de magnitude ≥ 4.5 survenus dans la dernière heure. Les données sont enregistrées dans un fichier CSV.

Source : https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
"""

import requests
import csv
import time
from datetime import datetime

# URL de l'API USGS pour les séismes récents (magnitude ≥ 4.5, dernière heure)
URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_hour.geojson"

# Nom du fichier CSV de sortie
FICHIER = "seismes_temps_reel.csv"

# Intervalle entre chaque requête (en secondes)
INTERVALLE = 30

# Initialisation du fichier CSV avec en-têtes
with open(FICHIER, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Horodatage", "Magnitude", "Lieu", "Date UTC", "Latitude", "Longitude"])

print("Démarrage du scraping en temps réel...")

# Boucle infinie (à interrompre manuellement)
while True:
    try:
        # Requête vers l'API USGS
        response = requests.get(URL)
        if response.status_code != 200:
            print(f"Erreur {response.status_code}. Nouvelle tentative dans {INTERVALLE} secondes.")
            time.sleep(INTERVALLE)
            continue

        data = response.json()
        features = data["features"]

        # Ouverture en mode ajout
        with open(FICHIER, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for quake in features:
                props = quake["properties"]
                coords = quake["geometry"]["coordinates"]
                magnitude = props["mag"]
                lieu = props["place"]
                date_utc = datetime.utcfromtimestamp(props["time"] / 1000).isoformat()
                lat, lon = coords[1], coords[0]
                horodatage = datetime.now().isoformat()
                writer.writerow([horodatage, magnitude, lieu, date_utc, lat, lon])

        print(f"{len(features)} séismes enregistrés à {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(INTERVALLE)

    except Exception as e:
        print(f"Erreur : {e}. Pause de sécurité...")
        time.sleep(INTERVALLE)

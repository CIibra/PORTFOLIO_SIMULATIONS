# -*- coding: utf-8 -*-
"""
Cas industriel : Isolation thermique d’un bâtiment (mur multicouche)
--------------------------------------------------------------------
Un mur est composé de plusieurs couches (béton, isolant, plâtre).
La chaleur traverse par conduction, avec convection intérieure et extérieure.

Résultats attendus :
- Calcul du flux thermique traversant le mur.
- Profil de température aux interfaces des couches.
- Coefficient global de transmission thermique (U-value).

"""

import numpy as np
import matplotlib.pyplot as plt

# Températures intérieure et extérieure
T_int = 293.0   # K (20 °C)
T_ext = 273.0   # K (0 °C)

# Convection intérieure et extérieure
h_int = 8.0     # W/(m².K)
h_ext = 25.0    # W/(m².K)

# Couches : (épaisseur m, conductivité W/m.K, nom)
layers = [
    (0.20, 1.4, "Béton"),     # 20 cm béton
    (0.10, 0.04, "Isolant"),  # 10 cm isolant
    (0.02, 0.25, "Plâtre")    # 2 cm plâtre
]

# Calcul des résistances
R_conv_int = 1/h_int
R_conv_ext = 1/h_ext
R_layers = [e/k for e,k,_ in layers]
R_tot = R_conv_int + sum(R_layers) + R_conv_ext

# Flux thermique
q_flux = (T_int - T_ext)/R_tot
U_value = 1/R_tot

print("Flux thermique q'' =", q_flux, "W/m²")
print("Coefficient U =", U_value, "W/(m².K)")

# Profil de température aux interfaces
T_profile = [T_int]
R_cum = R_conv_int
cum_thickness = 0
x_pos = [0]

for (e,k,name) in layers:
    R_cum += e/k
    cum_thickness += e
    T_profile.append(T_int - q_flux*R_cum)
    x_pos.append(cum_thickness)

# Ajouter la convection extérieure
T_profile.append(T_ext)
x_pos.append(cum_thickness + 0.01)  # position fictive pour convection ext

# Visualisation
plt.figure(figsize=(7,5))
plt.plot(x_pos, T_profile, marker="o")
plt.xlabel("Épaisseur mur (m)")
plt.ylabel("Température (K)")
plt.title("Profil de température à travers un mur multicouche")
plt.grid(True, alpha=0.3)
plt.show()


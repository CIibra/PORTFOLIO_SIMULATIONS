"""
Script général d'automatisation de simulations OpenFOAM.

UTILISATION :

1. Placer le dossier du cas de base (avec 0/, constant/, system/) dans un dossier accessible.
2. Modifier les paramètres dans la section 'PARAMÈTRES À FAIRE VARIER'.
   - Exemple : velocities = [1, 2, 3]
3. Lancer le script avec Python 3 :
       python3 run_parametric.py
4. Les dossiers de résultats seront créés dans 'results_dir', un dossier par combinaison.
5. Le script gère automatiquement :
   - suppression des anciens cas,
   - génération du maillage si absent,
   - lancement du solveur.
6. Pour ajouter d'autres paramètres :
   - Ajouter des listes pour chaque paramètre,
   - Modifier la fonction de modification correspondante,
   - Ajouter la variable dans la boucle principale.
"""

import os
import shutil
import subprocess

# -------------------------
# PARAMÈTRES À FAIRE VARIER
# -------------------------
# Exemple : vitesse d'entrée (modifiable selon le cas)
velocities = [1, 2]  # en m/s

# Cas de base et dossier de résultats
base_case = "baseCase"          # dossier contenant le cas OpenFOAM de référence
results_dir = "parametric_study"  # dossier où les simulations seront créées
os.makedirs(results_dir, exist_ok=True)

# -------------------------
# FONCTIONS UTILITAIRES
# -------------------------

def modify_U(case_dir, U_value):
    """
    Modifie la vitesse d'entrée dans 0/U pour la simulation.
    """
    U_file = os.path.join(case_dir, "0", "U")
    if not os.path.exists(U_file):
        print(f"Attention : fichier {U_file} introuvable, vérifiez le cas de base.")
        return
    with open(U_file, "r") as f:
        lines = f.readlines()
    with open(U_file, "w") as f:
        for line in lines:
            if "internalField" in line and "uniform" in line:
                f.write(f"internalField   uniform ({U_value} 0 0);\n")
            else:
                f.write(line)

def run_case(case_dir):
    """
    Génère le maillage si nécessaire et lance la simulation OpenFOAM.
    """
    polyMesh = os.path.join(case_dir, "constant", "polyMesh")
    if not os.path.exists(polyMesh) or not os.listdir(polyMesh):
        subprocess.run(f"cd {case_dir} && blockMesh", shell=True, check=True)
    subprocess.run(f"cd {case_dir} && simpleFoam", shell=True, check=True)

# -------------------------
# BOUCLE PRINCIPALE
# -------------------------
for U in velocities:
    # Nom du cas basé sur les paramètres
    case_name = f"U{U}"
    case_dir = os.path.join(results_dir, case_name)

    # Supprimer l'ancien cas si déjà existant
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

    # Copier le cas de base
    shutil.copytree(base_case, case_dir)

    # Modifier les paramètres
    modify_U(case_dir, U)

    # Lancer la simulation
    run_case(case_dir)

    print(f"Simulation terminée pour {case_name}")

print("Toutes les simulations paramétriques sont terminées.")

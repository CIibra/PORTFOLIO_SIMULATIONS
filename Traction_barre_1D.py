import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Étape 1 : Définition des paramètres du problème
# ======================================================================

# Propriétés du matériau et de la géométrie
E = 2e6      # Module d'Young (Pa)
A = 0.01        # Aire de la section (m^2)
L = 1.0         # Longueur totale (m)

# Paramètres de la MEF
N_elem = 4      # Nombre d'éléments (segments)
N_nodes = N_elem + 1 # Nombre de nœuds
L_elem = L / N_elem  # Longueur de chaque élément

# ======================================================================
# Étape 2 : Création des Matrices et Vecteurs Globaux
# ======================================================================

K_global = np.zeros((N_nodes, N_nodes))
F_global = np.zeros(N_nodes)

# Coordonnées initiales des nœuds
nodes_coords_initial = np.linspace(0, L, N_nodes) # [0.0, 0.25, 0.5, 0.75, 1.0]

# ======================================================================
# Étape 3 : Calcul de K^e et Assemblage
# ======================================================================

C = (E * A) / L_elem
K_elem = C * np.array([[1, -1], 
                       [-1, 1]])

for e in range(N_elem):
    i = e       # Nœud gauche
    j = e + 1   # Nœud droit
    
    K_global[i, i] += K_elem[0, 0]
    K_global[i, j] += K_elem[0, 1]
    K_global[j, i] += K_elem[1, 0]
    K_global[j, j] += K_elem[1, 1]

# ======================================================================
# Étape 4 : Application des Conditions aux Limites et des Forces
# ======================================================================

fixed_dof = 0  # Nœud 0 est encastré (u_0 = 0)
P_applied = 1000.0   # Force de traction appliquée (N)
F_global[N_nodes - 1] = P_applied # Force au dernier nœud

# ======================================================================
# Étape 5 : Résolution du Système Réduit
# ======================================================================

K_r = np.delete(K_global, fixed_dof, axis=0)
K_r = np.delete(K_r, fixed_dof, axis=1)
F_r = np.delete(F_global, fixed_dof, axis=0)

try:
    U_r = np.linalg.solve(K_r, F_r)
except np.linalg.LinAlgError:
    print("Erreur : La matrice de rigidité réduite est singulière.")
    U_r = np.zeros(N_nodes - 1)

U_global = np.insert(U_r, fixed_dof, 0.0)

# Coordonnées des nœuds après déformation
nodes_coords_deformed = nodes_coords_initial + U_global

# ======================================================================
# Étape 6 : Post-traitement (Contraintes et Effort Interne)
# ======================================================================

stresses = np.zeros(N_elem)
internal_forces = np.zeros(N_elem)

for e in range(N_elem):
    u_i = U_global[e]
    u_j = U_global[e+1]
    
    epsilon = (u_j - u_i) / L_elem
    stresses[e] = E * epsilon
    internal_forces[e] = stresses[e] * A

F_reaction = K_global[fixed_dof, :] @ U_global - F_global[fixed_dof]

# ======================================================================
# Étape 7 : Affichage des Résultats Textuels
# ======================================================================

print("Résultats de la simulation (MEF)")
print("-" * 35)

print("\nDéplacements Nodaux U:")
for i in range(N_nodes):
    print(f"Nœud {i} (x={nodes_coords_initial[i]:.2f} m) : u = {U_global[i]:.6e} m")

print("\nContraintes axiales (uniformes dans chaque élément):")
for e in range(N_elem):
    print(f"Élément {e}: σ = {stresses[e]:.2f} Pa (N = {internal_forces[e]:.2f} N)")

print(f"\nForce de réaction à l'encastrement (Nœud 0) : R = {F_reaction:.2f} N")

# ======================================================================
# Étape 8 : Visualisation avec Matplotlib
# ======================================================================

plt.style.use('seaborn-v0_8-darkgrid')

# --- Figure 1: Représentation de la barre déformée ---
plt.figure(figsize=(12, 5))
plt.title('Déformation de la Barre 1D sous Traction Axiale')

# Décalage vertical pour visualiser la barre déformée sans qu'elle écrase la barre initiale
vertical_offset = L / 10 # Un petit décalage pour la visualisation
amplification_factor = 1e5 # Facteur d'amplification visuelle pour mieux voir le déplacement sur l'axe X

# Barre initiale (ligne bleue en Y=0)
plt.plot(nodes_coords_initial, np.zeros_like(nodes_coords_initial), 'o--', color='blue', label='Barre initiale', zorder=2)
# Barre déformée (ligne rouge légèrement décalée vers le haut, avec amplification)
plt.plot(nodes_coords_deformed, np.zeros_like(nodes_coords_deformed) + vertical_offset, 'x-', color='red', label=f'Barre déformée (Déplacement x {amplification_factor:.0e})', zorder=2)


# Affichage des nœuds avec leurs étiquettes
for i in range(N_nodes):
    # Nœuds initiaux
    plt.text(nodes_coords_initial[i], -vertical_offset*0.5, f'N{i}', color='blue', ha='center', va='top', fontsize=9)
    plt.plot(nodes_coords_initial[i], 0, 'o', color='blue', markersize=8, zorder=3)
    
    # Nœuds déformés
    plt.text(nodes_coords_deformed[i], vertical_offset*1.5, f'N{i}\'', color='red', ha='center', va='bottom', fontsize=9)
    plt.plot(nodes_coords_deformed[i], vertical_offset, 'x', color='red', markersize=8, zorder=3)

    # Flèches de déplacement (représentation du déplacement axial)
    if i > 0: # Pas de flèche pour le nœud encastré
        plt.arrow(nodes_coords_initial[i], 0, 
                  U_global[i] * amplification_factor, 0, 
                  color='gray', width=0.005, head_width=0.02, head_length=0.02, 
                  length_includes_head=True, zorder=1)


# Condition limite : Encastrement (mur gris)
plt.axvline(x=nodes_coords_initial[fixed_dof], color='gray', linestyle=':', linewidth=2, label='Encastrement', zorder=0)
plt.fill_betweenx([-0.1, L/5], -0.1, nodes_coords_initial[fixed_dof], color='lightgray', alpha=0.5, zorder=0)

# Force appliquée (flèche verte)
plt.arrow(nodes_coords_initial[-1], 0, 0.1, 0, color='green', width=0.01, head_width=0.03, head_length=0.03, label=f'Force P = {P_applied} N', zorder=2)
plt.text(nodes_coords_initial[-1] + 0.12, 0, f'{P_applied} N', color='green', ha='left', va='center', fontsize=10)


plt.xlabel('Position x (m)')
plt.ylabel('Représentation Y (pour visualisation)')
plt.yticks([]) # Pas de graduations sur l'axe Y si on utilise un décalage
plt.xlim(-0.1, L + 0.25)
plt.ylim(-vertical_offset*2, vertical_offset*2) # Ajuster les limites Y
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Légende à l'extérieur pour ne pas chevaucher
plt.gca().set_aspect('equal', adjustable='box') # Aspect ratio pour éviter l'écrasement
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajuster la mise en page pour la légende externe

# --- Figure 2: Tracé des déplacements nodaux ---
plt.figure(figsize=(10, 5))
plt.title('Déplacements Nodaux (u) le long de la Barre')
plt.plot(nodes_coords_initial, U_global, 'o-', color='purple', label='Déplacement u(x)')
plt.xlabel('Position x (m)')
plt.ylabel('Déplacement u (m)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.axvline(x=nodes_coords_initial[fixed_dof], color='gray', linestyle=':', label='Encastrement')

# Ajout des valeurs des déplacements sur le graphique
for i, (x_val, u_val) in enumerate(zip(nodes_coords_initial, U_global)):
    plt.text(x_val, u_val * 1.1, f'{u_val:.2e}', ha='center', va='bottom' if u_val >=0 else 'top', fontsize=9, color='purple')
plt.tight_layout()

# --- Figure 3: Tracé des contraintes axiales ---
plt.figure(figsize=(10, 5))
plt.title('Contraintes Axiales ($\sigma$) le long de la Barre')

# Les contraintes sont constantes par élément. On les trace comme un escalier.
x_stress = []
y_stress = []
for e in range(N_elem):
    x_stress.extend([nodes_coords_initial[e], nodes_coords_initial[e+1]])
    y_stress.extend([stresses[e], stresses[e]])

plt.plot(x_stress, y_stress, 's-', color='green', label='Contrainte axiale $\sigma(x)$')
plt.xlabel('Position x (m)')
plt.ylabel('Contrainte $\sigma$ (Pa)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.axvline(x=nodes_coords_initial[fixed_dof], color='gray', linestyle=':', label='Encastrement')

# Ajout des valeurs des contraintes sur le graphique
for e in range(N_elem):
    mid_x = (nodes_coords_initial[e] + nodes_coords_initial[e+1]) / 2
    plt.text(mid_x, stresses[e] * 1.1, f'{stresses[e]:.0f} Pa', ha='center', va='bottom' if stresses[e] >= 0 else 'top', fontsize=9, color='green')
plt.tight_layout()

plt.show() # Affiche toutes les figures
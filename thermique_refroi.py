# -*- coding: utf-8 -*-
"""
Simulation thermique 2D (plaque) avec convection de surface et visualisations réduites.
- Schéma implicite (Backward Euler) sur maillage cartésien régulier
- Conditions aux limites: Dirichlet (gauche), Neumann isolant (droite & bas), Convection (haut)
- Visualisations: Température T aux snapshots, flux convectif moyen, bilan d'énergie, sondes
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Paramètres physiques et numériques
# -------------------------------------------------------------------------
rho = 7850.0
cp  = 500.0
k   = 45.0

h     = 35.0
T_inf = 293.0

Lx, Ly = 0.2, 0.1
Nx, Ny = 120, 60
dx, dy = Lx/(Nx-1), Ly/(Ny-1)

dt     = 0.2
t_end  = 60.0
n_steps = int(np.ceil(t_end/dt))

q_vol = 0.0

T0    = 333.0
T_fix = 333.0

times_to_plot = [0, 10, 30, 60]
TOL = 1e-9

# -------------------------------------------------------------------------
# Utilitaires
# -------------------------------------------------------------------------
def idx(i, j):
    return j*Nx + i

N = Nx * Ny
X = np.linspace(0, Lx, Nx)
Y = np.linspace(0, Ly, Ny)

# -------------------------------------------------------------------------
# Assemblage matrice implicite
# -------------------------------------------------------------------------
alpha_x = k / dx**2
alpha_y = k / dy**2
mass    = rho * cp

diagonals = np.zeros(N)
lower_x = np.zeros(N)
upper_x = np.zeros(N)
lower_y = np.zeros(N)
upper_y = np.zeros(N)
rhs_const = np.zeros(N)

for j in range(Ny):
    for i in range(Nx):
        p = idx(i, j)
        diagonals[p] = mass + dt * (2*alpha_x + 2*alpha_y)
        if i > 0:
            lower_x[p] = -dt * alpha_x
        if i < Nx-1:
            upper_x[p] = -dt * alpha_x
        if j > 0:
            lower_y[p] = -dt * alpha_y
        if j < Ny-1:
            upper_y[p] = -dt * alpha_y

# Dirichlet gauche
for j in range(Ny):
    p = idx(0, j)
    diagonals[p] = 1.0
    lower_x[p] = upper_x[p] = 0.0
    lower_y[p] = upper_y[p] = 0.0

# Neumann droite
for j in range(Ny):
    p = idx(Nx-1, j)
    diagonals[p] -= dt * alpha_x

# Neumann bas
for i in range(Nx):
    p = idx(i, 0)
    diagonals[p] -= dt * alpha_y

# Convection haut
for i in range(Nx):
    p = idx(i, Ny-1)
    hc = dt * (h / dy)
    diagonals[p] += hc
    rhs_const[p] += hc * T_inf

A = sp.diags(diagonals, 0, shape=(N, N))
A += sp.diags(lower_x[1:], -1, shape=(N, N))
A += sp.diags(upper_x[:-1], 1, shape=(N, N))
A += sp.diags(lower_y[Nx:], -Nx, shape=(N, N))
A += sp.diags(upper_y[:-Nx], Nx, shape=(N, N))

Mvec = np.full(N, mass)
solver = spla.factorized(A.tocsc())

# -------------------------------------------------------------------------
# Simulation + snapshots
# -------------------------------------------------------------------------
T = np.full(N, T0)
snapshots = {}
probe_pts = [(int(0.5*(Nx-1)), int(0.5*(Ny-1))), (Nx-2, Ny-1)]
labels = ["Centre", "Bord convectif"]
probes_hist = [[] for _ in probe_pts]
flux_mean = []
energy = []
time = []

area_cell = dx * dy

# état initial
t_current = 0.0
time.append(t_current)
T2D = T.reshape(Ny, Nx)
if any(abs(t_current - t_req) < TOL for t_req in times_to_plot):
    snapshots[0] = T2D.copy()

for k,(ix,iy) in enumerate(probe_pts):
    probes_hist[k].append(T2D[iy, ix])
q_conv_line = h * (T2D[Ny-1, :] - T_inf)
flux_mean.append(np.mean(q_conv_line))
energy.append((rho*cp) * np.sum(T2D) * area_cell)

# boucle
for n in range(1, n_steps+1):
    rhs = Mvec * T + rhs_const + dt * q_vol
    for j in range(Ny):
        rhs[idx(0, j)] = T_fix
    T = solver(rhs)

    t_current = n * dt
    time.append(t_current)

    T2D = T.reshape(Ny, Nx)

    for t_req in times_to_plot:
        if abs(t_current - t_req) < TOL and t_req not in snapshots:
            snapshots[t_req] = T2D.copy()

    for k,(ix,iy) in enumerate(probe_pts):
        probes_hist[k].append(T2D[iy, ix])

    q_conv_line = h * (T2D[Ny-1, :] - T_inf)
    flux_mean.append(np.mean(q_conv_line))

    energy.append((rho*cp) * np.sum(T2D) * area_cell)

# -------------------------------------------------------------------------
# Visualisation réduite: seulement T aux snapshots
# -------------------------------------------------------------------------
rows = len(times_to_plot)
fig, axes = plt.subplots(1, rows, figsize=(4*rows, 4))

for col, t in enumerate(times_to_plot):
    T2D = snapshots[t]
    im = axes[col].imshow(T2D, origin='lower', extent=[0,Lx,0,Ly], cmap='inferno')
    axes[col].set_title(f"T à {t} s")
    axes[col].set_xlabel("x (m)")
    axes[col].set_ylabel("y (m)")
    plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Visualisation 2: flux convectif moyen & chaleur extraite
# -------------------------------------------------------------------------
time_arr = np.array(time)
E0 = energy[0]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(time_arr, flux_mean, color='tab:blue')
ax[0].set_title("Flux convectif moyen au bord haut")
ax[0].set_xlabel("Temps (s)")
ax[0].set_ylabel("q'' (W/m²)")
ax[0].grid(True, alpha=0.3)

ax[1].plot(time_arr, E0 - np.array(energy), color='tab:red')
ax[1].set_title("Chaleur extraite ΔE(t)")
ax[1].set_xlabel("Temps (s)")
ax[1].set_ylabel("Énergie (J)")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Visualisation 3: sondes de température
# -------------------------------------------------------------------------
plt.figure(figsize=(7, 4))
for series, label in zip(probes_hist, labels):
    plt.plot(time_arr, series, label=label)
plt.title("Évolution temporelle des sondes de température")
plt.xlabel("Temps (s)")
plt.ylabel("Température (K)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Résumé terminal
# -------------------------------------------------------------------------
T2D_final = T.reshape(Ny, Nx)
print("Température min/max finales (K):", float(T2D_final.min()), float(T2D_final.max()))
print("Flux convectif moyen final (W/m²):", float(flux_mean[-1]))
print("Chaleur extraite totale (J):", float(E0 - energy[-1]))

import numpy as np
import matplotlib.pyplot as plt

# Domain
Nx, Ny = 200, 64
H = Ny - 1

# D2Q9
c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
w = np.array([4/9] + [1/9]*4 + [1/36]*4)
cs2 = 1/3

# Parameters
tau = 0.8               # relaxation time -> nu = cs2*(tau-0.5)
nu = cs2*(tau - 0.5)
Fx = 1e-6               # body force (adjust for desired u_max)
steps = 20000

# Fields
rho = np.ones((Nx, Ny))
ux = np.zeros((Nx, Ny))
uy = np.zeros((Nx, Ny))

# Distributions
def feq(rho, ux, uy):
    f_eq = np.zeros((9, Nx, Ny))
    u2 = ux**2 + uy**2
    for i in range(9):
        cu = c[i,0]*ux + c[i,1]*uy
        f_eq[i] = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    return f_eq

f = feq(rho, ux, uy)

# Guo forcing terms
def forcing_term(rho, ux, uy, Fx, Fy=0.0):
    F = np.zeros((9, Nx, Ny))
    for i in range(9):
        ci = c[i]
        cu = ci[0]*ux + ci[1]*uy
        Fi = w[i]*(1 - 0.5/tau) * (
            3*(ci[0]-3*cu*ci[0])*Fx + 3*(ci[1]-3*cu*ci[1])*Fy
        )
        F[i] = rho * Fi
    return F

# Helpers for streaming
def stream(f):
    f_str = np.empty_like(f)
    for i in range(9):
        f_str[i] = np.roll(np.roll(f[i], c[i,0], axis=0), c[i,1], axis=1)
    return f_str

# Bounce-back on y=0 and y=Ny-1
bounce_pairs = {1:3, 3:1, 2:4, 4:2, 5:7, 7:5, 8:6, 6:8}
def bounce_back(f):
    # bottom wall y=0
    y = 0
    for i,j in bounce_pairs.items():
        f[i,:,y] = f[j,:,y]
    # top wall y=Ny-1
    y = Ny-1
    for i,j in bounce_pairs.items():
        f[i,:,y] = f[j,:,y]

# Main loop
for t in range(steps):
    # Macros
    rho = np.sum(f, axis=0)
    ux = (np.sum(f * c[:,0][:,None,None], axis=0) + 0.5*rho*Fx) / rho
    uy = (np.sum(f * c[:,1][:,None,None], axis=0)) / rho

    # Equilibrium and forcing
    feq_vals = feq(rho, ux, uy)
    Fg = forcing_term(rho, ux, uy, Fx)

    # Collision
    f = f + -(1.0/tau)*(f - feq_vals) + Fg

    # Streaming
    f = stream(f)

    # Periodic in x
    # (roll already applies periodicity; nothing else needed)

    # Bounce-back
    bounce_back(f)

    # Optional: check convergence every few steps
    if t % 2000 == 0:
        u_center = ux[:, Ny//2].mean()
        print(f"t={t}, u_center≈{u_center:.6e}")

# Extract profile
u_profile = ux[Nx//2, :]
y = np.arange(Ny)

# Analytical reference
u_analytical = (Fx/(2*nu)) * (y*(H - y))
err = np.sqrt(np.mean((u_profile - u_analytical)**2)) / (np.max(u_analytical)+1e-12)
print(f"Relative RMSE vs analytical: {err:.4e}")

# --- Profil de vitesse simulé vs analytique ---
plt.figure(figsize=(6,4))
plt.plot(y, u_profile, 'r-', label="LBM simulé")
plt.plot(y, u_analytical, 'k--', label="Analytique")
plt.xlabel("y (noeuds)")
plt.ylabel("Vitesse u_x")
plt.title("Profil de vitesse de Poiseuille")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Champ de vitesse 2D ---
plt.figure(figsize=(6,4))
plt.imshow(ux.T, origin='lower', cmap='jet', aspect='auto')
plt.colorbar(label="Vitesse u_x")
plt.title("Champ de vitesse u_x")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# --- Champ de pression (rho ~ pression réduite) ---
plt.figure(figsize=(6,4))
plt.imshow(rho.T, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label="Pression (rho)")
plt.title("Champ de pression")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

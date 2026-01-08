# Importations FEniCS et autres
import numpy as np
import dolfinx.fem.petsc
from dolfinx import mesh, fem, io, plot
from ufl import TrialFunction, TestFunction, grad, dx, inner, Identity, tr, sqrt, conditional
from mpi4py import MPI
import pyvista as pv
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1. DÉFINITION DU MATÉRIAU ET DU MAILLAGE
# -----------------------------------------------------------

# Paramètres du matériau
E = 200e9      # Module d'Young (Pa)
nu = 0.3       # Coefficient de Poisson
mu = E / (2 * (1 + nu)) # Deuxième paramètre de Lamé
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu)) # Premier paramètre de Lamé

# Fonction pour calculer le tenseur de contrainte de Hooke (Isotrope)
def sigma(epsilon):
    return lambda_ * tr(epsilon) * Identity(epsilon.geometric_dimension()) + 2 * mu * epsilon

# Fonction pour calculer le tenseur de déformation
def epsilon(u):
    return 0.5 * (grad(u) + grad(u.T))

# Création du maillage (rectangle)
Lx = 2.0  # Longueur de la plaque
Hy = 1.0  # Hauteur de la plaque
Nx = 40   # Nombre d'éléments selon X
Ny = 20   # Nombre d'éléments selon Y

domain = mesh.create_rectangle(MPI.COMM_WORLD, 
                              [[0.0, 0.0], [Lx, Hy]], [Nx, Ny], 
                              cell_type=mesh.CellType.quadrilateral)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
u = TrialFunction(V)
v = TestFunction(V)

# -----------------------------------------------------------
# 2. DÉFINITION DE LA FORME FAIBLE (Équation d'équilibre)
# -----------------------------------------------------------

a = inner(sigma(epsilon(u)), epsilon(v)) * dx

P = 1e6 # Pa (1 MPa)
f_traction = fem.Constant(domain, [P, 0.0])

def right_boundary(x):
    return np.isclose(x[0], Lx)

facet_dim = domain.topology.dim - 1
facet_tags = mesh.locate_entities_boundary(domain, facet_dim, right_boundary)
ft = mesh.meshtags(domain, facet_dim, facet_tags, np.full_like(facet_tags, 1))
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

L = inner(f_traction, v) * ds(1)

# -----------------------------------------------------------
# 3. CONDITIONS AUX LIMITES (Dirichlet)
# -----------------------------------------------------------

def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_facets = mesh.locate_entities_boundary(domain, facet_dim, left_boundary)
left_dofs = fem.locate_dofs_geometrical(V, left_boundary)

u_zero = np.array([0.0, 0.0], dtype=np.float64) # Utiliser np.float64 pour dolfinx
bc = fem.dirichletbc(u_zero, left_dofs, V)

# -----------------------------------------------------------
# 4. RÉSOLUTION DU SYSTÈME LINÉAIRE
# -----------------------------------------------------------

u_solution = fem.Function(V)
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], u=u_solution, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
print("Démarrage de la résolution...")
problem.solve()
print("Résolution terminée.")

# -----------------------------------------------------------
# 5. POST-TRAITEMENT ET VISUALISATION AVEC PYVISTA
# -----------------------------------------------------------

# --- 5.1 Préparation des données pour PyVista ---

# Conversion du maillage DOLFINx en un maillage PyVista
topology, cell_types, geometry = plot.create_vtk_mesh(domain, domain.topology.dim)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Associer le champ de déplacement aux nœuds du maillage PyVista
grid.point_data["u"] = u_solution.x.array.reshape(-1, domain.geometry.dim)

# --- 5.2 Calcul et Affichage de la Contrainte de Von Mises ---

# L'espace fonctionnel pour la contrainte (DG0: constante par élément)
V_DG0 = fem.functionspace(domain, ("DG", 0))

# Calcul des composantes du tenseur de contrainte
sigma_tensor = sigma(epsilon(u_solution))
sigma_xx_expr = fem.Expression(sigma_tensor[0, 0], V_DG0.element.interpolation_points())
sigma_yy_expr = fem.Expression(sigma_tensor[1, 1], V_DG0.element.interpolation_points())
tau_xy_expr = fem.Expression(sigma_tensor[0, 1], V_DG0.element.interpolation_points())

sigma_xx = fem.Function(V_DG0)
sigma_yy = fem.Function(V_DG0)
tau_xy = fem.Function(V_DG0)

projection_problem_xx = fem.petsc.L2Projection(sigma_xx_expr, V_DG0, fem.Constant(domain, 0.0))
projection_problem_yy = fem.petsc.L2Projection(sigma_yy_expr, V_DG0, fem.Constant(domain, 0.0))
projection_problem_xy = fem.petsc.L2Projection(tau_xy_expr, V_DG0, fem.Constant(domain, 0.0))

projection_problem_xx.project(sigma_xx)
projection_problem_yy.project(sigma_yy)
projection_problem_xy.project(tau_xy)

# Calcul de la contrainte de Von Mises sur l'espace DG0
# C'est une approximation car sigma_xx, sigma_yy, tau_xy sont des fonctions DG0
von_mises_values = np.sqrt(sigma_xx.x.array**2 + sigma_yy.x.array**2 - 
                           sigma_xx.x.array * sigma_yy.x.array + 
                           3 * tau_xy.x.array**2)

# Associer la contrainte de Von Mises aux cellules du maillage PyVista
grid.cell_data["Von Mises stress (Pa)"] = von_mises_values

# --- 5.3 Création des Graphiques avec PyVista ---

# Plotter pour afficher les résultats
plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
plotter.set_background("white")

# --- Subplot 1 : Déformation de la Plaque ---
plotter.subplot(0, 0)
plotter.add_title("Déformation de la Plaque (facteur d'amplification x500)", color='black')

# Maillage non déformé
plotter.add_mesh(grid, style='wireframe', color='blue', line_width=0.5, label="Maillage initial")

# Maillage déformé (avec un facteur d'amplification)
amplification_factor = 500.0
deformed_grid = grid.copy()
deformed_grid.points[:, :domain.geometry.dim] += grid.point_data["u"] * amplification_factor
plotter.add_mesh(deformed_grid, show_edges=True, edge_color='black', color='red', opacity=0.8,
                 scalars="Von Mises stress (Pa)", cmap="jet",
                 stitle="Von Mises (Pa) - Amplified Deformation")


# Ajout de l'encastrement et de la force
plotter.add_lines([[0,0,0],[0,Hy,0]], color='grey', line_width=5, label="Encastrement")
plotter.add_text(f"Force = {P/1e6:.1f} MPa", position="upper_right", color='black')

plotter.view_xy() # Vue 2D
plotter.enable_parallel_projection()


# --- Subplot 2 : Carte de Contraintes de Von Mises ---
plotter.subplot(0, 1)
plotter.add_title("Contrainte de Von Mises (Pa)", color='black')

# Afficher le maillage avec les contraintes de Von Mises
plotter.add_mesh(grid, show_edges=True, edge_color='grey',
                 scalars="Von Mises stress (Pa)", cmap="jet",
                 stitle="Von Mises (Pa)")

# Ajout de l'encastrement
plotter.add_lines([[0,0,0],[0,Hy,0]], color='grey', line_width=5)

plotter.view_xy() # Vue 2D
plotter.enable_parallel_projection()


# --- Affichage du Plotter ---
plotter.show()

print("\nLa simulation 2D est terminée et les résultats sont affichés dans une fenêtre PyVista.")
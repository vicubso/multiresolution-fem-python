# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: torch
#     language: python
#     name: python3
# ---

# %%
# %reset -f
import numpy as np
from plane_elasticity import Q4, MultiQ4, MultiQ4Grid
import matplotlib.pyplot as plt

# %% [markdown]
# ## Draw deformed cantilever

# %%
# Create grid
length_x = 12
length_y = 6
nel_x = 4
nel_y = 3
n_subel_x = 4
n_subel_y = 4
nu = 1 / 3
D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu**2)  # Plane stress constitutive matrix
E = np.ones((nel_x*nel_y, n_subel_x*n_subel_y))
grid = MultiQ4Grid(length_x, length_y, nel_x, nel_y, n_subel_x, n_subel_y, D, E)
u = np.zeros(grid.n_dofs)

left_edge_nodes = np.arange(0, nel_y+1)
right_edge_nodes = np.arange(grid.n_nodes-nel_y-1, grid.n_nodes)

# Apply boundary conditions
all_dofs = np.arange(grid.n_dofs)
fixed_dofs = np.sort(np.concatenate([[1], 2*left_edge_nodes]))
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# Apply force
f = np.zeros(grid.n_dofs)
f[2*right_edge_nodes+1] = -0.002 # Force on the right edge
# f[-1]= -0.01
f[fixed_dofs] = 0

# Get stiffness matrix
K = grid.get_stiffness_matrix(E)
K[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
K[np.ix_(fixed_dofs, free_dofs)] = 0
K[np.ix_(free_dofs, fixed_dofs)] = 0

# Solve
u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])

# Plot
grid.draw(u, E, draw_subelements=True)
plt.axis('equal');
plt.axis('off');
plt.savefig("grid_cantilever.pdf")



# %% [markdown]
# ## Study effect of more subelements in compliance

# %%
compliance = []
N = 20
for n_subel in range(1,N):    
    # Create grid
    length_x = 12
    length_y = 6
    nel_x = 4
    nel_y = 3
    n_subel_x = n_subel
    n_subel_y = n_subel
    nu = 1 / 3
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu**2)  
    E = np.ones((nel_x*nel_y, n_subel_x*n_subel_y))
    grid = MultiQ4Grid(length_x, length_y, nel_x, nel_y, n_subel_x, n_subel_y, D, E)
    u = np.zeros(grid.n_dofs)

    left_edge_nodes = np.arange(0, nel_y+1)
    right_edge_nodes = np.arange(grid.n_nodes-nel_y-1, grid.n_nodes)

    # Apply boundary conditions
    all_dofs = np.arange(grid.n_dofs)
    fixed_dofs = np.sort(np.concatenate([[1], 2*left_edge_nodes]))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Apply force
    f = np.zeros(grid.n_dofs)
    f[2*right_edge_nodes+1] = -0.01 # Force on the right edge
    f[fixed_dofs] = 0

    # Get stiffness matrix
    K = grid.get_stiffness_matrix(E)
    K[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
    K[np.ix_(fixed_dofs, free_dofs)] = 0
    K[np.ix_(free_dofs, fixed_dofs)] = 0

    # Solve
    u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])

    # Get compliance
    compliance.append(u.T @ K @ u)

# %%
plt.plot(np.arange(1,N), compliance, 'o-')
plt.xlabel('# subelements')
# y label with LaTeX
plt.ylabel('$u ^\intercal K u$')
plt.xticks(np.arange(1,N,2))
plt.grid()
plt.savefig('compliance_vs_n_subelements.pdf')



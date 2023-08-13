# %%
%reset -f
import numpy as np
from plane_elasticity import Q4, MultiQ4, MultiQ4Grid
import matplotlib.pyplot as plt

# %% [markdown]
# ## Draw deformed cantilever

# %%
# Create grid
length_x = 26
length_y = 13
nel_x = 26
nel_y = 13
n_subel_x = 2
n_subel_y = 2
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
f[2*right_edge_nodes+1] = -0.001 # Force on the right edge
# f[-1]= -0.01
f[fixed_dofs] = 0

# Get stiffness matrix
K = grid.get_stiffness_matrix(E, ml_condensation=True)
K[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
K[np.ix_(fixed_dofs, free_dofs)] = 0
K[np.ix_(free_dofs, fixed_dofs)] = 0

# Solve
# u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])
u = np.linalg.solve(K, f)

# Plot
grid.draw(u, E, draw_subelements=True)
plt.axis('equal');
plt.axis('off');
# plt.savefig("grid_cantilever_2_by_2_big_ml.pdf")


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
plt.ylabel('$u ^\intercal K u$')
plt.xticks(np.arange(1,N,2))
plt.grid()
# plt.savefig('compliance_vs_n_subelements.pdf')

# %%
## Study ml vs direct condensation 
compliance_direct = []
compliance_ml = []
N = 20
for n_elm in range(1,N):    
    # Create grid
    length_x = n_elm*2
    length_y = n_elm
    nel_x = n_elm*2
    nel_y = n_elm
    n_subel_x = 2
    n_subel_y = 2
    nu = 1 / 3
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu**2)  
    E = np.ones((nel_x*nel_y, n_subel_x*n_subel_y))
    grid = MultiQ4Grid(length_x, length_y, nel_x, nel_y, n_subel_x, n_subel_y, D, E)

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
    K_direct = grid.get_stiffness_matrix(E, ml_condensation=False)
    K_direct[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
    K_direct[np.ix_(fixed_dofs, free_dofs)] = 0
    K_direct[np.ix_(free_dofs, fixed_dofs)] = 0
    
    K_ml = grid.get_stiffness_matrix(E, ml_condensation=True)
    K_ml[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
    K_ml[np.ix_(fixed_dofs, free_dofs)] = 0
    K_ml[np.ix_(free_dofs, fixed_dofs)] = 0

    # Solve
    u_direct = np.zeros(grid.n_dofs)
    u_direct = np.linalg.solve(K_direct, f)
    u_ml = np.zeros(grid.n_dofs)
    u_ml = np.linalg.solve(K_ml, f)

    # Get compliance
    compliance_direct.append(u_direct.T @ K_direct @ u_direct)
    compliance_ml.append(u_ml.T @ K_ml @ u_ml)

# %%
plt.plot(np.arange(1,N), compliance_direct, 'o-', label='Direct condensation')
plt.plot(np.arange(1,N), compliance_ml, 'x-', label='NN condensation')
plt.xlabel('nelx = 2 * nely')
plt.ylabel('$u ^\intercal K u$')
plt.xticks(np.arange(1,N,2))
plt.legend()
plt.grid()
plt.savefig('compliance_ml_vs_direct.pdf')



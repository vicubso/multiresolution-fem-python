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
from plane_elasticity import Q4, MultiQ4Vertices, MultiQ4VerticesGrid
import matplotlib.pyplot as plt

# %% [markdown]
# ## Test a single super-element

# %%
nu = 1 / 3
D = (1/(1-nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
n = 4
E = np.ones((n*n, 2*2))
superelm = MultiQ4Vertices(n, n , np.array([[0,1], [0,0], [1,1], [1,0]]), D, linear_sides=False)
K = superelm.get_stiffness_matrix(np.ones(n*n))

all_dofs = np.arange(0, 2*(n+1)**2)
fixed_dofs = np.arange(0,2*(n+1)+1,2)
fixed_dofs = np.union1d(fixed_dofs, [1])
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
u = np.zeros(2*(n+1)**2)
f = np.zeros(2*(n+1)**2)
f[-1] = -0.001


K[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
K[np.ix_(fixed_dofs, free_dofs)] = 0
K[np.ix_(free_dofs, fixed_dofs)] = 0

u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])
# print(np.linalg.norm(K @ u - f))
# print(np.linalg.det(K[np.ix_(free_dofs, free_dofs)]))

superelm.draw(u)
# superelm.draw(np.zeros(2*(n+1)**2))
plt.axis('off');
plt.savefig("no_linear_bcs.pdf")


# %% [markdown]
# ## Draw deformed cantilever

# %%
def cantilever(length_x = 12, length_y = 6, nel_x = 6, nel_y = 3 ,n_subel_x = 2, n_subel_y = 2, F=-0.01, ml_condensation=False, linear_sides=False, plot=False, filename=None):
    # Create grid object
    nu = 1 / 3
    D = (1/(1-nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    E = np.ones((nel_x*nel_y, n_subel_x*n_subel_y))
    grid = MultiQ4VerticesGrid(length_x, length_y, nel_x, nel_y, n_subel_x, n_subel_y, D, E, linear_sides=linear_sides)
    u = np.zeros(grid.n_dofs)

    left_edge_nodes = np.arange(0, nel_y+1)
    right_edge_nodes = np.arange(grid.n_nodes-nel_y-1, grid.n_nodes)
    right_edge_mid_node = int((grid.n_nodes-nel_y-1 + grid.n_nodes)/2)

    # Apply boundary conditions
    all_dofs = np.arange(grid.n_dofs)
    fixed_dofs = np.sort(np.concatenate([[1], 2*left_edge_nodes]))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Apply force
    f = np.zeros(grid.n_dofs)
    f[2*right_edge_mid_node+1] = F
    f[fixed_dofs] = 0

    # Get stiffness matrix
    K = grid.get_stiffness_matrix(E, ml_condensation=ml_condensation)
    K[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(len(fixed_dofs))
    K[np.ix_(fixed_dofs, free_dofs)] = 0
    K[np.ix_(free_dofs, fixed_dofs)] = 0

    # Solve
    # u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])
    u = np.linalg.solve(K, f)

    if plot:
        # Plot
        grid.draw(u, E, draw_subelements=True, ml_condensation=ml_condensation)
        plt.axis('equal');
        plt.axis('off');
        if filename:
            plt.savefig(filename)


    return grid, K, f, u


# %%
n = 2
grid, K, f, u = cantilever(n,n,n,n,2,2,F = -0.005, ml_condensation=False, linear_sides=True, plot=True);
np.linalg.norm(u)

# %% [markdown]
# ## Study effect of more subelements in compliance

# %%
compliance = []
N = 20
for n_subel in range(1,N):
    grid, K, f, u = cantilever(1,1,1,1,n_subel,n_subel,F=-0.01)
    compliance.append(u.T @ K @ u)

# %%
plt.plot(np.arange(1,N), compliance, 'o-')
plt.xlabel('# subelements')
plt.ylabel('$u ^\intercal K u$')
plt.xticks(np.arange(1,N,2))
plt.grid()
# plt.savefig('compliance_vs_n_subelements.pdf')

# %% [markdown]
# ## Study ml vs direct condensation 

# %%
compliance_complete = []
compliance_direct = []
compliance_ml = []
N = 20
for n_elm in np.arange(1,N):
    grid, K, f, u = cantilever(2*n_elm, 2*n_elm, 2*n_elm, 2*n_elm, 1,1, F=-0.01, ml_condensation=False)
    # compliance_complete.append(u.T @ K @ u)
    compliance_complete.append(u.T @ f)
    grid, K, f, u = cantilever(n_elm, n_elm, n_elm, n_elm, 2, 2, F=-0.01, ml_condensation=False)
    # compliance_direct.append(u.T @ K @ u)
    compliance_direct.append(u.T @ f)
    grid, K, f, u = cantilever(n_elm, n_elm, n_elm, n_elm, 2, 2, F=-0.01, ml_condensation=True)
    # compliance_ml.append(u.T @ K @ u)
    compliance_ml.append(u.T @ f)

compliance_complete = np.array(compliance_complete)
compliance_direct = np.array(compliance_direct)
compliance_ml = np.array(compliance_ml)

compliance_complete = compliance_complete * compliance_direct[0] / compliance_complete[0]


# %%
plt.plot(np.arange(1,N), compliance_complete, '.-', label='Complete system')
plt.plot(np.arange(1,N), compliance_direct, '+-', label='Direct condensation')
plt.plot(np.arange(1,N), compliance_ml, 'x-', label='ML condensation')
plt.xlabel('nelx = nely')
plt.ylabel('$u ^\intercal K u$')
plt.xticks(np.arange(1,N))
plt.legend()
plt.grid()
# plt.savefig('compliance_ml_vs_direct.pdf')



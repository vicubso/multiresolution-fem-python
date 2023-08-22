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
#     display_name: science
#     language: python
#     name: python3
# ---

# %%
# %reset -f
import numpy as np
from plane_elasticity import Q4, MultiQ4Sides, MultiQ4Vertices
import matplotlib.pyplot as plt
import time

# %%
nel_x = 3
nel_y = 3
nodes = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
nu = 1/3
D = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
element = MultiQ4Sides(nel_x, nel_y, nodes, D)
E = np.ones(element.n_elm)

K_0 = element.get_stiffness_matrix(E)
K_aa, N = element.get_condensed_stiffness_matrix_and_numerical_shape_functions(E, ml_condensation=False)

all_dofs = np.arange(element.side_dofs.shape[0])
fixed_dofs = np.sort(np.concatenate([[1], np.arange(0,2*nel_y+1)]))
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

K_aa[np.ix_(fixed_dofs, fixed_dofs)] = np.eye(fixed_dofs.shape[0])
K_aa[np.ix_(fixed_dofs, free_dofs)] = 0
K_aa[np.ix_(free_dofs, fixed_dofs)] = 0

f_aa = np.zeros(all_dofs.shape)
f_aa[-1] = -0.001

u_aa = np.zeros(all_dofs.shape)

u_aa = np.linalg.solve(K_aa, f_aa)

u = N @ u_aa

element.draw(u)

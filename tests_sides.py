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
from itertools import product

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

# %%
nel_x = 2; nel_y = 2
n_subel_x = 3; n_subel_y = 3
# Only 3x3 superelements are supported for now

length_x = 1; length_y = 1

n_elm = nel_x * nel_y; # number of superelements
n_nodes = 0
n_nodes += (nel_x + 1) * (nel_y + 1) # nodes at the corners of super-elements
n_nodes += (n_subel_y-1) * nel_y * (nel_x+1) # nodes at the left and right sides of super-elements
n_nodes += (n_subel_x-1) * nel_x * (nel_y+1) # nodes at the top and bottom sides of super-elements
n_dofs = 2 * n_nodes; # number of degrees of freedom
element_matrix = np.arange(0, n_elm).reshape(nel_y, nel_x, order='F')
n_nodes_per_superelement = 4 + 2*(n_subel_x-1) + 2*(n_subel_y-1)
n_dofs_per_superelement = 2 * n_nodes_per_superelement

# Build connectivity matrix.
# For nel_x = 2, nel_y = 2, the connectivity matrix is:
# [[0,   1,  2,  3,  4,  5,  6,  7, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33],
#  [6,   7,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39],
#  [26, 27, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59],
#  [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 51, 58, 59, 60, 61, 62, 63, 64, 65]]

connectivity_matrix = np.zeros((n_elm, n_dofs_per_superelement))
connectivity_matrix[0,:] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33])
elm = 1
for ix, iy in product(range(1,nel_x), range(1,nel_y)):
   


    elm += 1

print(connectivity_matrix)







# %%

# %% [markdown]
# Write some python code that contructs the connectivity matrix of a nel_x by nel_y mesh of elements, each of which has 4 corner nodes and, n_subel_x-1 nodes on each left and right side, and n_subel_y-1 nodes on each upper and lower side. Both local and global numbering is done column-wise, starting from the upper-left corner. As an example, for nel_x = 2, nel_y = 2, n_subel_x = 3, n_subel_y = 3, the connectivity should give
# [[0,   1,  2,  3,  4,  5,  6,  7, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33],
# [6,   7,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39],
# [26, 27, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59],
# [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 51, 58, 59, 60, 61, 62, 63, 64, 65]].
#
# Simply write some code inside the for loop in the following code:
#
# ```
# import numpy as np
# from itertools import product
#
# nel_x = 2; nel_y = 2
# n_subel_x = 3; n_subel_y = 3
# length_x = 1; length_y = 1
#
# n_elm = nel_x * nel_y; # number of superelements
# n_nodes = 0
# n_nodes += (nel_x + 1) * (nel_y + 1) # nodes at the corners of super-elements
# n_nodes += (n_subel_y-1) * nel_y * (nel_x+1) # nodes at the left and right sides of super-elements
# n_nodes += (n_subel_x-1) * nel_x * (nel_y+1) # nodes at the top and bottom sides of super-elements
# n_dofs = 2 * n_nodes; # number of degrees of freedom
# element_matrix = np.arange(0, n_elm).reshape(nel_y, nel_x, order='F')
# n_nodes_per_superelement = 4 + 2*(n_subel_x-1) + 2*(n_subel_y-1)
# n_dofs_per_superelement = 2 * n_nodes_per_superelement
#
# # Build connectivity matrix.
# # For nel_x = 2, nel_y = 2, n_subel_x = 3, n_subel_y = 3, the connectivity matrix is:
# # [[0,   1,  2,  3,  4,  5,  6,  7, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33],
# #  [6,   7,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39],
# #  [26, 27, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59],
# #  [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 51, 58, 59, 60, 61, 62, 63, 64, 65]]
#
# connectivity_matrix = np.zeros((n_elm, n_dofs_per_superelement))
# elm = 0
# for ix, iy in product(range(nel_x), range(nel_y)):
#     connectivity_matrix[elm, 0:2*(n_subel_x+1)] = np.arange(0,2*(n_subel_y+1)) + 2*iy*n_subel_y + 2*ix*(nel_y-1 + (n_subel_y-1)*(nel_y+1) + (n_subel_x-1)*(nel_y+1)) # Left side of the superelement
#
#     elm += 1
#
# print(connectivity_matrix)
# ```
#
#

# %%
import numpy as np
from itertools import product

nel_x = 2; nel_y = 2
n_subel_x = 3; n_subel_y = 3
length_x = 1; length_y = 1

n_elm = nel_x * nel_y; # number of superelements
n_nodes = 0
n_nodes += (nel_x + 1) * (nel_y + 1) # nodes at the corners of super-elements
n_nodes += (n_subel_y-1) * nel_y * (nel_x+1) # nodes at the left and right sides of super-elements
n_nodes += (n_subel_x-1) * nel_x * (nel_y+1) # nodes at the top and bottom sides of super-elements
n_dofs = 2 * n_nodes; # number of degrees of freedom
element_matrix = np.arange(0, n_elm).reshape(nel_y, nel_x, order='F')
n_nodes_per_superelement = 4 + 2*(n_subel_x-1) + 2*(n_subel_y-1)
n_dofs_per_superelement = 2 * n_nodes_per_superelement

# Build connectivity matrix.
connectivity_matrix = np.zeros((n_elm, n_dofs_per_superelement))
elm = 0
for ix, iy in product(range(nel_x), range(nel_y)):
    # Left side of the superelement
    connectivity_matrix[elm, 0:2*(n_subel_x+1)] = np.arange(0,2*(n_subel_y+1)) + 2*iy*n_subel_y + 2*ix*(nel_y-1 + (n_subel_y-1)*(nel_y+1) + (n_subel_x-1)*(nel_y+1))
    
    # Right side of the superelement
    connectivity_matrix[elm, 2*(n_subel_x+1):4*(n_subel_x+1)] = np.arange(0,2*(n_subel_y+1)) + 2*iy*n_subel_y + 2*ix*(nel_y-1 + (n_subel_y-1)*(nel_y+1) + (n_subel_x-1)*(nel_y+1)) + 2*(nel_y-1 + (n_subel_y-1)*(nel_y+1))
    
    # Top side of the superelement
    connectivity_matrix[elm, 4*(n_subel_x+1):6*(n_subel_x+1)] = np.arange(0,2*(n_subel_x+1)) + 2*iy*n_subel_y + 2*ix*(nel_y-1 + (n_subel_y-1)*(nel_y+1) + (n_subel_x-1)*(nel_y+1)) + 2*((nel_y-1 + (n_subel_y-1)*(nel_y+1))*(ix+1))
    
    # Bottom side of the superelement
    connectivity_matrix[elm, 6*(n_subel_x+1):] = np.arange(0,2*(n_subel_x+1)) + 2*iy*n_subel_y + 2*ix*(nel_y-1 + (n_subel_y-1)*(nel_y+1) + (n_subel_x-1)*(nel_y+1)) + 2*((nel_y-1 + (n_subel_y-1)*(nel_y+1))*(ix+1)) - 2*(nel_x)
    
    elm += 1

print(connectivity_matrix)



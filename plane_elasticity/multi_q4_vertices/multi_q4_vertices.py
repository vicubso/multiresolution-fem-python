import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from ..q4 import Q4
from .condensation_net_2_by_2 import CondensationNet2by2
import os
import torch
from torch import inf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiQ4Vertices:
    """
    Multi-resolution Q4 element.
    Consists of nel_x by nel_y Q4 elements.
    Different Young modulus for each sub-element are allowed and specified by E.
    Both the full elemental stiffness matrix and the condensed stiffness matrix to the 4 vertices are available.

    Nodes and sub-elements are numbered column-wise, starting from north-west corner: 
        0 --- 3 --- 6
        |  0  |  2  |
        1 --- 4 --- 7
        |  1  |  3  |
        2 --- 5 --- 8

    Sub-element numbering: 

        0 --- 2
        |     |
        1 --- 3 

    Super-element numbering:

        0 --- 4 --- 8 --- 12
        |     |     |     |
        1 --- 5 --- 9 --- 13
        |     |     |     | 
        2 --- 6 --- 10--- 14
        |     |     |     |
        3 --- 7 --- 11--- 15

    Condensed super-element numbering

        0 --- 4 --- 6 --- 8
        |                 |
        1                 9
        |                 | 
        2                 10
        |                 |
        3 --- 5 --- 7 --- 11
    """
    # TODO: Think about:
    #    - Which properties should be considered intrisict to the element, and hence treated as attributes of the class?
    #       - E.g. should displacements be considered an attribute of the element?
    #       - Should boundary conditions be considered an attribute? If so, maybe we should construct the modified stiffness matrix [K_freefree, 0; 0, I] instead of the usual one
    #    - E should probably an input parameter to get_stiffness_matrix(), instead of an attribute of the class. This is because for TO, the Young modulus will change at each iteration.
    #    - Should the super-element have an array of sub-elements as attribute?
    # TODO: Include BC, solv, etc. in this class

    def __init__(self, nel_x, nel_y, nodes, D, linear_sides=False):
        """
        Constructor for the multiresolution Q4 element
        input:
            nel_x (int): number of sub-elements in x direction
            nel_y (int): number of sub-elements in y direction
            nodes (4x2 numpy.array): Coordinates of the 4 vertex nodes of the element. Ordered column-wise, starting from north-west corner: nw,sw,ne,se
                0 -- ... -- 2
                |     |     |
               ...   ...   ...
                |     |     |
                1 -- ... -- 3
            D (3x3 numpy.array) Constitutive matrix of the material, without the Young's modulus E.
                E.g., D = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        """
        self.nel_x = nel_x
        self.nel_y = nel_y
        self.nodes = nodes
        self.D = D
        (
            self.n_elm,
            self.n_nodes,
            self.n_dofs,
            self.element_matrix,
            self.node_matrix,
            self.connectivity_matrix,
            self.h_x,
            self.h_y,
            self.vertex_dofs,
            self.edge_dofs,
            self.inner_dofs
        ) = self.set_up_discretization()  # TODO: Matbe just have inside constructor?
        self.linear_sides = linear_sides

    def set_up_discretization(self):
        """
        Set up discretization of the super-element consisting of nel_x by nel_y Q4 elements. Used in constructor.
        """
        n_elm = self.nel_x * self.nel_y; # number of elements
        n_nodes = (self.nel_x + 1) * (self.nel_y + 1); # number of nodes
        n_dofs = 2 * n_nodes; # number of degrees of freedom
        element_matrix = np.arange(0, n_elm).reshape(self.nel_y, self.nel_x, order='F')
        node_matrix = np.arange(0, n_nodes).reshape(self.nel_y + 1, self.nel_x + 1, order='F') 
        c_vec = (2*node_matrix[0:-1, 0:-1] + 1).reshape(n_elm, 1, order='F') # Auxiliar vector to build connectivity matrix
        connectivity_matrix = c_vec + np.concatenate([[-1, 0, 1, 2], 2*self.nel_y+np.array([1, 2, 3, 4])]) # Connectivity matrix
        h_x = (np.max(self.nodes[:, 0]) - np.min(self.nodes[:, 0]))/self.nel_x # Length of sub-elements in x direction
        h_y = (np.max(self.nodes[:, 1]) - np.min(self.nodes[:, 1]))/self.nel_y # Length of sub-elements in y direction
        vertex_nodes = np.array([0, self.nel_y, n_nodes-self.nel_y-1, n_nodes-1])
        edge_nodes = np.arange(1,self.nel_y)
        edge_nodes = np.union1d(edge_nodes, np.arange(self.nel_y+1, n_nodes-self.nel_y-1, self.nel_y+1))
        edge_nodes = np.union1d(edge_nodes, np.arange(2*self.nel_y+1, n_nodes-self.nel_y-1, self.nel_y+1))
        edge_nodes = np.union1d(edge_nodes, np.arange(n_nodes-self.nel_y, n_nodes-1))
        inner_nodes = np.setdiff1d(np.arange(n_nodes), np.union1d(vertex_nodes, edge_nodes))
        vertex_dofs = np.vstack([2*vertex_nodes, 2*vertex_nodes+1]).T.flatten() # DOFs associated with the 4 vertices
        edge_dofs = np.vstack([2*edge_nodes, 2*edge_nodes+1]).T.flatten() # DOFs associated with the 4 edges
        inner_dofs = np.vstack([2*inner_nodes, 2*inner_nodes+1]).T.flatten() # DOFs associated with the inner nodes
        return n_elm, n_nodes, n_dofs, element_matrix, node_matrix, connectivity_matrix, h_x, h_y, vertex_dofs, edge_dofs, inner_dofs 
    
    def get_stiffness_matrix(self, E):
        """
        Compute stiffness matrix of the super-element consisting of nel_x by nel_y Q4 elements.
        Different Young modulus for each sub-element are allowed and specified by E.
        input:
            E (nel_x*nel_y x 1 numpy.array): Young's modulus of each sub-element
        output:
            K (n_dofs x n_dofs numpy.array): Stiffness matrix of the super-element
        """
        K = np.zeros((self.n_dofs, self.n_dofs))
        X = np.array([[0,0], [0,-self.h_y], [self.h_x,0], [self.h_x,-self.h_y]]) # All the subelements are equal, with width h_x and height h_y. Rigid translation doesn't matter. 
        q4 = Q4(X, self.D) # Q4 object
        Ke = q4.get_stiffness_matrix(1) # Sub-element stiffness matrix, with E=1
        for e in range(self.n_elm):
            dofs = self.connectivity_matrix[e, :] # Super-element DOFs associated with the element
            K[np.ix_(dofs, dofs)] += E[e] *  Ke # Add sub-element stiffness matrix to super-element stiffness matrix
        if self.linear_sides: # Impose linear BC on the sides
            nw = 0 # North-west corner
            sw = self.nel_y # South-west corner
            ne = self.nel_x * (self.nel_y + 1) # North-east corner
            se = self.nel_x * (self.nel_y + 1) + self.nel_y # South-east corner
            left = np.arange(nw+1, sw) # Left side
            right = np.arange(ne+1, se) # Right side
            upper = np.arange(nw+self.nel_y+1, ne, self.nel_y+1) # Upper side
            lower = np.arange(nw+2*self.nel_y+1, ne, self.nel_y+1) # Lower side
            Q = np.zeros((self.n_dofs,self.n_dofs)) # TODO: should be sparse
            for i in range(len(left)):
                K[2*left[i], 2*nw] = 1 - (i+1)/self.nel_y
                K[2*left[i]+1, 2*nw+1] = 1 - (i+1)/self.nel_y
                K[2*left[i], 2*sw] = (i+1)/self.nel_y
                K[2*left[i]+1, 2*sw+1] = (i+1)/self.nel_y
                K[2*right[i], 2*ne] = 1 - (i+1)/self.nel_y
                K[2*right[i]+1, 2*ne+1] = 1 - (i+1)/self.nel_y
                K[2*right[i], 2*se] = (i+1)/self.nel_y
                K[2*right[i]+1, 2*se+1] = (i+1)/self.nel_y
            for i in range(len(upper)):
                K[2*upper[i], 2*nw] = 1 - (i+1)/self.nel_x
                K[2*upper[i]+1, 2*nw+1] = 1 - (i+1)/self.nel_x
                K[2*upper[i], 2*ne] = (i+1)/self.nel_x
                K[2*upper[i]+1, 2*ne+1] = (i+1)/self.nel_x
                K[2*lower[i], 2*sw] = 1 - (i+1)/self.nel_x
                K[2*lower[i]+1, 2*sw+1] = 1 - (i+1)/self.nel_x
                K[2*lower[i], 2*se] = (i+1)/self.nel_x
                K[2*lower[i]+1, 2*se+1] = (i+1)/self.nel_x
            K[np.ix_(self.edge_dofs, self.vertex_dofs)] = Q[np.ix_(self.edge_dofs, self.vertex_dofs)]
            K[np.ix_(self.edge_dofs, self.edge_dofs)] = -np.eye(self.edge_dofs.size)
            # print(K[np.ix_(2*upper+1, self.vertex_dofs)])
        return K

    def get_auxiliary_condensation_matrix(self, E):
        K_0 = self.get_stiffness_matrix(E)
        a = self.vertex_dofs
        b = np.union1d(self.edge_dofs, self.inner_dofs)
        K_bb = K_0[np.ix_(b,b)]
        K_ab = K_0[np.ix_(a,b)]
        M = np.linalg.solve(K_bb, K_ab.T)
        return M

    def get_condensed_stiffness_matrix_and_numerical_shape_functions(self, E, ml_condensation=False):
        """
        Compute stiffness matrix of super-element, condensed to the 4 vertices, and corresponding numerical shape functions.
        These are done in the same method to avoid computing -K_bb\K_ab twice.
        input:
            E (nel_x*nel_y x 1 numpy.array): Young's modulus of each sub-element
        output:
            K_vertices (8 x 8 numpy.array): Stiffness matrix of the super-element, condensed to the 4 vertices
            N (n_dofs x 8 numpy.array): Numerical shape functions
        """
        K_0 = self.get_stiffness_matrix(E)
        a = self.vertex_dofs
        b = np.union1d(self.edge_dofs, self.inner_dofs)
        K_aa = K_0[np.ix_(a,a)]
        other_dofs = np.setdiff1d(np.arange(self.n_dofs), self.vertex_dofs)
        K_ab = K_0[np.ix_(a,b)]
        # The computation of K_bb\K_ab (which is expensive) is only done once and saved in matrix Aux.
        # Can be done with a neural network or with a direct solve.
        if ml_condensation:
            if self.linear_sides:
                raise ValueError("ML condensation with linear BCs not supported yet")
            net = self.get_condensation_net()
            Aux = net.forward(torch.tensor(E, dtype=torch.float32)).detach().numpy()
            Aux = Aux.reshape(self.other_dofs.size, self.vertex_dofs.size, order='F')
        else:
            Aux = self.get_auxiliary_condensation_matrix(E) 
        K_vertices = K_aa - K_ab @ Aux # Condensed stiffness matrix to the 4 vertices
        N = np.zeros((self.n_dofs, a.size)) # Numerical shape functions
        N[a, :] = np.eye(a.size)
        N[b, :] = - Aux
        return K_vertices, N

    def get_condensation_net(self):
        """
        Load the network in charge of approximating the auxiliary matrix K_bb\K_ab.
        """
        # TODO: Net should take height and width of superelement as input. For not it is hard-coded to length_x=length_y=1
        # ... or most likely there some sort of linear transformation involving the jacobian, so that I can reuse the same network.
        # TODO: For now, net is only trained on 2x2 super-elements. Train more nets
        if self.nel_x == 2 and self.nel_y == 2:
            net = CondensationNet2by2()
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights_2_by_2.pth')
            net.load_state_dict(torch.load(weights_path))
        else :
            raise ValueError("Only 2x2 super-elements are supported for now")
        return net

    def draw(self, u, axs=plt):
        """
        Plots a single super-element given all displacements
        input:
            u (n_dofx1 numpy.array): Nodal displacements
            axs (matplotlib.pyplot.axes): Axes object where to draw the element
        """
        # TODO: Modify to shade rectangle according to E
        X = np.array([[0,0], [0,-self.h_y], [self.h_x,0], [self.h_x,-self.h_y]])
        k = 0
        for i, j in product(range(self.nel_x), range(self.nel_y)): # Draw sub-elements
            x = self.nodes[0,0] + i*self.h_x # Upper left corner of sub-element
            y = self.nodes[0,1] - j*self.h_y 
            nodes = X + np.array([x,y]) # Nodes of sub-element
            displ = u[self.connectivity_matrix[k,:]].reshape((-1,2))  # Nodal displacements of sub-element
            nodes += displ # Displace nodes
            nodes = nodes[[0,1,3,2,0],:] # Reorder nodes to draw the sub-element
            axs.plot(nodes[:,0], nodes[:,1], "k.-.", linewidth=0.2, markersize=1);
            k += 1
        # Draw super-element
        displ = u[self.vertex_dofs].reshape((-1,2))
        nodes = self.nodes + displ # Displace nodes
        nodes = nodes[[0,1,3,2,0],:]
        axs.plot(nodes[:,0], nodes[:,1], "bo-", linewidth=.75, markersize=4);
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from .multi_q4 import MultiQ4
from .condensation_net_2_by_2 import CondensationNet2by2

class MultiQ4Grid:
    """
    Grid consisting of (nel_x by nel_y) MultiQ4 elements, each consisting of (n_subel_x by n_subel_y) Q4 subelements.
    Every one of the stiffness matrices of the MultiQ4 elements is condensed to the vertices.
    Since all the MultiQ4 elements are equal (except for the Young's modulus), only one is created, and the elemental stiffness matrices are created by calling the corresponding method of the MultiQ4 class.
    """
    # TODO: Maybe I can have a class just called "Grid", that can go as many levels of subdivision as needed
    # TODO: Include BC, solv, etc. in this class
    def __init__(self, length_x, length_y, nel_x, nel_y, n_subel_x, n_subel_y, D, E):
        """
        Constructor for the MultiQ4Grid class.
        Input:
        length_x (float): length of the domain in x direction
        length_y (float): length of the domain in y direction
        nel_x (int): number of MultiQ4 elements in x direction
        nel_y (int): number of MultiQ4 elements in y direction
        n_subel_x (int): number of subelements in x direction, for each MultiQ4 element
        n_subel_y (int): number of subelements in y direction, for each MultiQ4 element
        D (3x3 numpy.array) Constitutive matrix of the material, without the Young's modulus E.
            E.g., D = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        """
        self.length_x = length_x
        self.length_y = length_y
        self.nel_x = nel_x
        self.nel_y = nel_y
        self.n_subel_x = n_subel_x
        self.n_subel_y = n_subel_y
        self.D = D
        self.E = E
        ( 
            self.n_elm,
            self.n_nodes,
            self.n_dofs,
            self.element_matrix,
            self.node_matrix,
            self.connectivity_matrix,
            self.h_x,
            self.h_y,
        ) = self.set_up_discretization()

    def set_up_discretization(self):
        """
        Set up discretization of the grid. Used in constructor.
        """
        n_elm = self.nel_x * self.nel_y; # number of elements
        n_nodes = (self.nel_x + 1) * (self.nel_y + 1); # number of nodes
        n_dofs = 2 * n_nodes; # number of degrees of freedom
        element_matrix = np.arange(0, n_elm).reshape(self.nel_y, self.nel_x, order='F')
        node_matrix = np.arange(0, n_nodes).reshape(self.nel_y + 1, self.nel_x + 1, order='F') 
        c_vec = (2*node_matrix[0:-1, 0:-1] + 1).reshape(n_elm, 1, order='F') # Auxiliar vector to build connectivity matrix
        connectivity_matrix = c_vec + np.concatenate([[-1, 0, 1, 2], 2*self.nel_y+np.array([1, 2, 3, 4])]) # Connectivity matrix
        h_x = self.length_x/self.nel_x # Length of elements in x direction
        h_y = self.length_y/self.nel_y # Length of elements in y direction
        elements = [None] * n_elm # Array of n_elm MultiQ4() objects # TODO: not sure if needed
        return n_elm, n_nodes, n_dofs, element_matrix, node_matrix, connectivity_matrix, h_x, h_y 
    
    def get_stiffness_matrix(self, E, ml_condensation=False):
        """
        Returns the stiffness matrix of the grid.
        Input:
            E (nel_x*nel_y x n_subel_x*n_subel_y numpy.array): Young's modulus of each sub-element
        Output:
            K (n_dofs x n_dofs numpy.array): Stiffness matrix of the grid
        """
        K = np.zeros((self.n_dofs, self.n_dofs))
        elm_nodes = np.array([[0,0], [0,-self.h_y], [self.h_x,0], [self.h_x,-self.h_y]]) # Coordinates of the nodes of the sub-element. All sub-elements are equal. Translation doesn't matter
        element = MultiQ4(self.n_subel_x, self.n_subel_y, elm_nodes, self.D) 
        for elm in range(self.n_elm):
            Ke, _ = element.get_condensed_stiffness_matrix_and_numerical_shape_functions(E[elm,:], ml_condensation=ml_condensation)
            dofs = self.connectivity_matrix[elm,:]
            K[np.ix_(dofs, dofs)] += Ke
        return K

    def draw(self, u, E, axs=plt, draw_subelements=False, ml_condensation=False):
        """
        Draws the grid given the displacements.
        Input:
        u (n_dofs numpy.array): Displacements of the nodes
        axs (matplotlib.pyplot.axes): Axes object where to draw the grid
        """
        # TODO: Modify to shade rectangle according to E
        # TODO: Tidy up. The draw_subelements functionality should be part of the draw method in the MultiQ4 class
        if draw_subelements:
            elements = [None] * self.n_elm # Array of n_elm MultiQ4() objects # TODO: not sure if needed
            X = np.array([[0,0], [0,-self.h_y], [self.h_x,0], [self.h_x,-self.h_y]])
            k = 0
            for i,j in product(range(self.nel_x), range(self.nel_y)):
                x = i*self.h_x # Upper left corner of sub-element
                y = self.length_y - j*self.h_y 
                elm_nodes = X + np.array([x,y]) # Coordinates of the nodes of the sub-element
                elements[k] = MultiQ4(self.n_subel_x, self.n_subel_y, elm_nodes, self.D)
                k += 1
            for e in range(self.n_elm):
                element = elements[e] 
                u_vertices = u[self.connectivity_matrix[e,:]] # Vertex displacements of element
                _, N = element.get_condensed_stiffness_matrix_and_numerical_shape_functions(E[e,:], ml_condensation=ml_condensation)
                u_element = N @ u_vertices # Displacements of the nodes of the element
                element.draw(u_element, axs)
        if not draw_subelements:
            X = np.array([[0,0], [0,-self.h_y], [self.h_x,0], [self.h_x,-self.h_y]])
            k = 0
            for i, j in product(range(self.nel_x), range(self.nel_y)): # Draw sub-elements
                x = i*self.h_x # Upper left corner of sub-element
                y = self.length_y - j*self.h_y 
                nodes = X + np.array([x,y]) # Nodes of sub-element
                displ = u[self.connectivity_matrix[k,:]].reshape((-1,2))  # Nodal displacements of sub-element
                nodes += displ # Displace nodes
                nodes = nodes[[0,1,3,2,0],:] # Reorder nodes to draw the sub-element
                axs.plot(nodes[:,0], nodes[:,1], "bo-", linewidth=.75, markersize=3);
                k += 1
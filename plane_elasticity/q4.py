import numpy as np
import matplotlib.pyplot as plt

class Q4:
    """
    Class for a 4-node quadrilateral element.
    Sub-element nodes are numbered column-wise, starting from north-west corner: nw, sw, ne, se 
        0 --- 2
        |     |
        1 --- 3
    """

    def __init__(self, nodes, D):
        """
        Constructor for the Q4 element
        input:
            nodes (4x2 numpy.array): Coordinates of the 4 nodes of the element. Ordered column-wise, starting from north-west corner: nw, sw, ne, se
            D (3x3 numpy.array) Constitutive matrix of the material, without the Young's modulus E.
                E.g., D = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        """
        self.nodes = nodes
        self.D = D

    def isoparametric_shape_functions(self, natural_coordinates):
        """
        Shape functions of the isoparametric element
        input:
            natural_coordinates (1x2 numpy.array): Natural coordinates of the point of interest
        output:
            N (1x4 numpy.array): Shape functions evaluated at the point of interest
        """
        xi = natural_coordinates[0]
        eta = natural_coordinates[1]
        N = np.array([(1-xi)*(1+eta), (1-xi)*(1-eta), (1+xi)*(1+eta), (1+xi)*(1-eta)])/4
        return N

    def diff_isoparametric_shape_functions(self, natural_coordinates):
        """"
        Derivative of isoparametric shape functions
        input:
            natural_coordinates (1x2 numpy.array): Natural coordinates of the point of interest
        output:
            dN (2x4 numpy.array): Derivatives of shape functions evaluated at the point of interest
            dN_{i,j} = dN_j/d(xi)_i
        """
        xi = natural_coordinates[0]
        eta = natural_coordinates[1]
        dN = np.array([[-(1+eta), -(1-eta), (1+eta), (1-eta)],
                       [(1-xi), -(1-xi), (1+xi), -(1+xi)]])/4
        return dN

    def isoparametric_strain_displacement_matrix(self, natural_coordinates):
        """
        Strain displacement matrix B of an isoparametric Q4 element
        input:
            natural_coordinates (1x2 numpy.array): Natural coordinates of the point of interest
        output:
            B (3x8 numpy.array) strain displacement matrix evaluated at the point of interest
        """
        dN = self.diff_isoparametric_shape_functions(natural_coordinates)
        xi = natural_coordinates[0]
        eta = natural_coordinates[1]
        B = np.array([[dN[0,0] , 0      , dN[0,1], 0      , dN[0,2], 0      , dN[0,3], 0      ],
                    [0       , dN[1,0], 0      , dN[1,1], 0      , dN[1,2], 0      , dN[1,3]],
                    [dN[1,0] , dN[0,0], dN[1,1], dN[0,1], dN[1,2], dN[0,2], dN[1,3], dN[0,3]]])
        return B

    def jacobian(self, natural_coordinates):
        """
        Jacobian matrix J
        input:
            natural_coordinates (2x1 numpy.array): Natural coordinates of the point of interest
        output:
            J (2x2 numpy.array): Jacobian matrix evaluated at the point of interest
        """
        dN = self.diff_isoparametric_shape_functions(natural_coordinates)
        J = dN @ self.nodes
        return J

    def get_stiffness_matrix(self, E):
        """
        Compute elemental stiffness matrix
        input:

        output:
            K (8x8 numpy.array): Stiffness matrix of the element
            E (int): Young's modulus of the material
        """
        K = np.zeros((8,8))
        gauss_points = np.array([[-1,1],[-1,-1],[1,1],[1,-1]]) / np.sqrt(3)
        # D = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]) # Plane stress constitutive matrix
        for i in range(4):
            J = self.jacobian(gauss_points[i,:])
            detJ = np.linalg.det(J)
            B = self.isoparametric_strain_displacement_matrix(gauss_points[i,:])
            K += B.T @ self.D @ B * detJ * E
        # K = (K + K.T)/2 # Ensure symmetry # TODO: Is this necessary/correct? User might input non-symmetric D
        return K

    def draw(self, u, axs=plt):
        """
        Plots a single element given the nodes
        input:
            u (8 numpy.array): Nodal displacements
            axs (matplotlib.pyplot.axes): Axes object where to draw the element
        """
        # TODO: Modify to shade rectangle according to E
        nodes = self.nodes + u.reshape((4,2)) # Add displacements to nodes
        nodes = nodes[[0,1,3,2,0],:] # Repeat first node to close the element
        axs.plot(nodes[:,0], nodes[:,1], 'bo-', linewidth=1.0);
        return
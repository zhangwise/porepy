# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import warnings
import numpy as np
import scipy.sparse as sps

from porepy.params import tensor
from porepy.grids import grid

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

#from porepy.grids import grid, mortar_grid

from porepy.utils import comp_geom as cg

#------------------------------------------------------------------------------#

#class PrimalMassVEMMixedDim(SolverMixedDim):
#
#    def __init__(self, physics='flow'):
#        self.physics = physics
#
#        self.discr = PrimalVEM(self.physics)
#        self.discr_ndof = self.discr.ndof
#        self.coupling_conditions = None #PrimalCoupling(self.discr)
#
#        self.solver = Coupler(self.discr, self.coupling_conditions)

#------------------------------------------------------------------------------#

class PrimalMassVEM(Solver):

#------------------------------------------------------------------------------#

    def __init__(self, physics='flow'):
        self.physics = physics

#------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.  In
        this case number of nodes. If a mortar grid is given the number of dof
        are equal to the number of cells, we are considering an
        inter-dimensional interface with flux variable as mortars.

        Parameter
        ---------
        g: grid.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, grid.Grid):
            return g.num_nodes
#        elif isinstance(g, mortar_grid.MortarGrid):
#            return g.num_cells
        else:
            raise ValueError

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        """
        Return the matrix and righ-hand side for a discretization of a second
        order elliptic equation using primal virtual element method.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse csr (g.num_faces+g_num_cells, g.num_faces+g_num_cells)
            Matrix obtained from the discretization.
        rhs: array (g.num_faces+g_num_cells)
            Right-hand side which contains the boundary conditions.
        """
        M = self.matrix(g, data)
        return M, self.rhs(g, data)

#------------------------------------------------------------------------------#

    def matrix(self, g, data):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using primal virtual element method. See self.matrix_rhs for a detaild
        description.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            return sps.csr_matrix((self.ndof(g), self.ndof(g)))

        # Retrieve the permeability, boundary conditions, and aperture
        # The aperture is needed in the hybrid-dimensional case, otherwise is
        # assumed unitary
        param = data['param']
        bc = param.get_bc(self)
        a = param.get_aperture()

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        nodes_fn, _, _ = sps.find(g.face_nodes)

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = cg.map_grid(g)

        # In the virtual cell approach the cell diameters should involve the
        # apertures, however to keep consistency with the hybrid-dimensional
        # approach and with the related hypotheses we avoid.
        diams = g.cell_diameters()

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.sum(np.square(cell_nodes.indptr[1:]-cell_nodes.indptr[:-1]))
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        ws, xi = cg.simplex_reference_quadrature(g.dim)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            faces_loc = faces[loc]

            normals = sign[loc]*f_normals[:, faces_loc]

            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
            nodes_loc = nodes[loc]
            nodes_map = dict(zip(nodes_loc, np.arange(nodes_loc.size)))

            node_faces = np.zeros((nodes_loc.size, faces_loc.size), dtype=bool)
            for f_loc, f in enumerate(faces_loc):
                loc = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f+1])
                nodes_l = np.array([nodes_map[n] for n in nodes_fn[loc]])
                node_faces[nodes_l, f_loc] = True

            # Compute the mass-H1 local matrix
            A = self.massH1(a[c], c_centers[:, c],
                            g.cell_volumes[c], normals, diams[c],
                            node_coords[:, nodes_loc], node_faces, ws, xi)

            # Save values for H1-mass local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx+cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))

#------------------------------------------------------------------------------#

    def rhs(self, g, data):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using dual virtual element method. See self.matrix_rhs for a detaild
        description.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        return np.zeros(self.ndof(g))

#------------------------------------------------------------------------------#

    def massH1(self, apert, c_center, c_volume, normals, diam, coord,
               node_faces, ws, xi):
        """ Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        apert : ndarray (g.dim, g.dim)
            Aperture of the cell.
        c_center : array (g.dim)
            Cell center.
        c_volume : scalar
            Cell volume.
        normals : ndarray (g.dim, num_faces_of_cell)
            Normal of the cell faces weighted by the face areas.
        diam : scalar
            Diameter of the cell.
        weight : scalar
            weight for the stabilization term. Optional, default = 0.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name
        dim = coord.shape[0]
        num_nodes = coord.shape[1]
        num_mono = dim+1

        mono = np.r_[[lambda _: 1],
                     [lambda pt, i=i: (pt[i] - c_center[i])/diam\
                                                       for i in np.arange(dim)]]

        grad = np.block([[np.zeros(dim)], [np.eye(num_mono-1)/diam]])

        # local matrix D
        D = np.array([[m(c) for m in mono] for c in coord.T])

        # local matrix G
        Gtilde = np.zeros((num_mono, num_mono))
        Gtilde[1:, 1:] = np.dot(grad[1:, :], grad[1:, :].T)*c_volume
        G = Gtilde.copy()
        G[0, :] = D.sum(axis=0)/num_nodes

        # local matrix F
        F = np.zeros((num_mono, num_nodes))
        F[0, :] = np.ones(num_nodes)/num_nodes
        prod = np.dot(grad[1:, :], normals)/2.**(dim-1)

        # Loop on the monomials
        for i in np.arange(1, num_mono):
            for j in np.arange(num_nodes):
                F[i, j] = np.sum(prod[i-1, node_faces[j, :]])

        assert np.allclose(G, np.dot(F, D)),\
                                         "G\n"+str(G)+"\nF*D\n"+str(np.dot(F,D))

        # local matrix H
        H = np.zeros((num_mono, num_mono))
        H[0, 0] = c_volume

        pts, weights = cg.quadrature_rule(c_center, coord, node_faces, ws, xi)
        for i in np.arange(1, num_mono):
            for j in np.arange(1, num_mono):
                for pt, weight in zip(pts.T, weights):
                    H[i, j] += weight*mono[i](pt)*mono[j](pt)

        # local matrix Pi_s
        Pi_s = np.linalg.solve(G, F)
        I_Pi = np.eye(num_nodes) - np.dot(D, Pi_s)

        # local H1-mass matrix
        M = np.dot(Pi_s.T, np.dot(H, Pi_s)) + c_volume*np.dot(I_Pi.T, I_Pi)
        return apert*M

#------------------------------------------------------------------------------#

# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import warnings
import numpy as np
import scipy.sparse as sps

from porepy.params import tensor

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

#from porepy.grids import grid, mortar_grid

from porepy.utils import comp_geom as cg

#------------------------------------------------------------------------------#

class PrimalVEMMixedDim(SolverMixedDim):

    def __init__(self, physics='flow'):
        self.physics = physics

        self.discr = PrimalVEM(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = PrimalCoupling(self.discr)

        self.solver = Coupler(self.discr, self.coupling_conditions)

#------------------------------------------------------------------------------#

class PrimalVEM(Solver):

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
        elif isinstance(g, mortar_grid.MortarGrid):
            return g.num_cells
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
        M, bc_weight = self.matrix(g, data, bc_weight=True)
        return M, self.rhs(g, data, bc_weight)

#------------------------------------------------------------------------------#

    def matrix(self, g, data, bc_weight=False):
        """
        Return the matrix for a discretization of a second order elliptic equation
        using primal virtual element method. See self.matrix_rhs for a detaild
        description.

        Additional parameter:
        --------------------
        bc_weight: to compute the infinity norm of the matrix and use it as a
            weight to impose the boundary conditions. Default True.

        Additional return:
        weight: if bc_weight is True return the weight computed.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            M = sps.csr_matrix((self.ndof(g), self.ndof(g)))
            if bc_weight:
                return M, 1
            return M

        # Retrieve the permeability, boundary conditions, and aperture
        # The aperture is needed in the hybrid-dimensional case, otherwise is
        # assumed unitary
        param = data['param']
        k = param.get_tensor(self)
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

        if not data.get('is_tangential', False):
                # Rotate the permeability tensor and delete last dimension
                if g.dim < 3:
                    k = k.copy()
                    k.rotate(R)
                    remove_dim = np.where(np.logical_not(dim))[0]
                    k.perm = np.delete(k.perm, (remove_dim), axis=0)
                    k.perm = np.delete(k.perm, (remove_dim), axis=1)

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

            # Compute the siff-H1 local matrix
            A = self.stiffH1(a[c]*k.perm[0:g.dim, 0:g.dim, c], c_centers[:, c],
                             g.cell_volumes[c], normals, diams[c],
                             node_coords[:, nodes_loc], node_faces)

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx+cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        norm = sps.linalg.norm(M, np.inf) if bc_weight else 1

        # assign the Dirichlet boundary conditions
        if bc and np.any(bc.is_dir):
            nodes, _, _, = sps.find(g.face_nodes)

            dir_nodes = np.array([\
                         nodes[g.face_nodes.indptr[f]:g.face_nodes.indptr[f+1]]\
                        for f in np.where(bc.is_dir)[0]]).ravel()

            M[dir_nodes, :] *= 0
            M[dir_nodes, dir_nodes] = norm

        if bc_weight:
            return M, norm
        return M

#------------------------------------------------------------------------------#

    def rhs(self, g, data, bc_weight=1):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using dual virtual element method. See self.matrix_rhs for a detaild
        description.

        Additional parameter:
        --------------------
        bc_weight: to use the infinity norm of the matrix to impose the
            boundary conditions. Default 1.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        param = data['param']
        f = param.get_source(self)

        if g.dim == 0:
            return np.array(f)

        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        assert not bool(bc is None) != bool(bc_val is None)

        rhs = np.zeros(self.ndof(g))
        if bc is None:
            return rhs

            is_dir = np.where(bc.is_dir)[0]
            nodes, _, _, = sps.find(g.face_nodes)

        if np.any(bc.is_dir):
            size = np.power(g.dim, 2)*is_dir.size
            I = np.empty(size, dtype=np.int)
            J = np.empty(size, dtype=np.int)
            dataIJ = np.empty(size)
            idx = 0

            size_rhs = g.dim*is_dir.size
            data_rhs = np.empty(size_rhs)
            I_rhs = np.empty(size_rhs, dtype=np.int)
            idx_rhs = 0

            for f in is_dir:
                loc = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f+1])
                nodes_loc = nodes[loc]

                A = self.massH1(g.face_areas[f], g.dim-1)
                b = bc_weight*g.face_areas[f]*bc_val[f]/g.dim

                # Save values for H1-mass local matrix in the global structure
                cols = np.tile(nodes_loc, (nodes_loc.size, 1))
                loc_idx = slice(idx, idx+cols.size)
                I[loc_idx] = cols.T.ravel()
                J[loc_idx] = cols.ravel()
                dataIJ[loc_idx] = A.ravel()
                idx += cols.size

                loc_idx = slice(idx_rhs, idx_rhs+nodes_loc.size)
                I_rhs[loc_idx] = nodes_loc
                data_rhs[loc_idx] = b.ravel()
                idx_rhs += nodes_loc.size

            # Construct the global matrices
            M = sps.csr_matrix((dataIJ, (I, J)), shape=(rhs.size, rhs.size))
            identity = (M.sum(axis=1) == 0).astype(np.float).ravel()
            M += sps.diags(identity, offsets=[0], shape=M.shape)

            M_rhs = sps.csr_matrix((data_rhs, (I_rhs, np.zeros(I_rhs.size))),
                                    shape=(rhs.size, 1))

            rhs = sps.linalg.spsolve(M, M_rhs)

        return rhs

#------------------------------------------------------------------------------#

    def stiffH1(self, K, c_center, c_volume, normals, diam, coord, node_faces):
        """ Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
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

        prod = np.dot(grad[1:, :], normals)/2.**(dim-1)

        F = np.zeros((num_mono, num_nodes))
        F[0, :] = np.ones(num_nodes)/num_nodes

        # Loop on the monomials
        for i in np.arange(1, num_mono):
            for j in np.arange(num_nodes):
                F[i, j] = np.sum(prod[i-1, node_faces[j, :]])

        assert np.allclose(G, np.dot(F, D)), "G\n"+str(G)+"\nF*D\n"+str(np.dot(F,D))

        # local matrix Pi_s
        Pi_s = np.linalg.solve(G, F)
        I_Pi = np.eye(num_nodes) - np.dot(D, Pi_s)

        # local H1-stiff matrix
        return np.dot(Pi_s.T, np.dot(Gtilde, Pi_s)) + np.dot(I_Pi.T, I_Pi)

#------------------------------------------------------------------------------#

class PrimalCoupling(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, discr):
        self.discr_ndof = discr.ndof

#------------------------------------------------------------------------------#

    def matrix_rhs(self, matrix, g_h, g_l, data_h, data_l, data_edge):
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: dictionary which stores the data for the higher dimensional
                grid
            data_l: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.
        """
        # pylint: disable=invalid-name

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        g_m = data_edge['mortar_grid']
        dof, cc = self.create_block_matrix([g_h, g_l, g_m])

        # Recover the information for the grid-grid mapping
        faces_h, cells_h, sign_h = sps.find(g_h.cell_faces)
        ind_faces_h = np.unique(faces_h, return_index=True)[1]
        cells_h = cells_h[ind_faces_h]
        sign_h = sign_h[ind_faces_h]

        # Velocity degree of freedom matrix
        U = sps.diags(sign_h)

        shape = (g_h.num_cells, g_m.num_cells)
        hat_E_int = g_m.mortar_to_high_int()
        hat_E_int = sps.bmat([[U*hat_E_int], [sps.csr_matrix(shape)]])

        hat_P_avg = g_m.high_to_mortar_avg()
        check_P_avg = g_m.low_to_mortar_avg()

        cc[0, 2] = matrix[0, 0] * hat_E_int
        cc[2, 0] = hat_E_int.T * matrix[0, 0]
        cc[2, 2] = hat_E_int.T * matrix[0, 0] * hat_E_int

        # Mortar mass matrix
        inv_M = sps.diags(1./g_m.cell_volumes)

        # Normal permeability and aperture of the intersection
        inv_k = 1./(2.*data_edge['kn'])
        aperture_h = data_h['param'].get_aperture()

        # Inverse of the normal permability matrix
        Eta = sps.diags(np.divide(inv_k, hat_P_avg*aperture_h[cells_h]))

        matrix[2, 2] += inv_M*Eta

        A = check_P_avg.T
        shape = (g_l.num_faces, A.shape[1])
        cc[1, 2] = sps.bmat([[sps.csr_matrix(shape)], [A]])
        cc[2, 1] = cc[1, 2].T

        matrix += cc
        dof = np.where(hat_E_int.sum(axis=1).A.astype(np.bool))[0]
        norm = np.linalg.norm(matrix[0, 0].diagonal(), np.inf)
        matrix[0, 0][dof, :] *= 0
        matrix[0, 0][dof, dof] = norm
        matrix[0, 2][dof, :] *= 0

        return matrix

#------------------------------------------------------------------------------#

class PrimalCouplingDFN(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, discr_ndof):

        self.discr_ndof = discr_ndof

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        """
        Construct the matrix (and right-hand side) for the coupling conditions
        of a DFN. We use the Lagrange multiplier to impose continuity of the
        normal fluxes at the intersections.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: Not used but kept for consistency
            data_l: Not used but kept for consistency
            data: Not used but kept for consistency

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.
        """
        # pylint: disable=invalid-name

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])
        faces, cells_h, sign = sps.find(g_h.cell_faces)
        ind = np.unique(faces, return_index=True)[1]
        sign = sign[ind][faces_h]

        # Compute the off-diagonal terms
        dataIJ, I, J = sign, cells_l, faces_h
        cc[1, 0] = sps.csr_matrix((dataIJ, (I, J)), (dof[1], dof[0]))
        cc[0, 1] = cc[1, 0].T

        return cc

#------------------------------------------------------------------------------#

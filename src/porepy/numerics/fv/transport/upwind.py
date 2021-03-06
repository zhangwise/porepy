from __future__ import division
import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.mixed_dim.abstract_coupling import AbstractCoupling

#------------------------------------------------------------------------------#


class UpwindMixedDim(SolverMixedDim):

    def __init__(self, physics='transport'):
        self.physics = physics

        self.discr = Upwind(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = UpwindCoupling(self.discr)

        self.solver = Coupler(self.discr, self.coupling_conditions)

    def cfl(self, gb):
        deltaT = gb.apply_function(self.discr.cfl,
                                   self.coupling_conditions.cfl).data
        return np.amin(deltaT)

    def outflow(self, gb):
        def bind(g, d):
            return self.discr.outflow(g, d), np.zeros(g.num_cells)
        return Coupler(self.discr, solver_fct=bind).matrix_rhs(gb)[0]

#------------------------------------------------------------------------------#


class Upwind(Solver):
    """
    Discretize a hyperbolic transport equation using a single point upstream
    weighting scheme.


    """
#------------------------------------------------------------------------------#

    def __init__(self, physics='transport'):
        self.physics = physics

#------------------------------------------------------------------------------#

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (concentration dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data, d_name='discharge'):
        """
        Return the matrix and righ-hand side for a discretization of a scalar
        linear transport problem using the upwind scheme.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.
        Note: if not specified the inflow boundary conditions are no-flow, while
        the outflow boundary conditions are open.

        The name of data in the input dictionary (data) are:
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.
        source : array (g.num_cells) of source (positive) or sink (negative) terms.
        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        d_name: (string) keyword for data field in data containing the dischages

        Return
        ------
        matrix: sparse csr (g.num_cells, g_num_cells)
            Upwind matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Right-hand side which contains the boundary conditions.

        Examples
        --------
        data = {'discharge': u, 'bc': bnd, 'bc_val': bnd_val}
        advect = upwind.Upwind()
        U, rhs = advect.matrix_rhs(g, data)

        data = {'deltaT': advect.cfl(g, data)}
        M, _ = mass.MassMatrix().matrix_rhs(g, data)

        M_minus_U = M - U
        invM = mass.MassMatrix().inv(M)

        # Loop over the time
        for i in np.arange( N ):
            conc = invM.dot((M_minus_U).dot(conc) + rhs)

        """
        if g.dim == 0:
            return sps.csr_matrix([0]), [0]

        param = data['param']
        discharge = data[d_name]
        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        has_bc = not(bc is None or bc_val is None)

        # Compute the face flux respect to the real direction of the normals
        indices = g.cell_faces.indices
        flow_faces = g.cell_faces.copy()
        flow_faces.data *= discharge[indices]

        # Retrieve the faces boundary and their numeration in the flow_faces
        # We need to impose no-flow for the inflow faces without boundary
        # condition
        mask = np.unique(indices, return_index=True)[1]
        bc_neu = g.get_all_boundary_faces()

        if has_bc:
            # If boundary conditions are imposed remove the faces from this
            # procedure.
            # For primal-like discretizations, internal boundaries
            # are handled by assigning Neumann conditions.
            is_dir = np.logical_and(bc.is_dir, np.logical_not(bc.is_internal))
            bc_dir = np.where(is_dir)[0]
            bc_neu = np.setdiff1d(bc_neu, bc_dir, assume_unique=True)
            bc_dir = mask[bc_dir]

            # Remove Dirichlet inflow
            inflow = flow_faces.copy()

            inflow.data[bc_dir] = inflow.data[bc_dir].clip(max=0)
            flow_faces.data[bc_dir] = flow_faces.data[bc_dir].clip(min=0)

        # Remove all Neumann
        bc_neu = mask[bc_neu]
        flow_faces.data[bc_neu] = 0

        # Determine the outflow faces
        if_faces = flow_faces.copy()
        if_faces.data = np.sign(if_faces.data)

        # Compute the inflow/outflow related to the cells of the problem
        flow_faces.data = flow_faces.data.clip(min=0)

        flow_cells = if_faces.transpose() * flow_faces
        flow_cells.tocsr()

        if not has_bc:
            return flow_cells, np.zeros(g.num_cells)

        # Impose the boundary conditions
        bc_val_dir = np.zeros(g.num_faces)
        if np.any(bc.is_dir):
            is_dir = np.where(bc.is_dir)[0]
            bc_val_dir[is_dir] = bc_val[is_dir]

        # We assume that for Neumann boundary condition a positive 'bc_val'
        # represents an outflow for the domain. A negative 'bc_val' represents
        # an inflow for the domain.
        bc_val_neu = np.zeros(g.num_faces)
        if np.any(bc.is_neu):
            is_neu = np.where(bc.is_neu)[0]
            bc_val_neu[is_neu] = bc_val[is_neu]

        return flow_cells, - inflow.transpose() * bc_val_dir \
            - np.abs(g.cell_faces.transpose()) * bc_val_neu

#------------------------------------------------------------------------------#

    def cfl(self, g, data, d_name='discharge'):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.
        d_name: (string) keyword for dischagre file in data dictionary

        Return
        ------
        deltaT: time step according to CFL condition.

        """
        if g.dim == 0:
            return np.inf
        # Retrieve the data, only "discharge" is mandatory
        param = data['param']
        discharge = data[d_name]
        aperture = param.get_aperture()
        phi = param.get_porosity()

        faces, cells, _ = sps.find(g.cell_faces)

        # Detect and remove the faces which have zero in discharge
        not_zero = ~np.isclose(np.zeros(faces.size), discharge[faces], atol=0)
        if not np.any(not_zero):
            return np.inf

        cells = cells[not_zero]
        faces = faces[not_zero]

        # Compute discrete distance cell to face centers
        dist_vector = g.face_centers[:, faces] - g.cell_centers[:, cells]
        # Element-wise scalar products between the distance vectors and the
        # normals
        dist = np.einsum('ij,ij->j', dist_vector, g.face_normals[:, faces])
        # Since discharge is multiplied by the aperture, we get rid of it!!!!
        # Additionally we consider the phi (porosity) and the cell-mapping
        coeff = (aperture * phi)[cells]
        # deltaT is deltaX/discharge with coefficient
        return np.amin(np.abs(np.divide(dist, discharge[faces])) * coeff)

#------------------------------------------------------------------------------#

    def discharge(self, g, beta, cell_apertures=None):
        """
        Return the normal component of the velocity, for each face, weighted by
        the face area and aperture.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        beta: (3x1) array which represents the constant velocity.
        cell_apertures: (g.num_faces) array of apertures

        Return
        ------
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        """
        if cell_apertures is None:
            face_apertures = np.ones(g.num_faces)
        else:
            face_apertures = abs(g.cell_faces) * cell_apertures
            r, _, _ = sps.find(g.cell_faces)
            face_apertures = face_apertures / np.bincount(r)

        beta = np.asarray(beta)
        assert beta.size == 3

        if g.dim == 0:
            dot_prod = np.dot(g.face_normals.ravel('F'), face_apertures * beta)
            return np.atleast_1d(dot_prod)

        return np.array([np.dot(n, a * beta)
                         for n, a in zip(g.face_normals.T, face_apertures)])

#------------------------------------------------------------------------------#

    def outflow(self, g, data, d_name='discharge'):
        if g.dim == 0:
            return sps.csr_matrix([0])

        param = data['param']
        discharge = data[d_name]
        bc = param.get_bc(self)
        bc_val = param.get_bc_val(self)

        has_bc = not(bc is None or bc_val is None)

        # Compute the face flux respect to the real direction of the normals
        indices = g.cell_faces.indices
        flow_faces = g.cell_faces.copy()
        flow_faces.data *= discharge[indices]

        # Retrieve the faces boundary and their numeration in the flow_faces
        # We need to impose no-flow for the inflow faces without boundary
        # condition
        mask = np.unique(indices, return_index=True)[1]
        bc_neu = g.tags['domain_boundary_faces'].nonzero()[0]

        if has_bc:
            # If boundary conditions are imposed remove the faces from this
            # procedure.
            bc_dir = np.where(bc.is_dir)[0]
            bc_neu = np.setdiff1d(bc_neu, bc_dir, assume_unique=True)
            bc_dir = mask[bc_dir]

            # Remove Dirichlet inflow
            inflow = flow_faces.copy()

            inflow.data[bc_dir] = inflow.data[bc_dir].clip(max=0)
            flow_faces.data[bc_dir] = flow_faces.data[bc_dir].clip(min=0)

        # Remove all Neumann
        bc_neu = mask[bc_neu]
        flow_faces.data[bc_neu] = 0

        # Determine the outflow faces
        if_faces = flow_faces.copy()
        if_faces.data = np.sign(if_faces.data)

        outflow_faces = if_faces.indices[if_faces.data > 0]
        domain_boundary_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        outflow_faces = np.intersect1d(outflow_faces,
                                       domain_boundary_faces,
                                       assume_unique=True)

        # va tutto bene se ho neumann omogeneo
        # gli outflow sono positivi

        if_outflow_faces = if_faces.copy()
        if_outflow_faces.data[:] = 0
        if_outflow_faces.data[np.in1d(if_faces.indices, outflow_faces)] = 1

        if_outflow_cells = if_outflow_faces.transpose() * flow_faces
        if_outflow_cells.tocsr()

        return if_outflow_cells

#------------------------------------------------------------------------------#


class UpwindCoupling(AbstractCoupling):

#------------------------------------------------------------------------------#

    def __init__(self, discr):
        self.discr_ndof = discr.ndof

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge, d_name='discharge'):
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

        # Normal component of the velocity from the higher dimensional grid
        discharge = data_edge[d_name]

        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        dof, cc = self.create_block_matrix(g_h, g_l)

        # 1d-1d
        if g_h.dim == g_l.dim:
            # Remember that face_cells are really cell-cell connections
            # in this case
            cells_l, cells_h = data_edge['face_cells'].nonzero()
            diag_cc11 = np.zeros(g_l.num_cells)
            diag_cc00 = np.zeros(g_h.num_cells)
            d_00 = np.bincount(cells_h, np.sign(discharge.clip(min=0)) * discharge,
                               minlength=g_h.num_cells)
            d_11 = np.bincount(cells_l, np.sign(discharge.clip(max=0)) * discharge,
                               minlength=g_l.num_cells)
            np.add.at(diag_cc00, range(g_h.num_cells), d_00)
            np.add.at(diag_cc11, range(g_l.num_cells), d_11)
        else:
            # Recover the information for the grid-grid mapping
            cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

            # Recover the correct sign for the velocity
            faces, _, sgn = sps.find(g_h.cell_faces)
            sgn = sgn[np.unique(faces, return_index=True)[1]]
            discharge = sgn[faces_h] * discharge[faces_h]

            # Determine which are the corresponding cells of the faces_h
            cell_faces_h = g_h.cell_faces.tocsr()[faces_h, :]
            cells_h = cell_faces_h.nonzero()[1]

            diag_cc11 = np.zeros(g_l.num_cells)
            np.add.at(diag_cc11, cells_l, np.sign(
                discharge.clip(max=0)) * discharge)

            diag_cc00 = np.zeros(g_h.num_cells)
            np.add.at(diag_cc00, cells_h, np.sign(
                discharge.clip(min=0)) * discharge)
        # Compute the outflow from the higher to the lower dimensional grid
        cc[1, 0] = sps.coo_matrix((-discharge.clip(min=0), (cells_l, cells_h)),
                                  shape=(dof[1], dof[0]))

        # Compute the inflow from the higher to the lower dimensional grid
        cc[0, 1] = sps.coo_matrix((discharge.clip(max=0), (cells_h, cells_l)),
                                  shape=(dof[0], dof[1]))

        cc[1, 1] = sps.dia_matrix((diag_cc11, 0), shape=(dof[1], dof[1]))

        cc[0, 0] = sps.dia_matrix((diag_cc00, 0), shape=(dof[0], dof[0]))

        if data_h['node_number'] == data_l['node_number']:
            # All contributions to be returned to the same block of the
            # global matrix in this case
            cc = np.array([np.sum(cc, axis=(0, 1))])
        return cc

#------------------------------------------------------------------------------#

    def cfl(self, g_h, g_l, data_h, data_l, data_edge, d_name='discharge'):
        """
        Return the time step according to the CFL condition.
        Note: the vector field is assumed to be given as the normal velocity,
        weighted with the face area, at each face.

        The name of data in the input dictionary (data) are:
        discharge : array (g.num_faces)
            Normal velocity at each face, weighted by the face area.

        Parameters:
            g_h: grid of higher dimension
            g_l: grid of lower dimension
            data_h: dictionary which stores the data for the higher dimensional
                grid
            data_l: dictionary which stores the data for the lower dimensional
                grid
            data: dictionary which stores the data for the edges of the grid
                bucket

        Return:
            deltaT: time step according to CFL condition.

        """
        # Retrieve the discharge, which is mandatory
        discharge = data_edge[d_name]
        aperture_h = data_h['param'].get_aperture()
        aperture_l = data_l['param'].get_aperture()
        phi_l = data_l['param'].get_porosity()
        if g_h.dim == g_l.dim:
            # More or less same as below, except we have cell_cells in the place
            # of face_cells (see grid_bucket.duplicate_without_dimension).
            phi_h = data_h['param'].get_porosity()
            cells_l, cells_h = data_edge['face_cells'].nonzero()
            not_zero = ~np.isclose(
                np.zeros(discharge.shape), discharge, atol=0)
            if not np.any(not_zero):
                return np.Inf

            diff = g_h.cell_centers[:, cells_h] - g_l.cell_centers[:, cells_l]
            dist = np.linalg.norm(diff, 2, axis=0)

            # Use minimum of cell values for convenience
            phi_l = phi_l[cells_l]
            phi_h = phi_h[cells_h]
            apt_h = aperture_h[cells_h]
            apt_l = aperture_l[cells_l]
            coeff = np.minimum(phi_h, phi_l) * np.minimum(apt_h, apt_l)
            return np.amin(np.abs(np.divide(dist, discharge)) * coeff)

        # Recover the information for the grid-grid mapping
        cells_l, faces_h, _ = sps.find(data_edge['face_cells'])

        # Detect and remove the faces which have zero in "discharge"
        not_zero = ~np.isclose(np.zeros(faces_h.size),
                               discharge[faces_h], atol=0)
        if not np.any(not_zero):
            return np.inf

        cells_l = cells_l[not_zero]
        faces_h = faces_h[not_zero]
        # Mapping from faces_h to cell_h
        cell_faces_h = g_h.cell_faces.tocsr()[faces_h, :]
        cells_h = cell_faces_h.nonzero()[1][not_zero]
        # Retrieve and map additional data
        aperture_h = aperture_h[cells_h]
        aperture_l = aperture_l[cells_l]
        phi_l = phi_l[cells_l]
        # Compute discrete distance cell to face centers for the lower
        # dimensional grid
        dist = 0.5 * np.divide(aperture_l, aperture_h)
        # Since discharge is multiplied by the aperture, we get rid of it!!!!
        discharge = np.divide(discharge[faces_h],
                              g_h.face_areas[faces_h] * aperture_h)
        # deltaT is deltaX/discharge with coefficient
        return np.amin(np.abs(np.divide(dist, discharge)) * phi_l)

#------------------------------------------------------------------------------#

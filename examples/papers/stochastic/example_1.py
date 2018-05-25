import numpy as np
import scipy.sparse as sps
import os
import sys

import porepy as pp
from porepy import cg
from select_networks import select_networks

#------------------------------------------------------------------------------#

def add_data_darcy(gb, domain, tol):
    gb.add_node_props(['param', 'is_tangent'])

    apert = 1e-2

    km = 7.5 * 1e-11
    kf_t = 1e5 * km # low 1e-5 * km - high 1e5 * km
    kf_n = kf_t

    for g, d in gb:
        param = pp.Parameters(g)

        rock = g.dim == gb.dim_max()
        kxx = (km if rock else kf_t) * np.ones(g.num_cells)
        d['is_tangential'] = True
        if rock:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)

        param.set_tensor("flow", perm)

        param.set_source("flow", np.zeros(g.num_cells))

        param.set_aperture(np.power(apert, gb.dim_max() - g.dim))

        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain['ymax'] - tol
            bottom = bound_face_centers[1, :] < domain['ymin'] + tol
            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol
            boundary = np.logical_or(left, right)

            labels = np.array(['neu'] * bound_faces.size)
            labels[boundary] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = 30 * 1e6

            param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    # Assign coupling permeability
    gb.add_edge_props('kn')
    for e, d in gb.edges():
        g = gb.nodes_of_edge(e)[0]
        d['kn'] = kf_n / gb.node_props(g, 'param').get_aperture()

#------------------------------------------------------------------------------#


def add_data_advection(gb, domain, deltaT, tol):

    gb.add_node_props('deltaT')
    for g, d in gb:
        param = d['param']
        d['deltaT'] = deltaT

        rock = g.dim == gb.dim_max()
        source = np.zeros(g.num_cells)
        param.set_source("transport", source)

        if rock:
            param.set_porosity(0.3*np.ones(g.num_cells))
        else:
            param.set_porosity(np.ones(g.num_cells))

        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain['ymax'] - tol
            bottom = bound_face_centers[1, :] < domain['ymin'] + tol
            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol
            boundary = np.logical_or(left, right)
            labels = np.array(['neu'] * bound_faces.size)
            labels[boundary] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = 1

            param.set_bc("transport", pp.BoundaryCondition(
                g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", pp.BoundaryCondition(
                g, np.empty(0), np.empty(0)))
        d['param'] = param

    # Assign coupling discharge
    gb.add_edge_props('param')
    for e, d in gb.edges():
        g = gb.nodes_of_edge(e)[1]
        d['discharge'] = gb.node_props(g, 'discharge')

#------------------------------------------------------------------------------#

def import_mesh(network, tol, ttol, mesh_kwargs):
    pts, edges = pp.importer.lines_from_csv(network)
    R = cg.rot(-2*np.pi/4.5, [0, 0, 1.])[0:2, 0:2]
    pts = np.dot(R, pts)

    pts = cg.snap_points_to_segments(pts, edges, tol=ttol)
    pts, edges = cg.remove_edge_crossings(pts, edges, tol=tol)

    # Ensure unique description of points
    pts, _, old_2_new = pp.utils.setmembership.unique_columns_tol(pts, tol=ttol)
    edges[:2] = old_2_new[edges[:2]]
    to_remove = np.where(edges[0, :] == edges[1, :])[0]
    edges = np.delete(edges, to_remove, axis=1)

    f_set = np.array([pts[:, e] for e in edges.T])

    domain = cg.bounding_box(pts)
    return pp.meshing.simplex_grid(f_set, domain, **mesh_kwargs), domain

#------------------------------------------------------------------------------#

def main(export_folder='result', network='network.csv', tol=1e-4):
    T = 10 * np.pi * 1e7 # low 60 - high 10
    Nt = 60*24
    deltaT = T / Nt
    export_every = 12
    if_coarse = True

    h_f = 40
    h_b = 200
    h_min = 2
    mesh_kwargs = {'mesh_size_frac': h_f,
                   'mesh_size_bound': h_b,
                   'mesh_size_min': h_min,
                   'tol': tol}

    gb, domain = import_mesh(network, tol, 1e2*tol, mesh_kwargs)
    gb.compute_geometry()

    if if_coarse:
        pp.coarsening.coarsen(gb, 'by_volume')
    gb.assign_node_ordering()

    # Choose and define the solvers and coupler
    darcy = pp.DualVEMMixedDim("flow")

    # Assign parameters
    add_data_darcy(gb, domain, tol)

    up = sps.linalg.spsolve(*darcy.matrix_rhs(gb))
    darcy.split(gb, "up", up)

    gb.add_node_props(['pressure', "P0u", "discharge"])
    darcy.extract_u(gb, "up", "discharge")
    darcy.extract_p(gb, "up", 'pressure')
    darcy.project_u(gb, "discharge", "P0u")

    save = pp.Exporter(gb, "darcy", folder=export_folder)
    save.write_vtk(['pressure', "P0u"])

    # compute the flow rate
    total_flow_rate = 0
    for g, d in gb:
        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]
            left = bound_face_centers[0, :] < domain['xmin'] + tol
            flow_rate = d['discharge'][bound_faces[left]]
            total_flow_rate += np.sum(flow_rate)

    #################################################################

    advection = pp.UpwindMixedDim('transport')
    mass = pp.MassMatrixMixedDim('transport')
    invMass = pp.InvMassMatrixMixedDim('transport')

    # Assign parameters
    add_data_advection(gb, domain, deltaT, tol)

    U, rhs = advection.matrix_rhs(gb)
    M, _ = mass.matrix_rhs(gb)
    OF = advection.outflow(gb)

    # Perform an LU factorization to speedup the solver
    IE_solver = sps.linalg.factorized((M+U).tocsc())

    theta = np.zeros(rhs.shape[0])

    # Loop over the time
    time = np.empty(Nt)
    i_export = 0
    step_to_export = np.empty(0)

    production = np.zeros(Nt)
    save.change_name('theta')

    for i in np.arange(Nt):
        # Update the solution
        production[i] = np.sum(OF.dot(theta)) / total_flow_rate
        theta = IE_solver(M.dot(theta) + rhs)

        if i % export_every == 0:
            advection.split(gb, "theta", theta)

            save.write_vtk(["theta"], i_export)
            step_to_export = np.r_[step_to_export, i]
            i_export += 1

    save.write_pvd(step_to_export*deltaT)

    times = deltaT * np.arange(Nt)
    np.savetxt(export_folder + '/production.txt', (times, np.abs(production)),
               delimiter=',')

if __name__ == "__main__":
    base = 'conductive_result_'

    export_folder = base+'original/'
    network = './networks/original/original_porepy.csv'
    main(export_folder, network)

    networks = select_networks('./networks/')
    for name, network, network_topo in networks.T:
        print("processing "+name)
        export_folder = base+name+'/'
        main(export_folder, network)
        print("processing "+name+" topo")
        export_folder = base+name+'_topo/'
        main(export_folder, network_topo)

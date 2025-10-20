"""
##############################################################################
# Parallel Partitioned Multi-Physics Simulation Framework (ParaSiF)          #
#                                                                            #
# Copyright (C) 2025 The ParaSiF Development Team                            #
# All rights reserved                                                        #
#                                                                            #
# This software is licensed under the GNU General Public License version 3   #
#                                                                            #
# ** GNU General Public License, version 3 **                                #
#                                                                            #
# This program is free software: you can redistribute it and/or modify       #
# it under the terms of the GNU General Public License as published by       #
# the Free Software Foundation, either version 3 of the License, or          #
# (at your option) any later version.                                        #
#                                                                            #
# This program is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of             #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
# GNU General Public License for more details.                               #
#                                                                            #
# You should have received a copy of the GNU General Public License          #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.      #
##############################################################################

    @file thermalStructure.py

    @author W. Liu

    @brief FEniCSx transient heat conduction solver for conjugate heat transfer
           coupling with backward Euler time stepping.

"""

# -------------------------
#%% Import packages
# -------------------------

import numpy as np
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem
from scipy.spatial import cKDTree

comm = MPI.COMM_WORLD

# -------------------------
#%% Control parameters
# -------------------------

quiet = True  # define quiet mode

t = 0.0  # initial time
T = 1.0  # final time
num_steps = 200 # number of time steps

kappa_val = 1.0  # thermal conductivity

nx, ny, nz = 20, 10, 10  # mesh divisions in x,y,z directions
p_min = np.array([-2.0, -1.0, -1.0], dtype=np.float64)  # domain min corner
p_max = np.array([ 2.0,  1.0,  1.0], dtype=np.float64)  # domain max corner

poly_order = 1 # finite element polynomial order

COUPLED_MARK = 88  # marker for coupled boundary facets

# Define the coupled boundary: left face at x = -2.0
def coupled_boundary(x):
    return np.isclose(x[0], -2.0)

T0 = 0.0 # baseline temperature
T_L = 500.0  # initial left boundary temperature

def line_mask(dof_coords_output): # mask for dofs along a 1-D line for ASCII output
    return (np.isclose(dof_coords_output[:, 1], 0.0, atol=1e-8) & 
            np.isclose(dof_coords_output[:, 2], 0.0, atol=1e-8))

xdmf_filename = "fenicsx_solid_coupled.xdmf" # XDMF output filename
line_coords_filename = "line_dofs_coords.dat" # line dof coordinates output filename
timeseries_filename = "dof_temperature_timeseries.dat" # temperature timeseries output filename

# -------------------------
#%% Problem / time parameters
# -------------------------

dt = (T - t) / num_steps # time step size

# -------------------------
#%% Create mesh and function space
# -------------------------

domain = mesh.create_box(comm,
                         [p_min, p_max],
                         [nx, ny, nz],
                         cell_type=mesh.CellType.hexahedron,
                         ghost_mode=mesh.GhostMode.shared_facet)

gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", poly_order))
kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))

# -------------------------
#%% Identify boundary facets and coupled facets
# -------------------------

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# exterior facets indices (global entity indices)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# all boundary DOFs
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
if not quiet:
    print("Rank", comm.rank, "Number of boundary dofs:", len(boundary_dofs), "boundary facets:", len(boundary_facets))

coupled_facets = mesh.locate_entities_boundary(domain, fdim, coupled_boundary)
if coupled_facets.size == 0:
    raise RuntimeError("No coupled facets found. Check boundary function or mesh coordinates.")

# marker id for the coupled boundary
facet_tag = mesh.meshtags(domain, fdim, coupled_facets, np.full_like(coupled_facets, COUPLED_MARK))

# coupled DOFs for Dirichlet BC
coupleddofs = fem.locate_dofs_topological(V, fdim, coupled_facets)
if not quiet:
    print("Rank", comm.rank, "Number of coupled dofs:", len(coupleddofs))

# Extract coordinates of DOFs
dof_coords = V.tabulate_dof_coordinates().reshape((-1, gdim))
coupled_dof_coords = dof_coords[coupleddofs]

# -------------------------
#%% Function to hold Dirichlet values
# -------------------------

u_bc = fem.Function(V)
u_bc.name = "u_bc"

# Create coupled boundary condition
class boundary_condition():
    def __init__(self, t, coupled_dof_coords):
        self.t = t
        self.tree = cKDTree(coupled_dof_coords)  # build KDTree for coupled DOF coords

    def __call__(self, x):
        tol = 1e-8
        values = np.zeros(x.shape[1])

        # Query nearest boundary point distance for each x
        dist, _ = self.tree.query(x.T)
        on_boundary = dist < tol
        values[on_boundary] = (x[0, on_boundary] *
                               x[1, on_boundary] *
                               x[2, on_boundary] * 0.0) + T_L  # steady 500K for testing
        return values

u_boundary = boundary_condition(t, coupled_dof_coords)
u_bc.interpolate(u_boundary)

bc = fem.dirichletbc(u_bc, coupleddofs)

# -------------------------
#%% Initial condition for interior
# -------------------------

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.x.array[:] = T0

# -------------------------
#%% Variational forms with backward Euler
# -------------------------

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, PETSc.ScalarType(0.0))  # source term

F = (u - u_n) / dt * v * ufl.dx + kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

# Assemble matrix with Dirichlet BCs applied
A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)

# Function for current solution at each time step
uh = fem.Function(V)
uh.name = "uh"
uh.x.array[:] = T0

# Set up PETSc KSP solver
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# -------------------------
#%% Setup XDMF output
# -------------------------

xdmf = io.XDMFFile(comm, xdmf_filename, "w")
xdmf.write_mesh(domain)

# -------------------------
#%% Function to compute heat flux across the coupled boundary
# -------------------------

# Create UFL measure on boundary with subdomain_data
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
# FacetNormal(domain) yields the outward normal on the domain boundary facets
n = ufl.FacetNormal(domain)

def compute_total_heat_flux(func_u):
    integrand = -kappa * ufl.dot(ufl.grad(func_u), n)
    total = fem.assemble_scalar(fem.form(integrand * ds(COUPLED_MARK)))
    # assemble_scalar returns a scalar on each rank; convert to global sum
    total_global = comm.allreduce(total, op=MPI.SUM)
    return float(total_global)

# Compute local heat flux vector and coordinates at each DOF on the coupled boundary.
def compute_heat_flux_on_coupled_boundary(V, u, kappa):
    # Create vector function space for flux
    V_g = fem.functionspace(domain, ("Lagrange", poly_order, (domain.geometry.dim, )))

    # Heat flux expression
    flux_expr = -kappa * ufl.grad(u)

    # Create a function for flux evaluation
    flux = fem.Function(V_g)
    flux.name = "flux"

    # Project flux = -kappa * grad(u) onto V
    w_f = ufl.TrialFunction(V_g)
    v_f = ufl.TestFunction(V_g)
    a_proj = ufl.inner(w_f, v_f) * ufl.dx
    L_proj = ufl.inner(flux_expr, v_f) * ufl.dx
    problem = LinearProblem(a_proj, L_proj)
    flux = problem.solve()

    # Evaluate flux at these dofs
    flux_vals = flux.x.array.reshape((-1, gdim))[coupleddofs]

    # Gather to root if needed
    coupled_dof_coords_flux = np.array(coupled_dof_coords, dtype=np.float64)
    flux_vals = np.array(flux_vals, dtype=np.float64)

    # Print flux values and coordinate components
    print("Heat flux vectors at coupled boundary DOFs:")
    for coord, flux_vec in zip(coupled_dof_coords_flux, flux_vals):
        x, y, z = coord  # unpack coordinate components
        print(f"x = {x:.6f}, y = {y:.6f}, z = {z:.6f} | Flux = [{flux_vec[0]:.6e}, {flux_vec[1]:.6e}, {flux_vec[2]:.6e}]")
    
    # Gather to root if needed
    coupled_dof_coords_flux = np.vstack(comm.allgather(coupled_dof_coords_flux))
    flux_vals = np.vstack(comm.allgather(flux_vals))

    # Compute total flux as sum of normal components at coupled DOFs
    total_flux = 0.0
    for coord, flux_vec in zip(coupled_dof_coords_flux, flux_vals):
        normal = np.array([0.0, -1.0, 0.0])  # normal vector on the left face at x = -2.0
        total_flux += np.dot(flux_vec, normal)

    return total_flux

# -------------------------
#%% Select DoFs for ASCII output
# -------------------------

dof_coords_output = V.tabulate_dof_coordinates()
line_dofs = np.where(line_mask(dof_coords_output))[0]
line_coords = dof_coords_output[line_dofs,0]
sort_idx = np.argsort(line_coords)
line_dofs = line_dofs[sort_idx]
line_coords = line_coords[sort_idx]

if domain.comm.rank == 0:
     # Save DOF coordinates along the selected line
    np.savetxt(line_coords_filename, line_coords)
    # Open ASCII file for temperature time series
    f_ascii = open(timeseries_filename, "w")
    header = "# time " + " ".join([f"x={x:.4f}" for x in line_coords])
    f_ascii.write(header + "\n")

# -------------------------
#%% Time-stepping loop
# -------------------------

for step in range(1, num_steps + 1):
    t += dt
    print(f"[rank {comm.rank}] Time step {step}/{num_steps}, t = {t:.6f}")

    # Update Diriclet boundary condition
    u_boundary.t += dt
    u_bc.interpolate(u_boundary)

    # Assemble RHS with current u_n
    with b.localForm() as loc_b:
        loc_b.set(0.0)
    assemble_vector(b, L)

    # Apply BCs to RHS
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear system A * uh = b
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Compute heat flux across the coupled boundary
    total_flux_push = compute_heat_flux_on_coupled_boundary(V, uh, kappa)
    total_flux = compute_total_heat_flux(uh)
    if comm.rank == 0:
        # positive flux means heat leaving the solid across the boundary (sign follows -kappa*grad·n)
        print(f"  Total heat flux across coupled boundary (integral) = {total_flux:.6e}; and total heat flux push (DOF sum) = {total_flux_push:.6e}")

    # Update u_n for next time step
    u_n.x.array[:] = uh.x.array

    # Write solution to XDMF file
    xdmf.write_function(uh, t)

    # Write ASCII temperature timeseries at selected line DOFs
    if domain.comm.rank == 0:
        line_temps = uh.x.array[line_dofs]
        f_ascii.write(f"{t:.6f} " + " ".join(f"{temp:.6f}" for temp in line_temps) + "\n")

# -------------------------
#%% Finalise
# -------------------------

# Close ASCII file after time-stepping
if domain.comm.rank == 0:
    f_ascii.close()
# Close XDMF file after time-stepping
xdmf.close()
# Indicate run completion
if comm.rank == 0:
    print("Run complete.")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
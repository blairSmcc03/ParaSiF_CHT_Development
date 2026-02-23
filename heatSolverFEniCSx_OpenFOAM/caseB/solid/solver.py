from mpi4py import MPI
import mui4py
import math

LOCAL_COMM_WORLD = MPI.COMM_WORLD
PYTHON_COMM_WORLD = mui4py.mpi_split_by_app()

import petsc4py
petsc4py.init(comm=PYTHON_COMM_WORLD)

from output import Output
from heatEquationFenics import HeatEquationFenics

# simulation parameters
final_time = 146
dt = 0.05
num_steps = math.ceil(final_time/dt)
poly_order = 2  # polynomial order for FEM

# Physical parameters
kappa_val = 1600  # thermal conductivity
alpha_val = 0.0426667

T0 = 273.15

# mesh parameters
nx, ny, nz = 10, 10, 1  # mesh divisions
lx, ly, lz = 0.2, 1.0, 0.01  # domain lengths



heatSolver = HeatEquationFenics((nx, ny, nz), (lx, ly, lz), alpha_val, kappa_val, poly_order, PYTHON_COMM_WORLD, LOCAL_COMM_WORLD, dt)

heatSolver.initialise_temperature_field(273.15)
heatSolver.set_left_boundary_condition(273.15)
heatSolver.set_right_boundary_condition(274.15)

dof_coords_output = heatSolver.V.tabulate_dof_coordinates()
output = Output(heatSolver.mesh.domain, dof_coords_output, PYTHON_COMM_WORLD)

t = 0
c = 1
inner_loop_iterations = 1  # number of iterations for the inner loop (e.g., for coupling with OpenFOAM)
writeInterval = 0.2
for step in range(1, num_steps + 1):
    t += dt
    for i in range(inner_loop_iterations):
        heatSolver.update_boundary_conditions(c)
        heatSolver.solve()
        c += 1  
    heatSolver.update_time()
    if step % int(writeInterval / dt) == 0:
        output.writeFunction(heatSolver.uh_out, t)

output.close()


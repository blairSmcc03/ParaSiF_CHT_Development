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

    @file compareSolution.py

    @author W. Liu

    @brief Compare FEniCSx simulation results with analytical solution for
           transient 1-D heat conduction problem.

"""

# -------------------------
#%% Import packages
# -------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
#%% Control parameters
# -------------------------

quiet = True  # define quiet mode
kappa = 1.0 # thermal conductivity
L = 4.0 # length of the 1-D domain
T_L, T_R, T0 = 500.0, 0.0, 0.0 # boundary and initial temperatures
N_terms = 200  # number of series terms
n_select = 5  # Define number of selected times including start and end
line_coords_filename = "line_dofs_coords.dat" # FEniCSx line dof coordinates filename
timeseries_filename = "dof_temperature_timeseries.dat" # FEniCSx temperature timeseries output filename

# -------------------------
#%% Analytical series solution
# -------------------------

def u_analytic(x, x0, t):
    # steady linear part
    u_s = T_L + (T_R - T_L) * (x-x0) / L
    # Fourier coefficients
    u_sum = np.zeros_like((x-x0))
    for n in range(1, N_terms+1):
        k = n * np.pi / L
        if n % 2 == 0:
            b_n = 2 * (T_R - T_L) / (n * np.pi)
        else:
            b_n = (4*T0 - 2*T_L - 2*T_R) / (n * np.pi)
        u_sum += b_n * np.sin(k*(x-x0)) * np.exp(-kappa * k**2 * t)
    return u_s + u_sum

# -------------------------
#%% Load FEniCSx ASCII data
# -------------------------

coords = np.loadtxt(line_coords_filename)
data = np.loadtxt(timeseries_filename, comments="#")

# Extract time and temperature data
times = data[:,0] # Time values
T_num = data[:,1:] # Temperature values at each time step
t_start = data[0, 0] # Start time
t_end   = data[-1, 0] # End time
x_left = coords[0]  # Left end assuming sorted along x-axis

# -------------------------
#%% Plot comparison at selected times
# -------------------------

select_times = np.linspace(t_start, t_end, n_select) # Generate equally spaced times between t_start and t_end
if not quiet:
    print("Select times:", select_times)

for t_sel in select_times:
    # find nearest time index in FEniCSx output
    idx = np.argmin(np.abs(times - t_sel))
    T_ana = u_analytic(coords, x_left, times[idx])
    plt.figure()
    plt.plot(coords, T_ana, 'k-', lw=2, label="Analytical")
    plt.plot(coords, T_num[idx,:], 'ro', mfc='none', label="FEniCSx thermal solver")
    plt.xlabel("x")
    plt.ylabel("Temperature")
    plt.title(f"t = {times[idx]:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"compare_t{times[idx]:.3f}.png", dpi=150)
    plt.close()

# -------------------------
#%% Finalise
# -------------------------

# Indicate run completion
print("Comparison plots saved as compare_t*.png")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np

n_particles = 60 # no. of particles in the ensemble; this depends on the no. of particles for which data is available
var_level = 70
grid = 64

print(f'number of particles: {n_particles}')
print(f'variance level: {var_level} percent')
print(f'grid: {grid}x{grid}')

base_mesh = UnitSquareMesh(grid,grid)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

mesh = bary_mesh_hier[-1]

time_delta = 1.0 # saving the fields at an interval of time_delta
t_start = 50
t_end = 70
time_array = np.arange(t_start, t_end + 1, time_delta)
t_stamps = np.round(time_array, 1)

print("time_stamps:", time_array)

V1 = VectorFunctionSpace(mesh, "CG", 2)

u_sto = Function(V1) # stochastic vel fields
u_t = Function(V1) # coarse grained fields

u_sto.assign(0) 
u_t.assign(0) 


current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)


def relative_l2_error(f_truth, f):
    """ compute the relative L2 error between two functions. See Wei's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return errornorm(f_truth, f)/norm(f_truth)


l2_vel_sto_v_truth_global = [] # relative l2 error between stochastic ensemble and coarse grained truth  

for i in t_stamps:

    print(f'loading the velocity and temperature fields at t = {i}')

    l2_vel_sto_v_truth_loc = [] 

    # loading data from coarse grained solution at one time instance 
    data_t = np.load('../../../deterministic/64x64/global_comparison/fields_as_arrays/u_coarse_grained_arrays_mesh_'+str(grid)+'_at_t_'+str(i)+'.npz')
    
    vel_array_t = data_t['vel_array'] 
    u_t.dat.data[:] = vel_array_t

    # going through each ensemble member and calculating l2 error 
    for j in range(n_particles):
    
        # loading data for each particle corresponding to time t = i
        data_sto = np.load('../ensemble_vel_data_as_arrays/stoch_sim_mesh_'+str(grid)+'_var_'+str(var_level)+'_particle_'+str(j)+'_fields_data_at_t_'+str(i)+'.npz')
  
        vel_array_sto = data_sto['vel_array'] 
        

        u_sto.dat.data[:] = vel_array_sto

        ### calculating relative l2 errors between stochastic ensemble and truth
        l2_vel_sto_v_truth_loc.append(relative_l2_error(u_t, u_sto))

    # collecting data from all ensemble members into one array
    l2_vel_sto_v_truth_global.append(l2_vel_sto_v_truth_loc)

    data_file = './l2_error_data/sto_ensemble_v_truth_var_'+str(var_level)+'_grid_'+str(grid)+'_particles_'+str(n_particles)+'_l2_error_t'+str(t_start)+'_to_t'+str(t_end)+'.npz'
    np.savez(data_file, l2_vel_stoc_v_truth = np.array(l2_vel_sto_v_truth_global))


print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))


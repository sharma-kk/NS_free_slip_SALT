import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np
from firedrake.petsc import PETSc

my_ensemble = Ensemble(COMM_WORLD, 1)
spatial_comm = my_ensemble.comm
ensemble_comm = my_ensemble.ensemble_comm

PETSc.Sys.Print(f'size of ensemble is {ensemble_comm.size}')
PETSc.Sys.Print(f'size of spatial communicators is {spatial_comm.size}')

n_ele = 64

base_mesh = UnitSquareMesh(n_ele,n_ele, comm = spatial_comm)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

mesh = bary_mesh_hier[-1]

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "DG", 1)

Z = V*W

up = Function(Z)

u,p = split(up)
v,q = TestFunctions(Z)

Dt = 0.1

x, y = SpatialCoordinate(mesh)

# define Reynolds no.
Re = 4*10**4

alpha = 0.01
beta = 8
f_ext = project(as_vector([alpha*y*sin(beta*pi*x), 
                           alpha*sin(beta*pi*x)*sin(beta*pi*y)]), V)

PETSc.Sys.Print(f'Re: {Re}, Dt: {Dt}, f_mag: {alpha}, mesh: {n_ele}x{n_ele}')

p_ = interpolate(Constant(0), W)
u_ = Function(V)
u_pert = Function(V)

data = np.load('../../deterministic/64x64/coarse_grained_vel_vector/u_coarse_grained_vel_array_mesh_64_c_1by64_t50.npz')
vel_array = data['vel_array'] # loading the vel. data array

u_.assign(0)
u_.dat.data[:] = vel_array


seed_no = ensemble_comm.rank  # seed no.

particle_no = ensemble_comm.rank + 50

pvd_print = 5


F = ( inner(u - u_, v)
     + Dt * 0.5 * (1/Re) * inner((nabla_grad(u) + nabla_grad(u_)), nabla_grad(v))
     + Dt * 0.5 *(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
     - Dt * p * div(v) - Dt * inner(f_ext, v)  
     + Dt * div(u) * q 
     + np.sqrt(Dt)*0.5*(inner(dot(u_pert, nabla_grad(u)), v) + inner(dot(u_pert, nabla_grad(u_)), v))
     + np.sqrt(Dt)*0.5*(inner((u_[0]+u[0])*grad(u_pert[0]) + (u_[1]+u[1])*grad(u_pert[1]) , v))) * dx  

bc_left_right = DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1,2))
bc_bot_top = DirichletBC(Z.sub(0).sub(1), Constant(0.0), (3,4))

BCs = []
BCs.append(bc_left_right)
BCs.append(bc_bot_top)

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

xi_data = np.load('../../deterministic/64x64/xi_calculation_visualization/calculated_xi_vectors/xi_matrix_75_eigvec_c_1_by_64_decor_t_1_Dt_t=50_to_t=110.npz')
xi_mat = xi_data['xi_mat']

### 64x64 grid ####
# 75 EOFs ----> 99% variance
# 32 EOFs ----> 90% variance
# 14 EOFs ----> 70% variance

PETSc.Sys.Print(f'loaded the xi matrix for particle {particle_no}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

# time stepping and visualization at other time steps
t_start = 50.0 + Dt
t_end = 70

n_t_steps = int((t_end - t_start)/Dt)

n_EOF = 32 # 32 EOFs covers 90 percent variance

PETSc.Sys.Print(f'no. of EOFs = {n_EOF}')
np.random.seed(seed_no)
rand_mat = np.random.normal(size=(n_t_steps+2, n_EOF))

# saving the vel. fields as an array at a particular time instance
data_file = './ensemble_vel_data_as_arrays/stoch_sim_mesh_64_var_90_particle_'+str(particle_no)+'_fields_data_at_t_50.0.npz'
np.savez(data_file, vel_array = u_.dat.data)

t = 50.0 + Dt
iter_n = 1
freq_stoc = 10 # saving vel. data every 1 time unit
freq_pvd = 100 # saving the .pvd files every 10 time units
big_t_step = freq_stoc*Dt 
current_time = time.strftime("%H:%M:%S", time.localtime())
PETSc.Sys.Print(f'Local time at the start of simulation for particle {particle_no}: {time.strftime("%H:%M:%S", time.localtime())}')
start_time = time.time()

PETSc.Sys.Print(f'particle no:{particle_no}, local time:{round(t,4)}')

while (round(t,4) <= t_end):
    vec_u_pert = np.zeros((xi_mat.shape[1], 2))
    for i in range(n_EOF):
        vec_u_pert +=  rand_mat[iter_n-1,i]*xi_mat[i, :,:]

    u_pert.assign(0)
    u_pert.dat.data[:] = vec_u_pert

    solve(F == 0, up, bcs = BCs, nullspace=nullspace) 
    u, p = up.subfunctions
    if iter_n%freq_stoc == 0:
        if iter_n == freq_stoc:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print(f'approx. running time for one big t_step for particle {particle_no} is {round(execution_time,2)} minutes')
            total_execution_time = ((t_end - t_start)/big_t_step)*execution_time
            PETSc.Sys.Print(f'approx. total running time for particle {particle_no} is {round(total_execution_time,2)} minutes')

        PETSc.Sys.Print(f'particle no:{particle_no}, simulation time:{round(t,4)}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

        data_file = './ensemble_vel_data_as_arrays/stoch_sim_mesh_64_var_90_particle_'+str(particle_no)+'_fields_data_at_t_'+str(round(t,4))+'.npz'
        np.savez(data_file, vel_array = u.dat.data)

    u_.assign(u)
    t += Dt
    iter_n +=1

print(f'Local time at the end of simulation for particle {particle_no} is {time.strftime("%H:%M:%S", time.localtime())}')

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

n_ele = 32

base_mesh = UnitSquareMesh(n_ele,n_ele, comm = spatial_comm)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

mesh = bary_mesh_hier[-1]

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "DG", 1)

Z = V*W

up = Function(Z)

u,p = split(up)
v,q = TestFunctions(Z)

Dt = 0.2

x, y = SpatialCoordinate(mesh)

# define Reynolds no.
Re = 2*10**4

alpha = 0.01
beta = 8
f_ext = project(as_vector([alpha*y*sin(beta*pi*x), 
                           alpha*sin(beta*pi*x)*sin(beta*pi*y)]), V)

PETSc.Sys.Print(f'Re: {Re}, Dt: {Dt}, f_mag: {alpha}, mesh: {n_ele}x{n_ele}')

p_ = interpolate(Constant(0), W)
u_ = Function(V)
u_pert = Function(V)

data = np.load('../../deterministic/32x32/coarse_grained_vel_vector/u_coarse_grained_vel_array_mesh_32_c_1by32_t50.npz')
vel_array = data['vel_array'] # loading the vel. data array

u_.assign(0)
u_.dat.data[:] = vel_array

seed_no = ensemble_comm.rank + 140 # seed no.

particle_no = ensemble_comm.rank + 40

# pvd_print = 9
# if ensemble_comm.rank%pvd_print == 0: # printing results correponding to every pvd_print particle
#     u_.rename("Velocity")
#     p_.rename("Pressure")
#     outfile = File('./results/Re_2e4_var_90_mesh_32_particle_'+str(ensemble_comm.rank)+'_t50_onwards.pvd', comm = spatial_comm)
#     outfile.write(u_, p_)


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

xi_data = np.load('../../deterministic/32x32/xi_calculation_visualization/calculated_xi_vectors/xi_matrix_56_eigvec_c_1_by_32_decor_t_1_Dt_t=50_to_t=110.npz')
xi_mat = xi_data['xi_mat']

PETSc.Sys.Print(f'loaded the xi matrix for particle {particle_no}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

delta_x = 1/4
delta_y = 1/4
n = 3
gridpoints = np.array([[delta_x + i * delta_x, delta_y + j * delta_y] for j in range(n) for i in range(n)])

# time stepping and visualization at other time steps
t_start = 50.0 + Dt
t_end = 70

n_t_steps = int((t_end - t_start)/Dt)

### 32x32 grid ####
# 56 EOFs ----> 99 % variance
# 23 EOFs ----> 90% variance
# 10 EOFs ----> 70% variance

n_EOF = 10 # 10 EOFs covers 70 percent variance

PETSc.Sys.Print(f'no. of EOFs = {n_EOF}')
np.random.seed(seed_no)
rand_mat = np.random.normal(size=(n_t_steps+2, n_EOF))
vel_data_sto = []
vel_data_sto.append(np.array(u_.at(gridpoints, tolerance=1e-10)))

t = 50.0 + Dt
iter_n = 1
freq_stoc = 5 # saving vel. data every 1 time unit
freq_pvd = 50 # saving the .pvd files every 10 time units
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

        vel_data_sto.append(np.array(u.at(gridpoints, tolerance=1e-10)))
        data_file = './data_from_stochastic_run/Re_2e4_var_70_mesh_32_vel_data_particle_'+str(particle_no)+'_t50_onwards.npz'
        np.savez(data_file, gridpoints = gridpoints, vel_data_sto = np.array(vel_data_sto))

    # if iter_n%freq_pvd == 0:
    #     if ensemble_comm.rank%pvd_print == 0: # printing results correponding to every pvd_print particle
    #         u.rename("Velocity")
    #         p.rename("Pressure")
    #         outfile.write(u, p)

    u_.assign(u)
    t += Dt
    iter_n +=1

print(f'Local time at the end of simulation for particle {particle_no} is {time.strftime("%H:%M:%S", time.localtime())}')

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np

n_ele = 64

with CheckpointFile("../h5_files/coarse_grained_fields_at_t50_mesh_64_c_1by64.h5", 'r') as afile:
     mesh = afile.load_mesh("coarse_mesh_64")
     u_ = afile.load_function(mesh, "coarse_vel")

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

print(f'Re: {Re}, Dt: {Dt}, f_mag: {alpha}, mesh: {n_ele}x{n_ele}')
p_ = interpolate(Constant(0), W)

F = ( inner(u - u_, v)
     + Dt * 0.5 * (1/Re) * inner((nabla_grad(u) + nabla_grad(u_)), nabla_grad(v))
     + Dt * 0.5 *(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
     - Dt * p * div(v) - Dt * inner(f_ext, v)  
     + Dt * div(u) * q ) * dx

bc_left_right = DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1,2))
bc_bot_top = DirichletBC(Z.sub(0).sub(1), Constant(0.0), (3,4))

BCs = []
BCs.append(bc_left_right)
BCs.append(bc_bot_top)

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

# saving the fields as arrays at t = 50
data_file ='./fields_as_arrays/u_deter_sim_arrays_mesh_64_at_t_50.0.npz'
np.savez(data_file, vel_array = u_.dat.data)

# time stepping and visualization at other time steps
t_start = 50.0 + Dt
t_end = 90

t = 50.0 + Dt
iter_n = 1
freq = 10 # prints every 1 time units
big_t_step = freq*Dt # 1 time units

print("Local time at the start of simulation:",time.strftime("%H:%M:%S", time.localtime()))
start_time = time.time()

while (round(t,4) <= t_end):
    solve(F == 0, up, bcs = BCs, nullspace=nullspace) 
    u, p = up.subfunctions
    
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/big_t_step)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)


        print("t=", round(t,4))
        print("Local time at this time instant:",time.strftime("%H:%M:%S", time.localtime()))
        data_file ='./fields_as_arrays/u_deter_sim_arrays_mesh_64_at_t_'+str(round(t,4))+'.npz'
        np.savez(data_file, vel_array = u.dat.data)

    u_.assign(u)
    t += Dt
    iter_n +=1

print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))
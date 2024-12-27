import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np

n_ele = 64

base_mesh = UnitSquareMesh(n_ele,n_ele)

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

print(f'Re: {Re}, Dt: {Dt}, f_mag: {alpha}, mesh: {n_ele}x{n_ele}')

p_ = interpolate(Constant(0), W)
u_ = Function(V)
u_pert = Function(V)

data = np.load('./coarse_grained_vel_vector/u_coarse_grained_vel_array_mesh_64_c_1by64_t50.npz')
vel_array = data['vel_array'] # loading the vel. data array

u_.assign(0)
u_.dat.data[:] = vel_array


F = ( inner(u - u_, v)
     + Dt * 0.5 * (1/Re) * inner((nabla_grad(u) + nabla_grad(u_)), nabla_grad(v))
     + Dt * 0.5 *(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
     - Dt * p * div(v) - Dt * inner(f_ext, v)  
     + Dt * div(u) * q) * dx  

bc_left_right = DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1,2))
bc_bot_top = DirichletBC(Z.sub(0).sub(1), Constant(0.0), (3,4))

BCs = []
BCs.append(bc_left_right)
BCs.append(bc_bot_top)

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

delta_x = 1/4
delta_y = 1/4
n = 3
gridpoints = np.array([[delta_x + i * delta_x, delta_y + j * delta_y] for j in range(n) for i in range(n)])

# time stepping and visualization at other time steps
t_start = 50.0 + Dt
t_end = 90

vel_data_det = []
vel_data_det.append(np.array(u_.at(gridpoints, tolerance=1e-10)))

t = 50.0 + Dt
iter_n = 1
freq_det = 10 # saving vel. data every 1 time unit
big_t_step = freq_det*Dt 
current_time = time.strftime("%H:%M:%S", time.localtime())
print(f'Local time at the start of simulation: {time.strftime("%H:%M:%S", time.localtime())}')
start_time = time.time()

while (round(t,4) <= t_end):
    solve(F == 0, up, bcs = BCs, nullspace=nullspace) 
    u, p = up.subfunctions
    if iter_n%freq_det == 0:
        if iter_n == freq_det:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            print(f'approx. running time for one big t_step  is {round(execution_time,2)} minutes')
            total_execution_time = ((t_end - t_start)/big_t_step)*execution_time
            print(f'approx. total running time is {round(total_execution_time,2)} minutes')

        print(f'simulation time:{round(t,4)}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

        vel_data_det.append(np.array(u.at(gridpoints, tolerance=1e-10)))
        data_file = './data_from_deterministic_run/Re_4e4_mesh_64_vel_data_deterministic_t50_onwards.npz'
        np.savez(data_file, gridpoints = gridpoints, vel_data_det = np.array(vel_data_det))

    u_.assign(u)
    t += Dt
    iter_n +=1

print(f'Local time at the end of simulation is {time.strftime("%H:%M:%S", time.localtime())}')
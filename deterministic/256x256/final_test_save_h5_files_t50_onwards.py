import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
from firedrake.petsc import PETSc

n_ele = 256

PETSc.Sys.Print("loading .h5 file, time:", time.strftime("%H:%M:%S",time.localtime()))

with CheckpointFile("./h5_files/final_test_grid_256_fields_at_time_t=50.0.h5", 'r') as afile:
     mesh = afile.load_mesh("mesh_256")
     u_ = afile.load_function(mesh, "velocity")

PETSc.Sys.Print(".h5 file loaded, time:", time.strftime("%H:%M:%S",time.localtime()))

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "DG", 1)

Z = V*W

up = Function(Z)

u,p = split(up)
v,q = TestFunctions(Z)

Dt = 0.025 # this is the time-step we decided !  

x, y = SpatialCoordinate(mesh)

# define Reynolds no.
Re = 8*10**4

alpha = 0.01
beta = 8
f_ext = project(as_vector([alpha*y*sin(beta*pi*x), 
                           alpha*sin(beta*pi*x)*sin(beta*pi*y)]), V)

PETSc.Sys.Print(f'Re: {Re}, Dt: {Dt}, f_mag: {alpha}, mesh: {n_ele}x{n_ele}')

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

# time stepping and visualization at other time steps
t_start = 50.0 + Dt
t_end = 110

t = 50.0 + Dt
iter_n = 1
freq = 4 # prints every 0.1 time units
t_step = freq*Dt # 0.1 time units

PETSc.Sys.Print("Local time at the start of simulation:",time.strftime("%H:%M:%S", time.localtime()))
start_time = time.time()

while (round(t,4) <= t_end):
    solve(F == 0, up, bcs = BCs, nullspace=nullspace) 
    u, p = up.subfunctions
    
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            PETSc.Sys.Print("Approx. total running time: %.2f minutes:" %total_execution_time)


        PETSc.Sys.Print("t=", round(t,4))
        PETSc.Sys.Print("Local time at this time instant:",time.strftime("%H:%M:%S", time.localtime()))
        u.rename("velocity")
        h5_file = "./h5_files/final_test_grid_256_fields_at_time_t="+ str(round(t,4)) + ".h5"
        PETSc.Sys.Print(f'Saving the fields at t={round(t,4)} into the .h5 file')
        with CheckpointFile(h5_file, 'w') as afile:
               afile.save_mesh(mesh)
               afile.save_function(u)

    u_.assign(u)
    t += Dt
    iter_n +=1

PETSc.Sys.Print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))
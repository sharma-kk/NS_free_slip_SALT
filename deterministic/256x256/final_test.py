import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
from firedrake.petsc import PETSc

n_ele = 256

base_mesh = UnitSquareMesh(n_ele,n_ele)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

mesh = bary_mesh_hier[-1]

mesh.name = "mesh_256"

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "DG", 1)

Z = V*W

up = Function(Z)

u,p = split(up)
v,q = TestFunctions(Z)

Dt = 0.025   # i have decided to use this as my time step size

x, y = SpatialCoordinate(mesh)

# define Reynolds no.
Re = 8*10**4

u_ = Function(V)
u_.assign(0)

alpha = 0.01
beta = 8
f_ext = project(as_vector([alpha*y*sin(beta*pi*x), 
                           alpha*sin(beta*pi*x)*sin(beta*pi*y)]), V)

PETSc.Sys.Print(f'Re: {Re}, Dt: {Dt}, f_mag: {alpha}, mesh: {n_ele}x{n_ele}')
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

u_.rename("velocity")
# f_ext.rename("ext_force")
p_.rename("pressure")

file_name = "final_test"
outfile = File("./results/"+file_name+".pvd")
outfile.write(u_, p_)

energy_ = 0.5*(norm(u_)**2)
KE = []
KE.append(round(energy_,7))
PETSc.Sys.Print(f'KE at time t=0: {round(energy_,7)}')

# time stepping and visualization at other time steps
t_start = Dt
t_end = 120

t = Dt
iter_n = 1
freq = 400 # prints every 10 time units
freq_k = 40 # freq after which we output KE data
freq_h = 400 # saves .h5 file every 10 time units
t_step = freq_k*Dt # 1 time units

PETSc.Sys.Print("Local time at the start of simulation:",time.strftime("%H:%M:%S", time.localtime()))
start_time = time.time()
data_file = "./KE_data/"+file_name+".txt"

while (round(t,4) <= t_end):
    solve(F == 0, up, bcs = BCs, nullspace=nullspace) 
    u, p = up.subfunctions
    energy = 0.5*(norm(u)**2)
    KE.append(round(energy,7))
    
    if iter_n%freq_k == 0:
        if iter_n == freq_k:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            PETSc.Sys.Print("Approx. total running time: %.2f minutes:" %total_execution_time)


        PETSc.Sys.Print("t=", round(t,4))
        PETSc.Sys.Print("kinetic energy:", KE[-1])
        PETSc.Sys.Print("Local time at this time instant:",time.strftime("%H:%M:%S", time.localtime()))
        
        with open(data_file, 'w') as ff:
            print(f'KE_over_time = {KE}', file = ff)

    if iter_n%freq == 0:
        p.rename("pressure")
        u.rename("velocity")
        outfile.write(u, p)
    
    if iter_n%freq_h == 0:
        h5_file = "./h5_files/"+file_name+"_grid_256_fields_at_time_t="+ str(round(t,4)) + ".h5"
        PETSc.Sys.Print(f'Saving the fields at t={round(t,4)} into the .h5 file')
        with CheckpointFile(h5_file, 'w') as afile:
               afile.save_mesh(mesh)
               afile.save_function(u)

    u_.assign(u)
    t += Dt
    iter_n +=1

PETSc.Sys.Print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))
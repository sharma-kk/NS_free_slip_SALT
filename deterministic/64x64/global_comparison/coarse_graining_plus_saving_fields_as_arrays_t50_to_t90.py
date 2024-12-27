import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
from firedrake.petsc import PETSc
import numpy as np


time_delta = 1.0 # saving the fields at an interval of time_delta
t_start = 50
t_end = 90
time_array = np.arange(t_start, t_end + 1, time_delta)
t_stamps = np.round(time_array, 1)

print("time_stamps:", time_array)

n_ele = 64

base_mesh = UnitSquareMesh(n_ele,n_ele)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

c_mesh = bary_mesh_hier[-1]
c_mesh.name = "coarse_mesh_64"

Vc = VectorFunctionSpace(c_mesh, "CG", 2)

coords_func_coarse = Function(Vc).interpolate(SpatialCoordinate(c_mesh))
coords_coarse = coords_func_coarse.dat.data

uc = Function(Vc)

for i in t_stamps:
     
    print("loading high resolution mesh and velocity for t="+str(i),
        "current time:",time.strftime("%H:%M:%S", time.localtime()))

    with CheckpointFile("../../256x256/h5_files/final_test_grid_256_fields_at_time_t="+str(i)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_256")
        u_ = afile.load_function(mesh, "velocity")


    PETSc.Sys.Print(".h5 file loaded, time:", time.strftime("%H:%M:%S",time.localtime()))

    Vf = VectorFunctionSpace(mesh, "CG", 2)

    print("Coarse graining.........",
        time.strftime("%H:%M:%S", time.localtime()))

    #####Averaging and Coarse graining#########
    u_trial = TrialFunction(Vf)
    u_test = TestFunction(Vf)

    u_avg = Function(Vf)

    c_sqr = Constant(1/(64**2)) # averaging solution within box of size 1/64x1/64

    a_vel = (c_sqr * inner(nabla_grad(u_trial), nabla_grad(u_test)) + inner(u_trial, u_test)) * dx
    l_vel = inner(u_, u_test) * dx

    bc_left_right = DirichletBC(Vf.sub(0), Constant(0.0), (1,2))
    bc_bot_top = DirichletBC(Vf.sub(1), Constant(0.0), (3,4))

    BCs = [] # making sure that n.v is zero after coarse graining
    BCs.append(bc_left_right)
    BCs.append(bc_bot_top)

    # step 1: spatial averaging using Helmholtz operator
    solve(a_vel==l_vel, u_avg, bcs = BCs)

    print("solved the PDEs (alpha-regularization)",
        time.strftime("%H:%M:%S", time.localtime()))

    u_avg.rename("avg_vel")

    print("retrieving velocity data........",
        time.strftime("%H:%M:%S", time.localtime()))
    uc.assign(0)
    u_avg_vals = np.array(u_avg.at(coords_coarse, tolerance=1e-10)) # the code runs without specifying tolerance also
    # the above is the costliest step. It takes ~15 min. With tolerance it takes seconds to process
    uc.dat.data[:] += u_avg_vals

    uc.rename("coarse_vel")
    
    data_file = './fields_as_arrays/u_coarse_grained_arrays_mesh_64_at_t_'+str(i)+'.npz'
    np.savez(data_file, vel_array = uc.dat.data)

print("Simulation completed !",
    time.strftime("%H:%M:%S", time.localtime()))
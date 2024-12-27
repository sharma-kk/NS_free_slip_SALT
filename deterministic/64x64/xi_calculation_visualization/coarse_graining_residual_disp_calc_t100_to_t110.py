import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np

dX = []
delta_x = 1/4
delta_y = 1/4
n = 3
vel_data_truth = []

gridpoints = np.array([[delta_x + i * delta_x, delta_y + j * delta_y] for j in range(n) for i in range(n)])

print(f'Monitoring fields at points: {gridpoints}')

Dt_uc = 0.1 # assumed decorrelated time (same as Dt)
t_start = 100
t_end = 110

print(f'Recording (u-u_avg) from high res. sim. data from t = {t_start} to t = {t_end} at an interval of Dt = {Dt_uc}')
time_array = np.arange(t_start, t_end, Dt_uc)
t_stamps = np.round(time_array, 2)

print("time_stamps:", time_array)

n_ele = 64

c = 1/64
print(f'coarse graining with c = 1/{n_ele}')

print(f'obtaining data for calibrating {n_ele}x{n_ele} mesh')

base_mesh = UnitSquareMesh(n_ele,n_ele)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

c_mesh = bary_mesh_hier[-1]
c_mesh.name = "coarse_mesh_64"

Vc = VectorFunctionSpace(c_mesh, "CG", 2)

coords_func_coarse = Function(Vc).interpolate(SpatialCoordinate(c_mesh))
coords_coarse = coords_func_coarse.dat.data

uc = Function(Vc)

outfile = File("./results/coarse_grained_vel_64_grid_c_1_by_64_t=100_to_t=110.pvd")
data_file = "./KE_data/coarse_graining_residual_disp_calc_t100_to_t110_data_mesh_64.txt"
KE = []

freq = int(1/Dt_uc)
iter_n = 1
t_step = freq*Dt_uc
start_time = time.time()

for i in t_stamps:
    
    print('time:', i)
    print("loading high resolution mesh and velocity for t="+str(i),
        "current time:",time.strftime("%H:%M:%S", time.localtime()))

    with CheckpointFile("../../../deterministic/256x256/h5_files/final_test_grid_256_fields_at_time_t="+str(i)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_256")
        u_ = afile.load_function(mesh, "velocity")


    print(".h5 file loaded, time:", time.strftime("%H:%M:%S",time.localtime()))

    Vf = VectorFunctionSpace(mesh, "CG", 2)

    print("Coarse graining.........",
        time.strftime("%H:%M:%S", time.localtime()))

    #####Averaging and Coarse graining#########
    u_trial = TrialFunction(Vf)
    u_test = TestFunction(Vf)

    u_avg = Function(Vf)
    
    c_sqr = Constant(c**2) # averaging solution within box of size 1/64x1/64

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

    print("calculating (u - u_avg)........",
        time.strftime("%H:%M:%S", time.localtime()))

    dX.append((np.array(u_.at(coords_coarse, tolerance=1e-10)) 
                - np.array(uc.at(coords_coarse, tolerance=1e-10))))
    
    print("Calculation done, saving the data into a separate file",
        time.strftime("%H:%M:%S", time.localtime()))
    
    print("shape of dX array:", np.array(dX).shape)

    dX_x = np.array(dX)[:,:,0] # x-component of velocity difference
    print("shape of dX1_x:", dX_x.shape)

    dX_y = np.array(dX)[:,:,1] # y-component of velocity difference
    print("shape of dX1_y:", dX_y.shape)

    data_file_1 = './data_for_xi_calculation/dX_data_t='+str(t_start)+'_to_t='+str(t_end)+'_grid_64_c_1by64.npz'
    np.savez(data_file_1, dX_x = dX_x, dX_y = dX_y)

    if round(i - int(i),1) == 0:
        print("saving coarse-grained velocity at observation points at time t =",i)
        vel_data_truth.append(np.array(uc.at(gridpoints, tolerance=1e-10)))
        data_file_2 = './coarse_grained_vel_data/coarse_grained_vel_data_t='+str(t_start)+'_to_t='+str(t_end)+'_grid_64_c_1by64.npz'
        print(f'shape of coarse-grained velocity data array: {np.array(vel_data_truth).shape}')
        np.savez(data_file_2, gridpoints = gridpoints, vel_data_truth = np.array(vel_data_truth))
        
        energy = 0.5*(norm(uc)**2)
        KE.append(energy)
        with open(data_file, 'w') as ff:
            print(f'KE = {KE}', file = ff)

        uc.rename("coarse_vel")
        outfile.write(uc)

    if iter_n == freq:
        end_time = time.time()
        execution_time = (end_time-start_time)/60 # running time for 5 time step 
        print("Approx. running time for 10 t_steps: %.2f minutes" %execution_time)
        total_execution_time = ((t_end - t_start)/t_step)*execution_time
        print("Approx. total running time: %.2f minutes:" %total_execution_time)

    
    iter_n +=1

print("simulation completed !!!", time.strftime("%H:%M:%S", time.localtime()))
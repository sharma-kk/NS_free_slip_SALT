import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
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

mesh = bary_mesh_hier[-1]

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "DG", 1)

u_cg = Function(V)
u_det = Function(V)
u_adap = Function(V)

u_cg.assign(0)
u_det.assign(0)
u_adap.assign(0)

outfile = File("./results/deterministic_adapted_coarse_grained_fields_grid_64_t50_to_t90.pvd")

current_time = time.strftime("%H:%M:%S", time.localtime())
print("Local time at the start of simulation:",current_time)

def relative_l2_error(f_truth, f):
    """ compute the relative L2 error between two functions. See Wei's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return errornorm(f_truth, f)/norm(f_truth)

def pattern_correlation(f_truth, f):
    """ compute the relative pattern correlation between two sacaler valued functions. See Say's paper for definition.
    :arg f_truth: the function against which we want the comparison; coarse grained truth
    :arg f: the deterministic or the adapted reference solution
    """
    return assemble(inner(f, f_truth)*dx)/ np.sqrt(assemble(inner(f, f)*dx) * assemble(inner(f_truth, f_truth)*dx))

l2_error_vort_det_vs_truth = [] ; l2_error_vel_det_vs_truth = []
l2_error_vort_adap_vs_truth = [] ; l2_error_vel_adap_vs_truth = []

pattern_corr_vort_det_vs_truth = []
pattern_corr_vort_adap_vs_truth = []

for i in t_stamps:

    print(f'loading the velocity and temperature fields at t = {i}')

    data_cg = np.load('./fields_as_arrays/u_coarse_grained_arrays_mesh_64_at_t_'+str(i)+'.npz')
    data_det = np.load('./fields_as_arrays/u_deter_sim_arrays_mesh_64_at_t_'+str(i)+'.npz')
    data_ad = np.load('./fields_as_arrays/u_adapted_sol_arrays_mesh_64_at_t_'+str(i)+'.npz')

    vel_array_cg = data_cg['vel_array'] 
    vel_array_det = data_det['vel_array']
    vel_array_ad = data_ad['vel_array'] 

    u_cg.dat.data[:] = vel_array_cg
    u_det.dat.data[:] = vel_array_det
    u_adap.dat.data[:] = vel_array_ad

    vort_cg= interpolate(u_cg[1].dx(0) - u_cg[0].dx(1), W)
    vort_det= interpolate(u_det[1].dx(0) - u_det[0].dx(1), W)
    vort_adap= interpolate(u_adap[1].dx(0) - u_adap[0].dx(1), W)

    ### calculating the l2 error and pattern correlation; deterministic vs truth
    l2_error_vort_det_vs_truth.append(relative_l2_error(vort_cg, vort_det))
    l2_error_vel_det_vs_truth.append(relative_l2_error(u_cg, u_det))
    pattern_corr_vort_det_vs_truth.append(pattern_correlation(vort_cg, vort_det))

    ### calculating the l2 error and pattern correlation; adapted sol. vs truth
    l2_error_vort_adap_vs_truth.append(relative_l2_error(vort_cg, vort_adap))
    l2_error_vel_adap_vs_truth.append(relative_l2_error(u_cg, u_adap))
    pattern_corr_vort_adap_vs_truth.append(pattern_correlation(vort_cg, vort_adap))

    u_cg.rename('vel_truth') ; vort_cg.rename('vort_truth')
    u_det.rename('vel_det') ; vort_det.rename('vort_det')
    u_adap.rename('vel_adap') ; vort_adap.rename('vort_adap')

    if i in [50.0, 60.0, 70.0, 80.0, 90.0]:
        outfile.write(u_cg, vort_cg, u_det, vort_det, u_adap, vort_adap)

    data_file = './l2_error_pattern_cor_data/det_vs_adap_vs_truth_grid_64_t50_to_t90.npz'
    np.savez(data_file, l2_vort_det_v_truth = np.array(l2_error_vort_det_vs_truth)
             , pc_vort_det_v_truth = np.array(pattern_corr_vort_det_vs_truth)
             , l2_vort_adap_v_truth = np.array(l2_error_vort_adap_vs_truth)
             , pc_vort_adap_v_truth = np.array(pattern_corr_vort_adap_vs_truth)
             , l2_vel_det_v_truth = np.array(l2_error_vel_det_vs_truth), l2_vel_adap_v_truth = np.array(l2_error_vel_adap_vs_truth))

print("Local time at the end of simulation:",time.strftime("%H:%M:%S", time.localtime()))
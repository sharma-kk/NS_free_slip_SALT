import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np

n_ele = 64
base_mesh = UnitSquareMesh(n_ele,n_ele)

bary_mesh_hier = BaryMeshHierarchy(base_mesh, 0)

c_mesh = bary_mesh_hier[-1]
c_mesh.name = "coarse_mesh_64"

Vc = VectorFunctionSpace(c_mesh, "CG", 2)

xi = Function(Vc)

# 75 eigenvectors caputre 99 percent of the total variance.

xi_data = np.load('./calculated_xi_vectors/xi_matrix_75_eigvec_c_1_by_64_decor_t_1_Dt_t=50_to_t=110.npz')
xi_mat = xi_data['xi_mat']

outfile  = File("./results/xi_vecs_grid_64_decor_t_1Dt_t=60_to_t=110.pvd") # there is typo here. it should be ...t=50..pvd

n_xi = 75 # no. of xi you want to print

for i in np.arange(n_xi):

    print(f'saving xi {i+1}')
    print("Local time:",time.strftime("%H:%M:%S", time.localtime()))
    
    xi.assign(0)
    xi.dat.data[:] = xi_mat[i,:]

    xi.rename("xi")

    outfile.write(xi)

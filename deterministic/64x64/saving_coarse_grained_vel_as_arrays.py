import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import time
from bary import *
import numpy as np

print("loading coarse grained velocity...",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("./h5_files/coarse_grained_fields_at_t50_mesh_64_c_1by64.h5", 'r') as afile:
     mesh = afile.load_mesh("coarse_mesh_64")
     u_ = afile.load_function(mesh, "coarse_vel")

print("finished loading!",
    time.strftime("%H:%M:%S", time.localtime()))

data_file ='./coarse_grained_vel_vector/u_coarse_grained_vel_array_mesh_64_c_1by64_t50.npz'
np.savez(data_file, vel_array = u_.dat.data)
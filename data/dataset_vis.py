import pickle
from pathlib import Path

import numpy as np
import pywavefront

from utils import save_glb
from geometry2 import Patch

def generate_sample_vis(file_path:str="vase-lion100K"):

    file_path = "../generated_triplet_data/triplets_size_30_N_5_all_monge_patch_normalized_pos_and_rot_non_uniform_sampling.pkl"
    # Load the triplets from the file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    samples_num_to_vis = 2
    # generate samples_num_to_vis random index in data
    random_indices = np.random.choice(len(data), samples_num_to_vis, replace=False)
    for i in random_indices:
        # generate random number from 0,1,2 for checking whether to take the anchor, positive or negative
        # random_num = np.random.randint(0, 3)
        for j in range(3):
            data_sample = data[i][j]
            v = data_sample.v
            f = data_sample.f
            save_glb(vertices=v, faces=f, path=Path("../generated_triplet_data/triplets_vis/triplet"+str(i)+"anc_pos_neg"+str(j)+".glb"))


def generate_mesh_vis(v, f):
    save_glb(vertices=v, faces=f,
             path=Path("patch_generated.glb"))


generate_sample_vis()















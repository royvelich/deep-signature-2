import random

import torch
from torch.cuda import is_available
# in order to fix import errors on server
import sys
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and its parent directory to the Python path
sys.path.extend([current_dir, os.path.abspath(os.path.join(current_dir, '..'))])

# till here

from generation import QuadraticMonagePatchGenerator2, SimplexNoisePatchGenerator, InverseFourierPatchGenerator, \
    TorusGenerator

# from deep_signature.utils2 import delete_files_in_folder

from tqdm import tqdm
import pickle

dataset_reg_and_unreg = False
device = torch.device("cuda" if is_available() else "cpu")

grid_size = 30
limit = 1

patch_generator_anc_pos = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=True)
patch_generator_neg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=True)
# patch_generator_anc_pos = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
# patch_generator_neg = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
if dataset_reg_and_unreg:
    patch_generator_anc_pos_reg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=False)
    patch_generator_neg_reg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=False)
    # patch_generator_anc_pos_reg = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
    # patch_generator_neg_reg = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
# patch_generator_neg = InverseFourierPatchGenerator(limit=limit, grid_size=grid_size, scale=0.5)
parts = 100
N = 1000 # how many triplet of patches to generate

neg_noise_low = 0.5
neg_noise_high = 2.0

for i in tqdm(range(parts)):
    triplets = []
    for j in tqdm(range(N)):
        sample_anc, k1_anc, k2_anc, point0_0_anc = patch_generator_anc_pos.generate()
        sample_pos, k1_pos, k2_pos, point0_0_pos = patch_generator_anc_pos.generate(k1=k1_anc, k2=k2_anc)
        if dataset_reg_and_unreg:
            sample_anc_reg, k1_anc_reg, k2_anc_reg, point0_0_anc_reg = patch_generator_anc_pos_reg.generate()
            sample_pos_reg, k1_pos_reg, k2_pos_reg, point0_0_pos_reg = patch_generator_anc_pos_reg.generate(k1=k1_anc_reg, k2=k2_anc_reg)
        rand_num = random.uniform(0,1)
        if rand_num>=0.75:
            sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(neg_noise_low,neg_noise_high), k2=k2_anc+random.uniform(neg_noise_low,neg_noise_high))
            if dataset_reg_and_unreg:
                sample_neg_reg, k1_neg_reg, k2_neg_reg, point0_0_neg_reg = patch_generator_neg_reg.generate(k1=k1_anc_reg+random.uniform(neg_noise_low,neg_noise_high), k2=k2_anc_reg+random.uniform(neg_noise_low,neg_noise_high))
        elif rand_num<0.75 and rand_num>=0.5:
            sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(neg_noise_low,neg_noise_high), k2=k2_anc+random.uniform(-neg_noise_high,-neg_noise_low))
            if dataset_reg_and_unreg:
                sample_neg_reg, k1_neg_reg, k2_neg_reg, point0_0_neg_reg = patch_generator_neg_reg.generate(k1=k1_anc_reg+random.uniform(neg_noise_low,neg_noise_high), k2=k2_anc_reg+random.uniform(-neg_noise_high,-neg_noise_low))
        elif rand_num<0.5 and rand_num>=0.25:
            sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(-neg_noise_high,-neg_noise_low), k2=k2_anc+random.uniform(neg_noise_low, neg_noise_high))
            if dataset_reg_and_unreg:
                sample_neg_reg, k1_neg_reg, k2_neg_reg, point0_0_neg_reg = patch_generator_neg_reg.generate(k1=k1_anc_reg+random.uniform(-neg_noise_high,-neg_noise_low), k2=k2_anc_reg+random.uniform(neg_noise_low, neg_noise_high))
        else:
            sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(-neg_noise_high,-neg_noise_low), k2=k2_anc+random.uniform(-neg_noise_high,-neg_noise_low))
            if dataset_reg_and_unreg:
                sample_neg_reg, k1_neg_reg, k2_neg_reg, point0_0_neg_reg = patch_generator_neg_reg.generate(k1=k1_anc_reg+random.uniform(-neg_noise_high,-neg_noise_low), k2=k2_anc_reg+random.uniform(-neg_noise_high,-neg_noise_low))


        if not dataset_reg_and_unreg:
            triplets.append((sample_anc, sample_pos, sample_neg))
        else:
            triplets.append((sample_anc, sample_pos, sample_neg, sample_anc_reg, sample_pos_reg, sample_neg_reg))

    # Define the file path to save the triplets
    # file_path = "../generated_triplet_data/triplets_size_"+str(grid_size)+"_N_"+str(N)+"_all_monge_patch_normalized_pos_and_rot_non_uniform_sampling.pkl"
    file_path = "../triplets_dataset/triplets_size_"+str(grid_size)+"_N_"+str(N)+"_all_monge_patch_non_uniform_sampling_part"+str(i)+".pkl"

    if dataset_reg_and_unreg:
        # file_path = "../triplets_size_"+str(grid_size)+"_N_"+str(N)+"_all_monge_patch_normalized_pos_and_rot_80_per_fps_sampling_reg_and_unreg.pkl"
        file_path = "../triplets_size_"+str(grid_size)+"_N_"+str(N)+"_all_torus_normalized_pos_and_rot_80_per_fps_sampling_reg_and_unreg.pkl"

    # Save the triplets into a file
    with open(file_path, 'wb') as f:
        pickle.dump(triplets, f)


    # # check file
    # file_path = "triplets_data.pkl"
    #
    # # Load the triplets from the file
    # with open(file_path, 'rb') as f:
    #     triplets = pickle.load(f)
    #
    # print(triplets)



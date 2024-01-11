import argparse
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
    TorusGenerator, QuadraticMonageParabolicPlanarPatchGenerator, QuadraticMonagePatchPointCloudGenerator

# from deep_signature.utils2 import delete_files_in_folder

from tqdm import tqdm
import pickle




dataset_reg_and_unreg = False
is_triplets = False
triplet_file = True
device = torch.device("cuda" if is_available() else "cpu")

parser = argparse.ArgumentParser(description='Generate triplets of patches')
parser.add_argument('-N', type=int, default=10, help='Number of triplets to generate')
parser.add_argument('-grid_size', type=int, default=100, help='Number of triplets to generate')
parser.add_argument('-parts', type=int, default=1, help='Number of triplets to generate')
parser.add_argument('-limit', type=int, default=1, help='Number of triplets to generate')
parser.add_argument('-neg_noise_low', type=float, default=0.5, help='Number of triplets to generate')
parser.add_argument('-neg_noise_high', type=float, default=2.0, help='Number of triplets to generate')
parser.add_argument('-patch_type', type=str, default='parabolic', help='Number of triplets to generate')

# Parse the command-line arguments
args = parser.parse_args()
N = args.N
parts = args.parts
grid_size = args.grid_size
limit = args.limit
neg_noise_low = args.neg_noise_low
neg_noise_high = args.neg_noise_high
patch_type = args.patch_type

patch_generator_anc_pos = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)
# patch_generator_anc_pos = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=True)

patch_generator_neg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=True)
# patch_generator_anc_pos = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
# patch_generator_neg = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)

if dataset_reg_and_unreg:
    patch_generator_anc_pos_reg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=False)
    patch_generator_neg_reg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size, downsample=False)
    # patch_generator_anc_pos_reg = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
    # patch_generator_neg_reg = TorusGenerator(limit=limit, grid_size=grid_size, downsample=False)
# patch_generator_neg = InverseFourierPatchGenerator(limit=limit, grid_size=grid_size, scale=0.5)




if is_triplets:
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
        file_path = "../triplets_dataset/triplets_size_"+str(grid_size)+"_N_"+str(N)+"_all_monge_patch_non_uniform_sampling_with_parabolic_patches.pkl"

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

else:
    if triplet_file:
        outputs_spherical = []
        outputs_hyperbolic = []
        outputs_parabolic = []

        patch_generator = QuadraticMonagePatchPointCloudGenerator(limit=limit, grid_size=grid_size, downsample=False)

        for i in tqdm(range(N)):
            sample_spherical, k1, k2, point0_0 = patch_generator.generate(patch_type="spherical")
            rand_num = random.uniform(0,1)
            if rand_num<0.3:
                sample_hyperbolic, k1, k2, point0_0 = patch_generator.generate(patch_type="hyperbolic", k1=k1, k2=-k2)
            elif rand_num<0.6 and rand_num>=0.3:
                sample_hyperbolic, k1, k2, point0_0 = patch_generator.generate(patch_type="hyperbolic", k1=-k1, k2=k2)
            else:
                sample_hyperbolic, k1, k2, point0_0 = patch_generator.generate(patch_type="hyperbolic")
            rand_num2 = random.uniform(0,1)
            if rand_num2<=0.3:
                sample_parabolic, k1, k2, point0_0 = patch_generator.generate(patch_type="parabolic", k1=-k1, k2=0)
            elif rand_num2<=0.6 and rand_num2>0.3:
                sample_parabolic, k1, k2, point0_0 = patch_generator.generate(patch_type="parabolic", k1=0, k2=-k2)
            else:
                sample_parabolic, k1, k2, point0_0 = patch_generator.generate(patch_type="parabolic")

            outputs_spherical.append(sample_spherical)
            outputs_hyperbolic.append(sample_hyperbolic)
            outputs_parabolic.append(sample_parabolic)

        file_path_spherical = "./spherical_monge_patches_" + str(grid_size) + "_N_" + str(N) + ".pkl"
        file_path_hyperbolic = "./hyperbolic_monge_patches_" + str(grid_size) + "_N_" + str(N) + ".pkl"
        file_path_parabolic = "./parabolic_monge_patches_" + str(grid_size) + "_N_" + str(N) + ".pkl"

        with open(file_path_spherical, 'wb') as f:
            pickle.dump(outputs_spherical, f)
        with open(file_path_hyperbolic, 'wb') as f:
            pickle.dump(outputs_hyperbolic, f)
        with open(file_path_parabolic, 'wb') as f:
            pickle.dump(outputs_parabolic, f)

    else:
        outputs= []
        patch_generator = QuadraticMonagePatchPointCloudGenerator(limit=limit, grid_size=grid_size, downsample=False)
        for i in tqdm(range(N)):
            sample, k1, k2, point0_0 = patch_generator.generate(patch_type=patch_type)

            outputs.append(sample)

        file_path = "./"+str(patch_type)+"_monge_patches_" + str(grid_size) + "_N_" + str(N) + ".pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(outputs, f)

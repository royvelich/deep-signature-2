import random

import torch
from torch.cuda import is_available



from generation import QuadraticMonagePatchGenerator2, SimplexNoisePatchGenerator,InverseFourierPatchGenerator

# from deep_signature.utils2 import delete_files_in_folder

from tqdm import tqdm
import pickle


device = torch.device("cuda" if is_available() else "cpu")

grid_size = 50
limit = 1

patch_generator_anc_pos = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size)
patch_generator_neg = QuadraticMonagePatchGenerator2(limit=limit, grid_size=grid_size)
# patch_generator_neg = InverseFourierPatchGenerator(limit=limit, grid_size=grid_size, scale=0.5)
N = 10 # how many triplet of patches to train on

neg_noise_low = 0.1
neg_noise_high = 1.0

triplets = []

for i in tqdm(range(N)):
    sample_anc, k1_anc, k2_anc, point0_0_anc = patch_generator_anc_pos.generate()
    sample_pos, k1_pos, k2_pos, point0_0_pos = patch_generator_anc_pos.generate(k1=k1_anc, k2=k2_anc)
    rand_num = random.uniform(0,1)
    if rand_num>=0.75:
        sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(neg_noise_low,neg_noise_high), k2=k2_anc+random.uniform(neg_noise_low,neg_noise_high))
    elif rand_num<0.75 and rand_num>=0.5:
        sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(neg_noise_low,neg_noise_high), k2=k2_anc+random.uniform(-neg_noise_high,-neg_noise_low))
    elif rand_num<0.5 and rand_num>=0.25:
        sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(-neg_noise_high,-neg_noise_low), k2=k2_anc+random.uniform(neg_noise_low, neg_noise_high))
    else:
        sample_neg, k1_neg, k2_neg, point0_0_neg = patch_generator_neg.generate(k1=k1_anc+random.uniform(-neg_noise_high,-neg_noise_low), k2=k2_anc+random.uniform(-neg_noise_high,-neg_noise_low))



    triplets.append((sample_anc, sample_pos, sample_neg))

# Define the file path to save the triplets
file_path = "../triplets_data_size_"+str(grid_size)+"_N_"+str(N)+"_all_monge_patch_normalized_pos_and_rot.pkl"

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



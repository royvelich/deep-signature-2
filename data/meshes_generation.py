import random
from pathlib import Path

import aspose.threed
import torch
from torch.cuda import is_available



from generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator

# from deep_signature.utils2 import delete_files_in_folder

from tqdm import tqdm
import pickle

from utils import save_glb, fix_pathologies, save_obj

grid_size = 100
limit = 1
meshes_num = 30

# shape_types = ["peak", "saddle", "parabolic"]
shape_types = ["peak"]
types_num = 2
downsample = True

meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
# meshes_generator = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)
output = []


for shape_type in shape_types:
    for j in range(1,types_num+1):
        if shape_type == "parabolic" and j>1:
            continue
        shape_type_tmp = shape_type + str(j)
        for i in tqdm(range(meshes_num)):
            mesh = meshes_generator.generate(grid_size_delta=0, shape=shape_type_tmp)
            # mesh.set_faces(fix_pathologies(mesh.v, mesh.f))
            path = "../mesh_different_sampling/non_uniform/grid_size_100/"+shape_type_tmp+"_"+str(i)
            # path = "C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2/3d_vis\saddle30"
            file_path = Path(path+".obj")
            # save_glb(vertices=mesh.v, faces=mesh.f,  path=file_path)
            save_obj(vertices=mesh.v, faces=mesh.f,  path=file_path)
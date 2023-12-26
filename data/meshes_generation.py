import random
from pathlib import Path

import aspose.threed
import torch
from torch.cuda import is_available



from generation import PeakSaddleGenerator

# from deep_signature.utils2 import delete_files_in_folder

from tqdm import tqdm
import pickle

from utils import save_glb, fix_pathologies

grid_size = 300
limit = 1

shape_type = "peak3_non_uniform"
downsample = True

meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
output = []
for i in tqdm(range(10)):
    mesh = meshes_generator.generate(grid_size_delta=0, shape=shape_type)
    # mesh.set_faces(fix_pathologies(mesh.v, mesh.f))



    path = "../mesh_different_sampling/non_uniform/same_ratio/"+shape_type+str(grid_size)
    file_path = Path(path+".glb")



    save_glb(vertices=mesh.v, faces=mesh.f,  path=file_path)

    aspose.threed.Scene.from_file(path+".glb").save(path+str(i)+".obj")

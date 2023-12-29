import random
from pathlib import Path

import aspose.threed
import torch
from torch.cuda import is_available



from generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator

# from deep_signature.utils2 import delete_files_in_folder

from tqdm import tqdm
import pickle

from utils import save_glb, fix_pathologies

grid_size = 300
limit = 1
meshes_num = 1

shape_type = "peak"
downsample = True

meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
# meshes_generator = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)
output = []
for i in tqdm(range(meshes_num)):
    mesh = meshes_generator.generate(grid_size_delta=0, shape=shape_type)
    # mesh, _, _, _ = meshes_generator.generate()
    # mesh.set_faces(fix_pathologies(mesh.v, mesh.f))



    path = "../mesh_different_sampling/non_uniform/same_ratio/"+shape_type+str(grid_size)
    file_path = Path(path+".glb")



    save_glb(vertices=mesh.v, faces=mesh.f,  path=file_path)

    aspose.threed.Scene.from_file(path+".glb").save(path+str(i)+".obj")

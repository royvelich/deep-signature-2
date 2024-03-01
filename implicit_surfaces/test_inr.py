import torch

from data.generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator
import numpy as np
import open3d as o3d
from visualize_pointclouds import visualize_pointclouds2
from model import SIREN
from loss import calculate_dirichlet_energy, rand_differences_loss,DirichletEnergyLoss

grid_size = 7
limit = 1
downsample = False

meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
# meshes_generator = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)

mesh = meshes_generator.generate(grid_size_delta=0, shape="peak1")

# load model from file
hidden_layer_config =  [256, 256, 256]
model = SIREN(n_in_features=2, n_out_features=1, hidden_layer_config=hidden_layer_config)
model.load_state_dict(torch.load("inr_model_peak_patch.pth"))
model.eval()
# torch x,y from mesh.v
input_v = torch.stack([torch.tensor(mesh.v[:, 0], dtype=torch.float32), torch.tensor(mesh.v[:, 1], dtype=torch.float32)], dim=1)
# forward pass
output = model(input_v)
output_to_vis = np.stack([input_v[:, 0].detach().numpy(), input_v[:, 1].detach().numpy(), output['model_out'].detach().numpy().flatten()], axis=1)

rand_sampled_uv = torch.rand(1000, 2) * 2 - 1
rand_sampled_f_uv = model(rand_sampled_uv)['model_out']
rand_output_to_vis = np.stack([rand_sampled_uv[:, 0].detach().numpy(), rand_sampled_uv[:, 1].detach().numpy(), rand_sampled_f_uv.detach().numpy().flatten()], axis=1)

visualize_pointclouds2(pointcloud1=mesh.v, pointcloud2=rand_output_to_vis)


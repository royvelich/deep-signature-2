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
epochs = 3000

meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
# meshes_generator = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)

mesh = meshes_generator.generate(grid_size_delta=0, shape="peak1")

# calculate mesh mid point normal
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.v)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.f)
mesh_o3d.compute_vertex_normals()
mid_point_index = mesh.get_mid_point()
mid_point_normal = mesh_o3d.vertex_normals[mid_point_index]
# visualize_pointclouds2(pointcloud1=mesh.v, fps_indices=mid_point_index, vector_field_to_visualize=np.array(mesh_o3d.vertex_normals)[mid_point_index], arrow_scale=1.0)

# calculate the tangent plane projection of all the points on the mesh
f_uv = np.dot(mesh.v - mesh.v[mid_point_index], mid_point_normal)
input_v = mesh.v - f_uv[:, None] * mid_point_normal[None, :]
input_v = torch.stack([torch.tensor(input_v[:, 0], dtype=torch.float32), torch.tensor(input_v[:, 1], dtype=torch.float32)], dim=1)
f_uv = torch.tensor(f_uv, dtype=torch.float32).view(-1, 1)

# declare the model
hidden_layer_config =  [256, 256, 256]
model = SIREN(n_in_features=2, n_out_features=1, hidden_layer_config=hidden_layer_config)
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_dirichlet = DirichletEnergyLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1000, gamma=0.5)
# train the model on input_v
for epoch in range(epochs):
    # forward pass
    output = model(input_v)
    # backward pass
    # loss = torch.norm(output['model_out']-f_uv) ** 2 + 0.01 * rand_differences_loss(model)
    dxdy = torch.autograd.grad(output["model_out"], output["model_in"],grad_outputs=torch.ones_like(output["model_out"]), create_graph=True)
    dx = dxdy[0][:,0]
    dy = dxdy[0][:,1]
    loss = torch.norm(output['model_out']-f_uv)**2 + torch.norm(dx)**2 + torch.norm(dy)**2
    loss.backward()
    # update the weights
    model.optimizer.step()
    model.optimizer.zero_grad()
    # scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# save model
torch.save(model.state_dict(), "inr_model_peak_patch.pth")
# visualize the output
output = model(input_v)
output_to_vis = np.stack([input_v[:, 0].detach().numpy(), input_v[:, 1].detach().numpy(), output['model_out'].detach().numpy().flatten()], axis=1)

rand_sampled_uv = torch.rand(1000, 2) * 2 - 1
rand_sampled_f_uv = model(rand_sampled_uv)['model_out']
rand_output_to_vis = np.stack([rand_sampled_uv[:, 0].detach().numpy(), rand_sampled_uv[:, 1].detach().numpy(), rand_sampled_f_uv.detach().numpy().flatten()], axis=1)

visualize_pointclouds2(pointcloud1=mesh.v, pointcloud2=rand_output_to_vis, fps_indices=mid_point_index, vector_field_to_visualize=np.array(mesh_o3d.vertex_normals)[mid_point_index], arrow_scale=1.0)







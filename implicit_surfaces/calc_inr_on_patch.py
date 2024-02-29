import torch

from data.generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator
import numpy as np
import open3d as o3d
from visualize_pointclouds import visualize_pointclouds2
from model import SIREN

grid_size = 7
limit = 1
downsample = False
epochs = 1000

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
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# train the model on input_v
for epoch in range(epochs):
    # forward pass
    output = model(input_v)
    # backward pass
    loss = torch.norm(output['model_out']-f_uv)**2
    loss.backward()
    # update the weights
    model.optimizer.step()
    model.optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# visualize the output
output = model(input_v)
output_to_vis = np.stack([input_v[:, 0].detach().numpy(), input_v[:, 1].detach().numpy(), output['model_out'].detach().numpy().flatten()], axis=1)
visualize_pointclouds2(pointcloud1=output_to_vis, pointcloud2=mesh.v, fps_indices=mid_point_index, vector_field_to_visualize=np.array(mesh_o3d.vertex_normals)[mid_point_index], arrow_scale=1.0)







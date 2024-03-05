import torch

from data.generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator
import numpy as np
import open3d as o3d

from implicit_surfaces.differential_geometry_on_inr import calculate_shape_operator_and_principal_directions
# from visualize_pointclouds import visualize_pointclouds2
from model import SIREN
from loss import calculate_dirichlet_energy, rand_differences_loss,DirichletEnergyLoss, dirichlet_loss

grid_size = 15
limit = 1
downsample = False
epochs = 1000
shape_type = "saddle3"


meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
# meshes_generator = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)


mesh = meshes_generator.generate(grid_size_delta=0, shape=shape_type)

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
x_unit_vector = torch.tensor([1.0, 0.0], dtype=torch.float32)
y_unit_vector = torch.tensor([0.0, 1.0], dtype=torch.float32)
center_point_index = mesh.get_mid_point()

for epoch in range(epochs):
    # forward pass
    output = model(input_v)
    # output_center = {"model_out": output["model_out"][2].view(1, 1), "model_in": output["model_in"][2].view(1, 2)}


    # backward pass
    # loss = torch.norm(output['model_out']-f_uv) ** 2 + 0.01 * rand_differences_loss(model)
    dxdy = torch.autograd.grad(output["model_out"], output["model_in"],grad_outputs=torch.ones_like(output["model_out"]), create_graph=True)
    dx = dxdy[0][:, 0]
    dy = dxdy[0][:, 1]
    loss_reconstruction = torch.norm(output['model_out']-f_uv)**2 + 0.003*(torch.norm(dx)**2 + torch.norm(dy)**2 + 3*dirichlet_loss(model, gaussian_weights=True))
    if epoch > 500:
        e1, e2, grad = calculate_shape_operator_and_principal_directions(output, center_point_index)
        # rotate input_v and f_uv to align e1 and e2 with x and y unit vectors
        # e1 vs x_unit_vector angle differentiable

        with torch.no_grad():
            dot_product = torch.dot(e1, x_unit_vector)
            if dot_product<1.01 and dot_product>0.99 or dot_product>-1.01 and dot_product<-0.99:
                angle_to_rotate = torch.tensor(0.0)
            else:
                angle_to_rotate = torch.acos(dot_product)*0.01

            # angle_to_rotate = torch.atan2(e1[0] * x_unit_vector[1] - e1[1] * x_unit_vector[0], torch.dot(e1, x_unit_vector))

            # rotate input_v and f_uv to align e1 and e2 with x and y unit vectors rotate around z axis
        input_v = torch.stack([input_v[:, 0]*torch.cos(angle_to_rotate) - input_v[:, 1]*torch.sin(angle_to_rotate), input_v[:, 0]*torch.sin(angle_to_rotate) + input_v[:, 1]*torch.cos(angle_to_rotate)], dim=1)
        # f_uv = f_uv*torch.cos(angle_to_rotate) - f_uv*torch.sin(angle_to_rotate)
        # loss_principal_directions = torch.norm(e1-x_unit_vector) ** 2 + torch.norm(e2-y_unit_vector) ** 2 + torch.dot(e1, e2) ** 2
        loss_principal_directions = torch.norm(torch.dot(e1,x_unit_vector)-1) ** 2 + torch.norm(torch.dot(e2,y_unit_vector)-1) ** 2 + torch.norm(torch.dot(e1, e2)) ** 2
        loss = loss_reconstruction + 0.1*loss_principal_directions
    else:
        loss = loss_reconstruction
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    # update the weights
    model.optimizer.step()
    model.optimizer.zero_grad()
    # scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# save model
torch.save(model.state_dict(), "inr_model_"+shape_type+"_patch.pth")
# visualize the output
output = model(input_v)
output_to_vis = np.stack([input_v[:, 0].detach().numpy(), input_v[:, 1].detach().numpy(), output['model_out'].detach().numpy().flatten()], axis=1)

rand_sampled_uv = torch.rand(1000, 2) * 2 - 1
rand_sampled_f_uv = model(rand_sampled_uv)['model_out']
rand_output_to_vis = np.stack([rand_sampled_uv[:, 0].detach().numpy(), rand_sampled_uv[:, 1].detach().numpy(), rand_sampled_f_uv.detach().numpy().flatten()], axis=1)

# visualize_pointclouds2(pointcloud1=mesh.v, pointcloud2=rand_output_to_vis, fps_indices=mid_point_index, vector_fields_to_visualize=np.array(mesh_o3d.vertex_normals)[mid_point_index], arrow_scale=1.0)

# calculate_shape_operator_and_principal_directions(output, mesh)







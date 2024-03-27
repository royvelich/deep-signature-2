import torch

from data.generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator
import numpy as np
import open3d as o3d

from implicit_surfaces.differential_geometry_on_inr import calculate_shape_operator_and_principal_directions
# from visualize_pointclouds import visualize_pointclouds2
from implicit_surfaces.model import SIREN
from implicit_surfaces.loss import dirichlet_loss



grid_size = 20
limit = 1
downsample = False
epochs = 10000
shape_type = "random_order_2"
number_of_patches = 2
patches = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(number_of_patches):
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
    mesh.v = mesh.v - mesh.v[mid_point_index]
    f_uv = np.dot(mesh.v, mid_point_normal)
    input_v = mesh.v - f_uv[:, None] * mid_point_normal[None, :]
    # input_v = torch.stack([torch.tensor(input_v[:, 0], dtype=torch.float32), torch.tensor(input_v[:, 1], dtype=torch.float32)], dim=1)
    f_uv = torch.tensor(f_uv, dtype=torch.float32).view(-1, 1)

    sampled_vector_index = np.random.randint(0, len(input_v))
    # calculate the inner product of the sampled vector with all the other points and take the minimum which is not zero
    second_sampled_vector_index = np.argpartition(np.abs(np.dot(input_v, input_v[sampled_vector_index])), 1)[1]
    first_vector_base_tangent_plane = input_v[sampled_vector_index]/np.linalg.norm(input_v[sampled_vector_index])
    second_vector_base_tangent_plane = input_v[second_sampled_vector_index] - np.dot(input_v[second_sampled_vector_index], input_v[sampled_vector_index]) * input_v[sampled_vector_index]
    second_vector_base_tangent_plane = second_vector_base_tangent_plane/np.linalg.norm(second_vector_base_tangent_plane)
    input_v[:, 0] = np.dot(input_v, first_vector_base_tangent_plane)
    input_v[:, 1] = np.dot(input_v, second_vector_base_tangent_plane)

    input_v_torch = torch.stack(
        [torch.tensor(input_v[:, 0], dtype=torch.float32), torch.tensor(input_v[:, 1], dtype=torch.float32)], dim=1)
    f_uv = torch.tensor(f_uv, dtype=torch.float32).view(-1, 1)
    input_v_torch = input_v_torch.cuda()
    f_uv = f_uv.cuda()
    patches.append((input_v, f_uv, mid_point_index))

hidden_dim = 256
hidden_layer_config = [hidden_dim, hidden_dim]
base_model = SIREN(n_in_features=2, n_out_features=1, hidden_layer_config=hidden_layer_config)

x_unit_vector = torch.tensor([1.0, 0.0], dtype=torch.float32).to(device)
y_unit_vector = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)
center_point = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)

base_model = base_model.cuda()
base_model.optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5, weight_decay=0.001)

for epoch in range(epochs):
    for input_v_torch, f_uv, mid_point_index in patches:
        output = base_model(input_v_torch)
        # output_center = base_model(center_point.view(1, 2))
        # center_loss = torch.norm(output_center['model_out']) ** 2
        loss_reconstruction = torch.mean(torch.norm(output['model_out']-f_uv)**2)
        loss_smooth = dirichlet_loss(base_model)

        e1, e2, grad = calculate_shape_operator_and_principal_directions(output, mid_point=mid_point_index)
        e1 = e1.to(input_v_torch.device)
        e2 = e2.to(input_v_torch.device)


        dot_product = torch.dot(e1, x_unit_vector)
        print("dot_product", dot_product)

        loss_principal_directions = torch.norm(torch.dot(e1,x_unit_vector)-1) ** 2 + torch.norm(torch.dot(e2,y_unit_vector)-1) ** 2 + torch.norm(torch.dot(e1, e2)) ** 2
        loss = 1000.0*loss_reconstruction  + 1.0*loss_principal_directions + 0.001*loss_smooth


        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        # update the weights
        base_model.optimizer.step()
        base_model.optimizer.zero_grad()
        # scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    if epoch%100 == 0:
        torch.save(base_model.state_dict(), "base_inr_model_patch_with_reg_dirchlet_loss_iter.pth")


# save model
torch.save(base_model.state_dict(), "base_inr_model_patch_with_reg_dirchlet_loss_iter.pth")








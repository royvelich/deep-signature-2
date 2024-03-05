import igl
import torch

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import visualize_meshes
from data.generation import PeakSaddleGenerator
from visualize_pointclouds import visualize_pointclouds2
from visualize_meshes import visualize_meshes_func2, visualize_meshes_func3
from model import SIREN




def calculate_shape_operator_and_principal_directions(output, mid_point, mesh=None, return_pushed_forward_vectors= False):
    grad_f = torch.autograd.grad(output['model_out'], output["model_in"], grad_outputs=torch.ones_like(output['model_out']), create_graph=True)[0]

    dxdf = grad_f[:, 0]
    dydf = grad_f[:, 1]
    dxxdf_and_dxydf = torch.autograd.grad(dxdf, output["model_in"], grad_outputs=torch.ones_like(dxdf), create_graph=True)[0]
    dxydf_and_dyydf = torch.autograd.grad(dydf, output["model_in"], grad_outputs=torch.ones_like(dydf), create_graph=True)[0]
    # Hf = torch.stack([dxxdf_and_dxydf, dxydf_and_dyydf], dim=1)

    # calculate first fundamental form
    f_x = torch.stack([torch.ones_like(grad_f[:, 0]), torch.zeros_like(grad_f[:, 0]), grad_f[:,0]], dim=0).T
    f_y = torch.stack([torch.zeros_like(grad_f[:, 1]), torch.ones_like(grad_f[:, 1]), grad_f[:,1]], dim=0).T

    # Calculate the first fundamental form I for each point vectorized

    E = torch.sum(f_x * f_x, dim=1)
    F = torch.sum(f_x * f_y, dim=1)
    G = torch.sum(f_y * f_y, dim=1)
    I = torch.stack([E, F, F, G], dim=1).reshape(-1, 2, 2)

    # calculate the second fundamental form II
    # N = torch.nn.functional.normalize(torch.cross(f_x, f_y), dim=0)
    # mesh_o3d = o3d.geometry.TriangleMesh()
    # mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.v)
    # mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.f)
    # mesh_o3d.compute_vertex_normals()
    # normals = torch.tensor(mesh_o3d.vertex_normals, dtype=torch.float32)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(mesh.v)
    #
    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # point_cloud.orient_normals_consistent_tangent_plane(6)
    # # point_cloud.orient_normals_towards_camera_location([0, 0, 1])
    # normals = np.asarray(point_cloud.normals)
    # normals = torch.tensor(normals, dtype=torch.float32)
    normals = torch.nn.functional.normalize(torch.cross(f_x, f_y), dim=1)

    f_xx = torch.stack([torch.zeros_like(dxxdf_and_dxydf[:, 0]), torch.zeros_like(dxxdf_and_dxydf[:, 0]), dxxdf_and_dxydf[:, 0]], dim=0).T
    f_xy = torch.stack([torch.zeros_like(dxydf_and_dyydf[:, 1]), torch.zeros_like(dxydf_and_dyydf[:, 1]), dxxdf_and_dxydf[:, 1]], dim=0).T
    f_yy = torch.stack([torch.zeros_like(dxydf_and_dyydf[:, 1]), torch.zeros_like(dxydf_and_dyydf[:, 1]), dxydf_and_dyydf[:, 1]], dim=0).T

    L = torch.sum(f_xx * normals, dim=1)
    M = torch.sum(f_xy * normals, dim=1)
    N = torch.sum(f_yy * normals, dim=1)
    II = torch.stack([L, M, M, N], dim=1).reshape(-1, 2, 2)

    # Calculate the shape operator S
    S = -torch.inverse(I) @ II

    # calculate the principal directions
    eigenvalues, eigenvectors = torch.linalg.eigh(S)

    # plot the eigenvectors of each point on a 2d plot using plt
    # eigenvectors = eigenvectors.reshape(-1, 2, 2)
    # eigenvalues = eigenvalues.reshape(-1, 2)
    # eigenvectors = eigenvectors.detach().numpy()
    # fig, ax = plt.subplots()
    # ax.quiver(mesh.v[:, 0], mesh.v[:, 1], eigenvectors[:, 0, 0], eigenvectors[:, 0, 1], color='r')
    # ax.quiver(mesh.v[:, 0], mesh.v[:, 1], eigenvectors[:, 1, 0], eigenvectors[:, 1, 1], color='b')
    # ax.set_aspect('equal')
    # plt.show()

    # calculate the differential that pushes the principal directions to 3D
    # stack f_x and f_y to get the basis of the tangent space
    tangent_basis = torch.stack([f_x, f_y], dim=1)

    e1_e2 = torch.einsum('ijk,ikl->ijl', eigenvectors, tangent_basis)
    e1 = e1_e2[:, 0, :]
    e2 = e1_e2[:, 1, :]
    # normalize e1 and e2
    e1 = e1 / torch.linalg.norm(e1, axis=1)[:, None]
    e2 = e2 / torch.linalg.norm(e2, axis=1)[:, None]

    # center point index
    center_point_index = mid_point
    # igl_e1, igl_e2, igl_k1, igl_k2 = igl.principal_curvature(mesh.v, mesh.f)
    # # change from float64 to float32 in np
    # igl_e1 = igl_e1.astype(np.float32)
    # igl_e2 = igl_e2.astype(np.float32)

    # inr surface reconstruction
    # rand_sampled_uv = torch.rand(1000, 2) * 2 - 1
    # rand_sampled_f_uv = model(rand_sampled_uv)['model_out']
    # rand_output_to_vis = np.stack([rand_sampled_uv[:, 0].detach().numpy(), rand_sampled_uv[:, 1].detach().numpy(),
    #                                rand_sampled_f_uv.detach().numpy().flatten()], axis=1)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(rand_output_to_vis)
    #
    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # surface_reconstruction = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, 0.5)
    # create
    # visualize with wireframe of triangle mesh and lightning and shader
    # o3d.visualization.draw_geometries([surface_reconstruction], mesh_show_wireframe=True, mesh_show_back_face=True, mesh_show_normal=True)
    # mesh_to_plot = (np.array(surface_reconstruction.vertices, dtype=np.float32), np.array(surface_reconstruction.triangles, dtype=np.float32))


    # visualize_pointclouds2(mesh.v, vector_fields_to_visualize=e1, vector_field_to_visualize2=e2, fps_indices=np.arange(0, len(mesh.v)), arrow_scale=0.1)
    # visualize_pointclouds2(mesh.v, vector_fields_to_visualize=[e1[center_point_index], e2[center_point_index], igl_e1[center_point_index], igl_e2[center_point_index]], fps_indices=center_point_index, arrow_scale=0.1)
    # visualize_meshes.visualize_meshes_func(mesh_to_plot, vector_fields_to_visualize=[e1[center_point_index], e2[center_point_index], igl_e1[center_point_index], igl_e2[center_point_index]], fps_indices=center_point_index, arrow_scale=0.1)
    # using igl to calculate principal curvatures and directions
    if return_pushed_forward_vectors:
        return e1[center_point_index], e2[center_point_index], grad_f

    return eigenvectors[center_point_index, :, 0], eigenvectors[center_point_index, :, 1], grad_f


def sanity_check():
    grid_size = 15
    limit = 1
    downsample = False

    shape_type = "saddle3"
    model_path = "inr_model_saddle3_patch.pth"

    # meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=grid_size, downsample=downsample)
    # meshes_generator = QuadraticMonageParabolicPlanarPatchGenerator(limit=limit, grid_size=grid_size, downsample=True)

    # mesh = meshes_generator.generate(grid_size_delta=0, shape=shape_type)

    # want to calculate principal directions using shape operator on inr
    # for that we need to calculate the shape operator

    # load model from file
    hidden_layer_config = [256, 256, 256]
    model = SIREN(n_in_features=2, n_out_features=1, hidden_layer_config=hidden_layer_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rand_sampled_uv = torch.rand(1000, 2) * 2 - 1
    # concatante 1 point 0,0
    # rand_sampled_uv = torch.cat([rand_sampled_uv, torch.tensor([[0, 0]], dtype=torch.float32)], dim=0)
    # rand_sampled__output = model(rand_sampled_uv)
    # rand_sampled_f_uv = rand_sampled__output['model_out']
    # rand_output_to_vis = np.stack([rand_sampled_uv[:, 0].detach().numpy(), rand_sampled_uv[:, 1].detach().numpy(),
    #                                rand_sampled_f_uv.detach().numpy().flatten()], axis=1)
    # sample reugular grid
    regular_sampled_uv = torch.tensor(np.mgrid[-1:1:0.01, -1:1:0.01].reshape(2, -1).T, dtype=torch.float32)
    regular_sampled__output = model(regular_sampled_uv)
    regular_sampled_f_uv = regular_sampled__output['model_out']
    regular_output_to_vis = np.stack([regular_sampled_uv[:, 0].detach().numpy(), regular_sampled_uv[:, 1].detach().numpy(),
                                        regular_sampled_f_uv.detach().numpy().flatten()], axis=1)




    # input_v = torch.stack(
    #     [torch.tensor(mesh.v[:, 0], dtype=torch.float32), torch.tensor(mesh.v[:, 1], dtype=torch.float32)], dim=1)

    # output = model(input_v)
    # mid point
    # index_of_point_closest_to_zero = np.argmin(rand_sampled_uv[:, 0].detach().numpy() ** 2 + rand_sampled_uv[:, 1].detach().numpy() ** 2+ rand_sampled_f_uv.detach().numpy().flatten() ** 2)
    # e1, e2, grad = calculate_shape_operator_and_principal_directions(rand_sampled__output, index_of_point_closest_to_zero, return_pushed_forward_vectors=True)


    regular_index_of_point_closest_to_zero = np.argmin(regular_sampled_uv[:, 0].detach().numpy() ** 2 + regular_sampled_uv[:, 1].detach().numpy() ** 2+ regular_sampled_f_uv.detach().numpy().flatten() ** 2)
    e1, e2, grad = calculate_shape_operator_and_principal_directions(regular_sampled__output, regular_index_of_point_closest_to_zero, return_pushed_forward_vectors=True)

    visualize_pointclouds2(regular_output_to_vis, vector_fields_to_visualize=[e1.detach().numpy(), e2.detach().numpy()], fps_indices=regular_index_of_point_closest_to_zero, arrow_scale=0.1)

# sanity_check()
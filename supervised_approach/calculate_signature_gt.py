import torch
import numpy as np

import igl
import open3d as o3d
import trimesh
import pyvista as pv
import scipy as sp
from matplotlib import pyplot as plt

from data.generation import PeakSaddleGenerator
from torch_geometric.nn import fps
from visualize_pointclouds import visualize_pointclouds2

def naive_consistent_dir(verts, d):
    k_num = 6
    overall_count = 0
    idx = sp.spatial.cKDTree(verts).query(verts, k=k_num)[1]
    for i in range(len(verts)):
        count = 0
        flip_dir = []
        for j in idx[i]:
            if np.dot(d[i], d[j]) < 0:
                count += 1
            else:
                flip_dir.append(j)

        if count > k_num/2:
            d[i] = -d[i]
            for j in flip_dir:
                d[j] = -d[j]
            overall_count += count

    print(overall_count)
    return d







def consistent_principal_directions_using_normals(verts, dir1, dir2):
    """
    Calculate the consistent principal directions for a given mesh.

    Parameters
    ----------
    verts : np.array
        Vertices of the mesh.
    dir : np.array
        Direction of the mesh.

    Returns
    -------
    np.array
        Consistent principal directions of the mesh.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(verts)

    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud.orient_normals_consistent_tangent_plane(10)
    # point_cloud.orient_normals_towards_camera_location([0, 0, 1])
    normals = np.asarray(point_cloud.normals)
    # calculate the dot product between the normals and the direction
    normal_according_to_dir = np.cross(dir1, dir2)
    dot = np.sum(normals * normal_according_to_dir, axis=1)
    dot = dot.reshape(-1, 1)
    # if the dot product is negative, then the direction is not consistent with the normals
    # so we invert the direction
    dir1 = np.where(dot < 0, -dir1, dir1)
    dir2 = np.where(dot < 0, -dir2, dir2)
    return dir1, dir2





def der_on_mesh(v, g, d, h):
    # for each vertex v_i calculate the nearest neighbor s.t. v_j=argmin_{v_j} ||v_i+hd-v_j||
    v_step = v + h * d
    v_min_step = v - h * d
    # calculate the nearest neighbor that cannot be the same vertex
    idx = sp.spatial.cKDTree(v).query(v_step,k=2)[1]
    idx_min = sp.spatial.cKDTree(v).query(v_min_step,k=2)[1]
    # can still be the same vertex - how to avoid this?
    # need to check if for each index i, idx[i,0] != i if so, then idx[i,0] is the nearest neighbor else idx[i,1] is the nearest neighbor
    a = np.where(idx == np.arange(len(v)).reshape(-1, 1))
    idx[a[0], a[1]] = idx[a[0], np.abs(a[1] - 1)]
    idx = idx[:, 0]
    b = np.where(idx_min == np.arange(len(v)).reshape(-1, 1))
    idx_min[b[0], b[1]] = idx_min[b[0], np.abs(b[1] - 1)]
    idx_min = idx_min[:, 0]
    g_der = (g[idx] - g[idx_min]) / (2 * h)
    # g_der = (g[idx] - g) / h
    return g_der


def calculate_mean_curvature_based_signature(verts, faces):
    """
    Calculate the mean curvature for a given mesh.

    Parameters
    ----------
    verts : np.array
        Vertices of the mesh.
    faces : np.array
        Faces of the mesh.

    Returns
    -------
    torch.Tensor
        Mean curvature of the mesh.
    """
    # calculate k1 and k2
    # as far as i understood, radius=5 means radius of the neighborhood is 5*average_edge_length
    d1, d2, k1, k2 = igl.principal_curvature(verts, faces, radius=3)
    d1 = d1 / np.linalg.norm(d1, axis=1)[:, np.newaxis]
    d2 = d2 / np.linalg.norm(d2, axis=1)[:, np.newaxis]
    # d1, d2 = consistent_principal_directions_using_normals(verts, d1, d2)
    d1 = naive_consistent_dir(verts, d1)
    d2 = naive_consistent_dir(verts, d2)
    # calculate mean curvature
    mean_curvature = torch.tensor((k1 + k2) / 2)
    # calulate the mean edge length
    e = igl.edge_lengths(verts, faces)
    mean_edge_length = np.mean(e)
    # mean_edge_length=0.001 - sanity check, got all 0s
    const = 1.0
    mean_curvature_d1 = der_on_mesh(verts, mean_curvature, d1, const*mean_edge_length)
    mean_curvature_d2 = der_on_mesh(verts, mean_curvature, d2, const*mean_edge_length)
    mean_curvature_d11 = der_on_mesh(verts, mean_curvature_d1, d1, const*mean_edge_length)
    signature = torch.cat((mean_curvature[:, np.newaxis], mean_curvature_d1[:, np.newaxis],
                           mean_curvature_d2[:, np.newaxis], mean_curvature_d11[:, np.newaxis]), dim=1)
    return signature, d1, d2
















# load some mesh
# v = torch.tensor(v, dtype=torch.float32)
# load obj file
# v, f = igl.read_triangle_mesh("C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2\mesh_different_sampling\peak2306.obj")

# calculate_mean_curvature_based_signature(v,f)

limit = 1
grid_size = 100
shape_type_tmp = "peak1"
downsample = False

meshes_generator = PeakSaddleGenerator(limit=limit, grid_size=int(grid_size), downsample=downsample)

mesh = meshes_generator.generate(grid_size_delta=0, shape=shape_type_tmp)

signature1, d1, d2 = calculate_mean_curvature_based_signature(mesh.v, mesh.f)

meshes_generator2 = PeakSaddleGenerator(limit=limit, grid_size=grid_size*2, downsample=downsample)

mesh2 = meshes_generator2.generate(grid_size_delta=0, shape=shape_type_tmp)

signature2,mesh_2_d1, mesh_2d2 = calculate_mean_curvature_based_signature(mesh2.v, mesh2.f)

# sample the same xyz points from both meshes using fps
# sample 1000 points from the first mesh and then take the 1000 nearest neighbors from the second mesh
fps_sampled = fps(torch.tensor(mesh.v), ratio=0.1)
mesh2_corresponding_indices = sp.spatial.cKDTree(mesh2.v).query(mesh.v[fps_sampled])[1]
visualize_pointclouds2(mesh.v, vector_field_to_visualize=d1[fps_sampled], fps_indices=fps_sampled, arrow_scale=0.01)
visualize_pointclouds2(mesh2.v, vector_field_to_visualize=mesh_2_d1[mesh2_corresponding_indices], fps_indices=mesh2_corresponding_indices, arrow_scale=0.01)
# plot for each signaure1[:,i] and signature2[:,i] the function signature1[:,i][fps_sampled] and signature2[:,i][mesh2_corresponding_indices]
for i in range(signature1.shape[1]):
    plt.title(f"signature {i} - mesh {len(mesh.v)} vs mesh {len(mesh2.v)}")
    plt.scatter(signature1[:,i][fps_sampled], signature2[:,i][mesh2_corresponding_indices])
    # plot also y=x line in red
    plt.plot(np.linspace(min(signature1[:,i][fps_sampled]), max(signature1[:,i][fps_sampled]), len(fps_sampled)), np.linspace(min(signature2[:,i][mesh2_corresponding_indices]), max(signature2[:,i][mesh2_corresponding_indices]), len(fps_sampled)), 'r')
    plt.show()



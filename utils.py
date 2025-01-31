import os

import igl
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from pyvista import PolyData
from sklearn.neighbors import KDTree

# import pyvista as pv


def rearange_mesh_faces(faces):
    faces = faces[np.arange(len(faces)) % 4 != 0]
    faces = faces.reshape(-1, 3)
    return faces

def calc_dki_j(points, desired_principal_dir, k_i, delta):
    """
    calculate derivative of k_i in the direction of desired_principal_dir
    :param points:
    :param desired_principal_dir:
    :param k_i:
    :param delta:
    :return:
    """
    output = np.zeros((len(points), 1))
    A = KDTree(points)
    for i in range(len(points)):
        p = points[i]
        p_delta = p + delta * desired_principal_dir[i]
        p_delta = p_delta[np.newaxis, :]
        p_minus_delta = p - delta * desired_principal_dir[i]
        p_minus_delta = p_minus_delta[np.newaxis, :]
        # calculate closest point to p_delta using kd-tree
        closest_point_p_delta = A.query(p_delta,k=2)
        dist_p_delta = closest_point_p_delta[0][0][0]
        p_delta = closest_point_p_delta[1][0][0]
        # calculate closest point to p_minus_delta
        closest_point_p_minus_delta = A.query(p_minus_delta, k=2)
        dist_p_minus_delta = closest_point_p_minus_delta[0][0][0]
        p_minus_delta = closest_point_p_minus_delta[1][0][0]
        # check if the closest point is the same as the point we are calculating the curvature for
        if closest_point_p_delta[1][0][0] == i:
            # dist_p_delta = closest_point_p_delta[0][0][1]
            p_delta = closest_point_p_delta[1][0][1]
        if closest_point_p_minus_delta[1][0][0] == i:
            # dist_p_minus_delta = closest_point_p_minus_delta[0][0][1]
            p_minus_delta = closest_point_p_minus_delta[1][0][1]



        output[i] = (k_i[p_delta] - k_i[p_minus_delta]) / (np.linalg.norm(points[p_delta] - points[p_minus_delta]))

    return output

def plot_mesh_and_color_by_k_or_dki_j(surf : PolyData, k, title):
    surf.plot(scalars=k, show_edges=False, text=title, cmap='jet', cpos='xy', screenshot='k.png')



# def plot_mesh_with_vector_field(surf : PolyData,visualization_surf : PolyData, k,  principal_directions1, principal_directions2,title):
#     plotter = pv.Plotter()
#     visualization_surf['principal_directions1'] = principal_directions1
#     visualization_surf['principal_directions2'] = principal_directions2
#     glyphs1 = visualization_surf.glyph(orient='principal_directions1', scale=False, factor=0.1)
#     glyphs2 = visualization_surf.glyph(orient='principal_directions2', scale=False, factor=0.1)
#     plotter.add_mesh(surf, scalars=k, show_edges=False)
#
#     plotter.add_mesh(glyphs1, color='blue', opacity=1)
#     plotter.add_mesh(glyphs2, color='red', opacity=1)
#     plotter.add_title(title=title)
#     plotter.show(screenshot='k.png')



# def generate_surface(grid_points=200):
#     grid_points_count = grid_points
#     grid_linspace = np.linspace(-1, 1, grid_points_count)
#     grid_x, grid_y = np.meshgrid(grid_linspace, grid_linspace)
#     points = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(shape=grid_points_count * grid_points_count)],
#                          axis=1)
#
#     points_tensor = torch.tensor(points)
#
#     # make the patch more irregular and more real world like
#     # indices = fps(x=points_tensor, ratio=0.1)
#     # sampled_points = points_tensor[indices]
#     # sampled_points = sampled_points.cpu().detach().numpy()
#     # sampled_points[:, 2] = 0.1*numpy.sin(2*numpy.pi*sampled_points[:, 0]*0.5) + numpy.cos(4*numpy.pi*sampled_points[:, 1]*0.1)
#     points_tensor = points_tensor.cpu().detach().numpy()
#     # scalar rand between 0 to 100
#     a = np.random.rand()/2
#     b = np.random.rand()/2
#     points_tensor[:, 2] = a* np.sin(2 * np.pi * points_tensor[:, 0] * 0.5) + b*np.cos(
#         4 * np.pi * points_tensor[:, 1] * 0.1)
#     # points_tensor[:, 2] = 0.1 * np.sin(2 * np.pi * points_tensor[:, 0] * 0.5) + np.cos(
#     #     4 * np.pi * points_tensor[:, 1] * 0.1)
#     # points_tensor[:, 2] = points_tensor[:, 0]**2 + points_tensor[:, 1]**2
#
#     cloud = pv.PolyData(points_tensor)
#     # cloud.plot(point_size=5)
#
#     surf = cloud.delaunay_2d()
#     # surf.plot(show_edges=True)
#
#     v = surf.points
#     f = surf.faces
#     f = rearange_mesh_faces(f)
#     return surf, v,f

def calculate_derivatives(v, f,delta=0.01):
    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    dk1_1 = calc_dki_j(v, v1, k1, delta)

    dk1_2 = calc_dki_j(v, v2, k1, delta)

    dk2_1 = calc_dki_j(v, v1, k2, delta)

    dk2_2 = calc_dki_j(v, v2, k2, delta)

    dk1_22 = calc_dki_j(v, v2, dk1_2, delta)

    dk2_11 = calc_dki_j(v, v1, dk2_1, delta)
    return k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11


def calculate_pearson_corr_matrix(k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11):



    # plot_mesh_and_color_by_k_or_dki_j(surf, k1, "k1")
    # plot_mesh_and_color_by_k_or_dki_j(surf, k2, "k2")


#     calculate pearson correlation matrix
    pearson_corr_matrix = np.corrcoef(np.hstack([k1[:,np.newaxis],k2[:,np.newaxis],dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11]),rowvar=False)

    return pearson_corr_matrix


def init_wandb(lr,max_epochs, weight_decay):
    # Set a custom temporary directory for WandB
    wandb_dir = "./wandb_tmp_dir"
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_TEMP"] = wandb_dir

    wandb.login(key="fbd4729419e01772f8a97c41e71e422c6311e896")
    wandb.init(project="train_on_patches",
               # entity="geometric-dl",
               config={
                   "learning_rate": lr,
                   "architecture": "Point Net Max pool",
                   "dataset": "100000-generated-patch-triplets anc,pos,neg - monge patches",
                   "epochs": max_epochs,
                   "weight_decay": weight_decay
               })

    return WandbLogger()

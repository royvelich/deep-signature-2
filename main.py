import igl
import numpy as np
import pyvista
import torch
from torch_geometric.nn import fps

import dataset_generator
from utils import rearange_mesh_faces, calc_dki_j, plot_mesh_and_color_by_k_or_dki_j, plot_mesh_with_vector_field, \
    generate_surface, calculate_pearson_corr_matrix, calculate_derivatives


def main():
    grid_points_count = 200

    delta = (1/grid_points_count) * 20
    surf, v, f = generate_surface(grid_points_count)
    k1_total,k2_total, dk1_1_total, dk1_2_total, dk2_1_total, dk2_2_total, dk1_22_total, dk2_11_total = calculate_derivatives(v, f, delta)


    for i in range(10):
        surf, v, f =generate_surface(grid_points_count)
        k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11 = calculate_derivatives(v, f, delta)
        k1_total= np.concatenate([k1_total, k1])
        k2_total = np.concatenate([k2_total, k2])
        dk1_1_total= np.concatenate([dk1_1_total, dk1_1])
        dk1_2_total = np.concatenate([dk1_2_total, dk1_2])
        dk2_1_total = np.concatenate([dk2_1_total, dk2_1])
        dk2_2_total = np.concatenate([dk2_2_total, dk2_2])
        dk1_22_total = np.concatenate([dk1_22_total, dk1_22])
        dk2_11_total = np.concatenate([dk2_11_total, dk2_11])
    calculate_pearson_corr_matrix(k1_total, k2_total, dk1_1_total, dk1_2_total, dk2_1_total, dk2_2_total, dk1_22_total, dk2_11_total)
    # calculate_pearson_corr_matrix(v, f, delta=delta)

    # format the faces in the right format
    surf.plot( show_edges=False, cmap='jet', cpos='xy', screenshot='k.png')

    # generate less dense surface for the vector field visualization
    surf_vec_field, v_surf_vec_field, f_surf_vec_field = generate_surface(30)

    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    v_less_dense1, v_less_dense2, k_less_dense1, k_less_dense2 = igl.principal_curvature(v_surf_vec_field, f_surf_vec_field)

    plot_mesh_with_vector_field(surf, surf_vec_field, k1,v_less_dense1 , v_less_dense2, title="k1")
    plot_mesh_with_vector_field(surf, surf_vec_field, k2,v_less_dense1 , v_less_dense2, title="k2")

    # plot_mesh_and_color_by_k_or_dki_j(surf, k1, "k1")
    # plot_mesh_and_color_by_k_or_dki_j(surf, k2, "k2")
    dk1_1 = calc_dki_j(v, v1, k1, delta)
    # plot_mesh_and_color_by_k_or_dki_j(surf, dk1_1, "dk1_1")
    plot_mesh_with_vector_field(surf, surf_vec_field, dk1_1,v_less_dense1 , v_less_dense2, title="dk1_1")

    dk1_2 = calc_dki_j(v, v2, k1, delta)
    plot_mesh_with_vector_field(surf, surf_vec_field, k2,v_less_dense1 , v_less_dense2, title="dk1_2")

    # plot_mesh_and_color_by_k_or_dki_j(surf, dk1_2, "dk1_2")
    dk2_1 = calc_dki_j(v, v1, k2, delta)
    plot_mesh_with_vector_field(surf, surf_vec_field, k2,v_less_dense1 , v_less_dense2, title="dk2_1")

    # plot_mesh_and_color_by_k_or_dki_j(surf, dk2_1, "dk2_1")
    dk2_2 = calc_dki_j(v, v2, k2, delta)
    plot_mesh_with_vector_field(surf, surf_vec_field, k2,v_less_dense1 , v_less_dense2, title="dk2_2")

    # plot_mesh_and_color_by_k_or_dki_j(surf, dk2_2, "dk2_2")
    dk1_22 = calc_dki_j(v, v2, dk1_2, delta)
    plot_mesh_with_vector_field(surf, surf_vec_field, k2,v_less_dense1 , v_less_dense2, title="dk1_22")

    # plot_mesh_and_color_by_k_or_dki_j(surf, dk1_22, "dk1_22")
    dk2_11 = calc_dki_j(v, v1, dk2_1, delta)
    plot_mesh_with_vector_field(surf, surf_vec_field, k2,v_less_dense1 , v_less_dense2, title="dk2_11")

    # plot_mesh_and_color_by_k_or_dki_j(surf, dk2_11, "dk2_11")

if __name__ == "__main__":
    main()
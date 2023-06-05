import igl
import numpy
import pyvista
import torch
from torch_geometric.nn import fps

from utils import rearange_mesh_faces, calc_dki_j, plot_mesh_and_color_by_k_or_dki_j


def main():
    grid_points_count = 200
    grid_linspace = numpy.linspace(-1, 1, grid_points_count)
    grid_x, grid_y = numpy.meshgrid(grid_linspace, grid_linspace)
    points = numpy.stack([grid_x.ravel(), grid_y.ravel(), numpy.zeros(shape=grid_points_count*grid_points_count)], axis=1)

    points_tensor = torch.tensor(points)

    # make the patch more irregular and more real world like
    # indices = fps(x=points_tensor, ratio=0.1)
    # sampled_points = points_tensor[indices]
    # sampled_points = sampled_points.cpu().detach().numpy()
    # sampled_points[:, 2] = 0.1*numpy.sin(2*numpy.pi*sampled_points[:, 0]*0.5) + numpy.cos(4*numpy.pi*sampled_points[:, 1]*0.1)
    points_tensor = points_tensor.cpu().detach().numpy()
    points_tensor[:, 2] = 0.1*numpy.sin(2*numpy.pi*points_tensor[:, 0]*0.5) + numpy.cos(4*numpy.pi*points_tensor[:, 1]*0.1)
    cloud = pyvista.PolyData(points_tensor)
    # cloud.plot(point_size=5)

    surf = cloud.delaunay_2d()
    # surf.plot(show_edges=True)

    v = surf.points
    f = surf.faces
    f = rearange_mesh_faces(f)
    # format the faces in the right format


    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    plot_mesh_and_color_by_k_or_dki_j(surf, k1, "k1")
    plot_mesh_and_color_by_k_or_dki_j(surf, k2, "k2")
    delta = (1/grid_points_count) * 20
    dk1_1 = calc_dki_j(v, v1, k1, delta)
    # plot_mesh_and_color_by_k_or_dki_j(surf, dk1_1, "dk1_1")

    dk1_2 = calc_dki_j(v, v2, k1, delta)
    plot_mesh_and_color_by_k_or_dki_j(surf, dk1_2, "dk1_2")
    dk2_1 = calc_dki_j(v, v1, k2, delta)
    plot_mesh_and_color_by_k_or_dki_j(surf, dk2_1, "dk2_1")
    dk2_2 = calc_dki_j(v, v2, k2, delta)
    plot_mesh_and_color_by_k_or_dki_j(surf, dk2_2, "dk2_2")
    dk1_22 = calc_dki_j(v, v2, dk1_2, delta)
    plot_mesh_and_color_by_k_or_dki_j(surf, dk1_22, "dk1_22")
    dk2_11 = calc_dki_j(v, v1, dk2_1, delta)
    plot_mesh_and_color_by_k_or_dki_j(surf, dk2_11, "dk2_11")

if __name__ == "__main__":
    main()
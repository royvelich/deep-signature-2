import igl
import numpy
import pyvista
import torch
from torch_geometric.nn import fps

from utils import rearange_mesh_faces, calc_dk1_1


def main():
    grid_points_count = 200
    grid_linspace = numpy.linspace(-1, 1, grid_points_count)
    grid_x, grid_y = numpy.meshgrid(grid_linspace, grid_linspace)
    points = numpy.stack([grid_x.ravel(), grid_y.ravel(), numpy.zeros(shape=grid_points_count*grid_points_count)], axis=1)

    points_tensor = torch.tensor(points)
    indices = fps(x=points_tensor, ratio=0.1)
    sampled_points = points_tensor[indices]
    sampled_points = sampled_points.cpu().detach().numpy()
    sampled_points[:, 2] = 0.1*numpy.sin(2*numpy.pi*sampled_points[:, 0]*0.5) + numpy.cos(4*numpy.pi*sampled_points[:, 1]*0.1)

    cloud = pyvista.PolyData(sampled_points)
    # cloud.plot(point_size=5)

    surf = cloud.delaunay_2d()
    # surf.plot(show_edges=True)

    v = surf.points
    f = surf.faces
    f = rearange_mesh_faces(f)
    # format the faces in the right format


    v1, v2, k1, k2 = igl.principal_curvature(v, f)

    dk1_1 = calc_dk1_1(v, v1, k1)


if __name__ == "__main__":
    main()
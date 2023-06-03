import numpy
import pyvista
from pathlib import Path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import torch
from pytorch3d.ops import sample_farthest_points

def normalize_curve(curve: numpy.ndarray) -> numpy.ndarray:
    curve = curve - numpy.mean(curve)
    scale = numpy.max(
        [numpy.abs(numpy.min(curve[:, 0])),
        numpy.abs(numpy.max(curve[:, 0])),
        numpy.abs(numpy.min(curve[:, 1])),
        numpy.abs(numpy.max(curve[:, 1]))])
    curve = curve / scale
    return curve


def main():
    # file_path = Path('/home/roy/Documents/circles/curves.npy')
    # curves = numpy.load(file=str(file_path), allow_pickle=True)
    # curve = curves[5]
    # curve = normalize_curve(curve=curve)
    # curve_x = curve[:, 0]
    # curve_y = curve[:, 1]
    # curve_z = numpy.zeros_like(a=curve_x)
    #
    # grid_points_count = 90
    # grid_linspace = numpy.linspace(-1, 1, grid_points_count)
    # grid_x, grid_y = numpy.meshgrid(grid_linspace, grid_linspace)
    #
    # edge_source = numpy.c_[curve_x, curve_y, curve_z]
    # curve_points = numpy.c_[curve_x, curve_y]
    # polygon = Polygon(curve_points)
    # interior = []
    #
    # for i in range(grid_points_count):
    #     for j in range(grid_points_count):
    #         grid_point = Point(grid_linspace[i], grid_linspace[j])
    #         if polygon.contains(grid_point):
    #             interior.append(numpy.array([grid_point.x, grid_point.y, 0]))

    grid_points_count = 200
    grid_linspace = numpy.linspace(-1, 1, grid_points_count)
    grid_x, grid_y = numpy.meshgrid(grid_linspace, grid_linspace)
    points = numpy.stack([grid_x.ravel(), grid_y.ravel(), numpy.zeros(shape=grid_points_count*grid_points_count)], axis=1)

    points_tensor = torch.tensor(points)
    points_tensor = points_tensor.unsqueeze(dim=0)
    sampled_points = sample_farthest_points(points=points_tensor, K=5000, random_start_point=True)
    sampled_points = sampled_points[0].squeeze(dim=0).cpu().detach().numpy()
    sampled_points[:, 2] = 0.1*numpy.sin(2*numpy.pi*sampled_points[:, 0]*0.5) + numpy.cos(4*numpy.pi*sampled_points[:, 1]*0.1)

    cloud = pyvista.PolyData(sampled_points)
    cloud.plot(point_size=5)

    # mesh = pyvista.PolyData(edge_source)
    surf = cloud.delaunay_2d()
    surf.plot(show_edges=True)

    # curve = numpy.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [-0.5, 1, 0], [0, 0, 0], [0.5, 1, 0]])
    #
    # cloud = pyvista.PolyData(curve)
    # cloud.plot(point_size=5)
    #
    # # circ0 = pyvista.Polygon(center=(0, 0, 0), n_sides=7, radius=0.3)
    # # curve2 = numpy.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 1, 0]])
    # # points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    # curve2 = numpy.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0]])
    # faces = numpy.hstack([[3, 0, 2, 1]])
    # mesh = pyvista.PolyData(curve2, faces)
    # # poly = pyvista.lines_from_points(curve2)
    # # mesh = pyvista.PolyData(lines=poly)
    # surf = cloud.delaunay_2d(edge_source=mesh)
    # surf.plot(show_edges=True)


if __name__ == "__main__":
    main()
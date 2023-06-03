import os
import shutil
from typing import Tuple, Optional, Iterable
import sys

sys.path.insert(0, '..')
from collections import namedtuple
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from skimage.segmentation import active_contour
from tqdm import tqdm
from utils import print_cfg
from math import ceil
import hydra
from omegaconf import DictConfig
from dgl.geometry import farthest_point_sampler
import torch
from data_utils.general import process_edges

Verbose = False
Illustrate = False
print_orig = print
print = lambda *args, **kwargs: print_orig(*args, **kwargs) if Verbose else None

FileNames = namedtuple('FileNames',
                       ['prefix', 'uv', 'points', 'rest_pos', 'faces', 'uv_faces', 'edges', 'bdry_points', 'edge_t_ind',
                        'int_edge_ind'])

Mesh = namedtuple('Mesh', ['uv', 'points', 'rest_pos', 'faces', 'uv_faces', 'edges', 'bdry_points', 'edge_t_ind',
                           'int_edge_ind'])

patch_file_names = FileNames(prefix='v_', uv='uv.npy', points='points.npy', rest_pos='rest_pos.npy',
                             faces='faces.npy', uv_faces='uv_faces.npy', edges='edges.npy',
                             bdry_points='bdry_points.npy', edge_t_ind='edge_t_ind.npy',
                             int_edge_ind='int_edge_ind.npy')


def watershed(mask: np.ndarray, start_point: np.ndarray) -> np.ndarray:
    """
    marks every reachable point from the starting point without crossing over non-zero elements in the mask.
    :param mask: a binary mask where 0's are "explorable".
    :param start_point: the starting point for the water shed.
    :return: the same mask, with points that are reachable from the starting point marked with 2's.
    """
    p = start_point
    mask[p[0], p[1]] = 2
    directions = np.asarray([[0, 1], [0, -1], [1, 0], [-1, 0]])
    p_list = [p]
    curr_dir_idx_list = [0]
    while p_list:
        idx = curr_dir_idx_list[-1]
        if idx > 3:
            p_list.pop()
            curr_dir_idx_list.pop()
            continue
        p = p_list[-1]
        d = directions[idx]
        new_p = p + d
        curr_dir_idx_list[-1] += 1
        if np.any(new_p < 0) or np.any(new_p >= mask.shape[0]) or mask[new_p[0], new_p[1]] > 0:
            continue

        mask[new_p[0], new_p[1]] = 2
        # add the new point
        p_list.append(new_p)
        curr_dir_idx_list.append(0)
    return mask


def generate_mask(res: int) -> np.ndarray:
    """
    generates a boolean bask that indicates what pixels are inside the patch.
    :param res: the number of points in each axis
    :return: a numpy array of shape (resxres)
    """
    directions = np.asarray([[0, 1], [0, -1], [1, 0], [-1, 0]])
    while True:
        mask = np.zeros((res, res))
        for _ in range(np.sqrt(res).astype(int)):
            x = np.random.randint(res // 4, 3 * res // 4, 2)
            for _ in range(1, res):
                direction = directions[np.random.randint(0, 4)]
                x += direction
                x[x < 0] = 0
                x[x == res] = res - 1
                mask[x[0], x[1]] = 1
        # np.con
        mask = signal.convolve2d(mask, np.ones([res // 20] * 2), mode='same')
        mask[mask >= 0.2] = 1
        mask[mask < 1] = 0
        range_arr = np.arange(res)
        ones_arr = np.ones_like(range_arr)
        snake = [[ones_arr * 0, range_arr], [range_arr, ones_arr * range_arr[-1]],
                 [ones_arr * range_arr[-1], range_arr[::-1]], [range_arr[::-1], 0 * ones_arr]]
        bdry = np.concatenate([np.stack(x, axis=-1) for x in snake], axis=0)[:-1, :]
        mask[bdry[:, 0], bdry[:, 1]] = 0
        snake = active_contour(mask, bdry, boundary_condition='periodic', alpha=0.1)
        snake_mask = np.zeros_like(mask)
        funcs = [np.ceil, np.floor]
        snake = [np.stack([f1(snake[:, 0]), f2(snake[:, 1])], axis=1) for f1 in funcs for f2 in funcs]
        snake = np.concatenate(snake, axis=0).astype(int)
        snake[snake >= res] = res - 1
        snake[snake < 0] = 0
        snake_mask[snake[:, 0], snake[:, 1]] = 1
        snake_mask = signal.convolve2d(snake_mask, np.ones([res // 20] * 2), mode='same')
        snake_mask[snake_mask >= 0.2] = 1
        snake_mask[snake_mask < 1] = 0
        snake_mask[bdry[:, 0], bdry[:, 1]] = 0
        snake_mask = watershed(snake_mask, np.asarray([0, 0]))
        snake_mask[snake_mask > 0] = 1
        final_mask = 1 - snake_mask
        if np.all(final_mask == 0):
            print('Zero Mask Dumped.', file=sys.stderr)
        else:
            return 1 - snake_mask


def fps(points: np.ndarray, n: int) -> np.ndarray:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    points_tensor = torch.as_tensor(points[None, ...], device=device)
    ind = farthest_point_sampler(points_tensor, n.item()).cpu().numpy().squeeze()
    return points[ind]


def sample_points(mask: np.ndarray, num_points: int, perform_fps: bool = False) -> np.ndarray:
    """
    samples points inside a mask.
    :param mask: a (hxw) array.
    :param num_points: the number of points to sample inside the mask.
    :param perform_fps: [bool] if True(default) regularizes the points by sampling them using farthest points sampling.
    :return: ndarray of shape (num_points, 2) containing the 2D coordinates of the chosen points.
    """
    fps_coef = 10
    #
    res = 1 / mask.shape[0]
    assert mask.shape[0] == mask.shape[1]
    eff_n = fps_coef * num_points if perform_fps else num_points
    args = np.stack(np.nonzero(mask), axis=1)
    chosen = np.random.choice(args.shape[0], eff_n, replace=True)
    chosen = args[chosen, :]
    chosen = (chosen + np.random.rand(*chosen.shape))
    chosen = fps(chosen, num_points)
    # normalizing
    chosen = chosen * res - 0.5
    return chosen


def clean_triangulation(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    removes triangles that have outlier edge lengths.
    :param points: an array of (N, 3) shape.
    :param tri: an integer array of (T, 3) shape.
    :return: (points, clean_mesh) where points is the same as the input, but clean_mesh is the cleaned mesh.
    """
    # TODO: update descreption
    p = points[tri, :]
    e = []
    l = []
    for i in range(3):
        j = (i + 1) % 3
        e.append(p[:, i, :] - p[:, j, :])
        l.append(np.linalg.norm(e[-1], axis=-1))
    l = np.stack(l, axis=0)  # (3, mesh.shape[0])
    # e = np.stack(e, axis=0)  # (3 (num_edges), mesh.shape[0], 3 (dims))
    # edge length based
    center_stat, dist_stat = l.mean(), 3 * l.std()
    mask = np.sum(np.logical_and(l < center_stat + dist_stat, l > center_stat - dist_stat).astype(int), axis=0) == 3
    ind = np.nonzero(mask)[0]
    tri = tri[ind]
    # area based
    return tri


def triangulate(points: np.ndarray) -> np.ndarray:
    """
    triangulates 2d pointclouds
    :param points: a list of pointclouds
    :return: integer ndarray of shape (T, 3)
    """
    tri = clean_triangulation(points, Delaunay(points).simplices.astype(np.int64))
    if Illustrate:
        bdry, _, _, _, _ = process_edges(tri)
        plt.triplot(points[:, 0], points[:, 1], tri)
        plt.plot(points[:, 0], points[:, 1], 'o', c='b')
        plt.plot(points[bdry, 0], points[bdry, 1], 'o', c='r')
        plt.show()
    return tri


def random_mesh_augmentation(points: np.ndarray, num_harmonics: int = 20, freq_scale: float = 10.,
                             coef_scale: float = 300) -> np.ndarray:
    """
    augments meshes in 3D space.
    :param points: ndarray of shape (N,2) representing the points to be embedded.
    :param num_harmonics: the number of harmonic functions in the sum.
    :param freq_scale: the standard variation of the frequency magnitude.
    :param coef_scale: the standard variation of the summation coefficients.
    :return: ndarray of shape (N, 3) containing the embedded points (in the same order).
    """
    vecs = np.random.rand(2, num_harmonics)
    vecs *= np.random.randn(1, num_harmonics) * freq_scale / np.linalg.norm(vecs, axis=0, keepdims=True)
    coef = coef_scale * np.random.randn(1, num_harmonics)
    proj = points @ vecs
    z = (np.sin(proj) * coef).sum(axis=-1)
    # uniform 3d rotation
    new_points = Rotation.random().apply(np.concatenate([points, z[:, None]], axis=-1))
    # if generate_3:
    #     vecs = np.random.rand(2, num_harmonics, 3)
    #     vecs *= np.random.randn(1, num_harmonics, 3) * freq_scale / np.linalg.norm(vecs, axis=0, keepdims=True)
    #     coef = coef_scale * np.random.randn(1, num_harmonics, 3)
    #     proj = (points @ vecs.reshape(vecs.shape[0], -1)).reshape(points.shape[0], -1, 3)
    #     print(proj.shape) # (N, num_harmonics, 3)
    #     z = (np.sin(proj) * coef).sum(axis=1)
    #     new_points.append(z)

    return new_points


def check_mesh(tri: np.ndarray, num_points: np.ndarray, bdry_points: Optional[np.ndarray] = None,
               bdry_edges: Optional[np.ndarray] = None) -> bool:
    """
    validates that the mesh has a continuous and cyclic boundary.
    :param tri: ndarray of shape (T,3) representing the faces.
    :param num_points: number of points
    :param bdry_points: ndarray(_, )the boundary points indices.
    :param bdry_edges: ndarray (_,2), the boundary edges represented by their vertex indices.
    :return: True if the triangulation is valid, False otherwise.
    """
    if bdry_points is None or bdry_edges is None:
        bdry_points, bdry_edges, _, _, _ = process_edges(tri)
    bdry_edges[1, 0], bdry_edges[1, 1] = bdry_edges[1, 1], bdry_edges[1, 0]
    edge_map = -np.ones(num_points, dtype=int)
    reverse_edge_map = -np.ones(num_points, dtype=int)
    edge_map[bdry_edges[:, 0]] = bdry_edges[:, 1]
    reverse_edge_map[bdry_edges[:, 1]] = bdry_edges[:, 0]
    node_list = [-1]  # an efficient way to disallow -1s
    for i in range(bdry_edges.shape[0]):
        curr = node_list[-1] if node_list[-1] >= 0 else bdry_points[0]
        v, u = edge_map[curr], reverse_edge_map[curr]
        if v not in node_list:
            node_list.append(v)
        elif u not in node_list:
            node_list.append(u)
        else:
            found = False
            for x in bdry_edges[np.any(bdry_edges == curr, axis=-1)]:
                x = x[0] if x[1] == curr else x[1]
                if x not in node_list:
                    node_list.append(x)
                    found = True
                    break
            if not found:
                warn('Incomplete Boundary')
                return False
    node_list[0] = bdry_points[0]  # drop the -1
    cyclic_cond = node_list[-1] == node_list[0]
    genus_0_cond = np.unique(node_list[:-1]).shape[0] == (len(node_list) - 1)
    res = cyclic_cond and genus_0_cond
    if not res:
        if cyclic_cond:
            warn('Mesh Failed Test: Mesh is not genus 0.')
        elif genus_0_cond:
            warn('Mesh Failed Test: Boundary is not cyclic.')
        else:
            warn('Mesh Failed Test: Boundary is not cyclic, and mesh is not genus 0.')
    return res


def normalize_arr(arr, factor):
    center = arr.mean(axis=0, keepdims=True)
    return (arr - center) * factor + center


def average_edge_length(root: str):
    tot = []
    for dirname in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dirname)):
            continue
        f_dirname = os.path.join(root, dirname)
        edges = np.load(os.path.join(f_dirname, 'edges.npy'))
        points = np.load(os.path.join(f_dirname, 'points.npy'))
        tot.append(np.mean(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=-1)))
    print(np.mean(np.asarray(tot)))


def process_mesh(uv: np.ndarray, aug_p: np.ndarray, faces: np.ndarray, validate: bool = False,
                 uv_edge_scale: float = 0.1, aug_edge_scale: float = 0.1, rest_pos: Optional[np.ndarray] = None,
                 rest_pos_scale: Optional[float] = 0.1) -> Optional[Mesh]:
    bdry_points, bdry_edges, edge_t_ind, edges, int_edge_ind = process_edges(faces)
    # validating triangulation
    if validate and not check_mesh(faces, uv.shape[0], bdry_points, bdry_edges):
        return None
    # normalizing uv edge length
    mean_uv_edge_l = np.mean(np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=-1))
    uv = normalize_arr(uv, uv_edge_scale / mean_uv_edge_l)

    # normalizing mesh edge length
    mean_aug_edge_l = np.mean(np.linalg.norm(aug_p[edges[:, 0]] - aug_p[edges[:, 1]], axis=-1))
    aug_p = normalize_arr(aug_p, aug_edge_scale / mean_aug_edge_l)

    # normalizing rest pose
    if rest_pos is None:
        rest_pos = uv * rest_pos_scale / uv_edge_scale
    else:
        mean_aug_edge_l = np.mean(np.linalg.norm(rest_pos[edges[:, 0]] - rest_pos[edges[:, 1]], axis=-1))
        rest_pos = normalize_arr(rest_pos, rest_pos_scale / mean_aug_edge_l)

    # returning result
    mesh = Mesh(uv=uv, points=aug_p, rest_pos=rest_pos, faces=faces, edges=edges,
                bdry_points=bdry_points, edge_t_ind=edge_t_ind, int_edge_ind=int_edge_ind)
    return mesh


def add_discontinuity_to_uv(uv: np.ndarray, p: float = 5e-2):
    assert p >= 0 and p <= 1
    if np.random.rand(1) > p:
        return uv
    u = uv[:, 0]
    bias = np.random.rand(1) * (u.max() - u.min()) + u.min()
    u = u - bias
    mask = u > 0
    new_u = u[mask]
    if np.random.rand(1) > 0.5:
        max_u = u.max()
        new_u = max_u - new_u
    else:
        min_u = u.min()
        new_u = new_u + min_u
    u[mask] = new_u

    # random rotation
    theta = np.random.rand(1) * 2 * np.pi
    v = uv[:, 1]
    u, v = u * np.cos(theta) + v * np.sin(theta), v * np.cos(theta) - u * np.sin(theta)
    uv[:, 0], uv[:, 1] = u, v
    return uv


def generate_dataset(n: int, res: int, num_points, num_harmonics: int = 30, freq_scale: float = 10.,
                     coef_scale: float = 0.03, perform_fps: bool = True, uv_as_rest: bool = True,
                     discont_uv: bool = False) -> Iterable[Mesh]:
    assert n > 0
    pbar = tqdm(total=n)
    i = 0
    if not isinstance(num_points, Iterable):
        num_points = np.ones(n, dtype=int) * num_points
    while i < n:
        # generating mask
        mask = generate_mask(res)
        # sampling points
        uv = sample_points(mask, num_points[i], perform_fps=perform_fps)
        # triangulating
        faces = triangulate(uv)
        # 3d embedding
        aug_p = random_mesh_augmentation(uv, num_harmonics, freq_scale, coef_scale)
        rest_pos = None if uv_as_rest is None else random_mesh_augmentation(uv, num_harmonics, freq_scale, coef_scale)
        mesh = process_mesh(uv=uv, aug_p=aug_p, faces=faces, validate=True, rest_pos=rest_pos)
        if mesh is None:
            continue
        else:
            i += 1
        # add a discontinuity to uv (if needed)
        if discont_uv:
            uv = add_discontinuity_to_uv(uv, 1)  # p=5e-2)
            mesh = mesh._replace(uv=uv)
        # returning
        pbar.update(1)
        yield mesh


def generate_and_save_data_set(root: str, n: int, res, num_points, num_harmonics: int = 30,
                               freq_scale: float = 10., coef_scale: float = 0.03,
                               perform_fps: bool = True, append: bool = False, prefix: str = patch_file_names.prefix,
                               num_threads: int = 1, thread_num: int = 0, uv_as_rest: bool = True,
                               discont_uv: bool = False):
    assert thread_num < num_threads
    start = 0
    if num_threads == 1:
        if not os.path.exists(root):
            os.mkdir(root)
        else:
            assert os.path.isdir(root)
            if append:
                start = max([0] + [int(x[len(prefix):]) for x in os.listdir(root) if x.startswith(prefix)]) + 1
            else:
                raise RuntimeError(
                    'root directory already exists, to enable adding files to existing directories set "append=True".')
    else:
        if thread_num == 0:
            assert not os.path.exists(root)
            os.mkdir(root)
        else:
            assert os.path.exists(root)
        l = ceil(n / num_threads)
        assert l * num_threads >= n
        start = l * thread_num
        n = l if start + l <= n else n - start
        np.random.seed(num_threads ** 3 + 11)  # find a better way
        num_points = num_points[start: start + n]
        assert num_points.shape[0] == n
        ##
    pbar = tqdm(total=n)
    for i, mesh in enumerate(
            generate_dataset(n, res, num_points, num_harmonics=num_harmonics, freq_scale=freq_scale,
                             coef_scale=coef_scale, perform_fps=perform_fps, uv_as_rest=uv_as_rest,
                             discont_uv=discont_uv)):
        int_dir = os.path.join(root, prefix + str(start + i))
        save_patch_mesh(mesh, int_dir)
        pbar.update(1)


def _int_save(root, file_name, data):
    if data is None:
        warn(f'{file_name}: is None!, not saved.')
    np.save(os.path.join(root, file_name), data)


def save_patch_mesh(mesh: Mesh, root: str):
    # note: this does not load rest_pos
    assert not os.path.exists(root)
    os.mkdir(root)
    _int_save(root, patch_file_names.uv, mesh.uv)
    _int_save(root, patch_file_names.points, mesh.points)
    _int_save(root, patch_file_names.rest_pos, mesh.rest_pos)
    _int_save(root, patch_file_names.faces, mesh.faces)
    _int_save(root, patch_file_names.uv_faces, mesh.uv_faces)
    _int_save(root, patch_file_names.edges, mesh.edges)
    _int_save(root, patch_file_names.bdry_points, mesh.bdry_points)
    _int_save(root, patch_file_names.edge_t_ind, mesh.edge_t_ind)
    _int_save(root, patch_file_names.int_edge_ind, mesh.int_edge_ind)


def load_patch_mesh(path: str) -> Mesh:
    uv = np.load(os.path.join(path, patch_file_names.uv))
    points = np.load(os.path.join(path, patch_file_names.points))
    rest_pos = np.load(os.path.join(path, patch_file_names.rest_pos))
    faces = np.load(os.path.join(path, patch_file_names.faces))
    path_uv_faces = os.path.join(path, patch_file_names.uv_faces)
    if os.path.exists(path_uv_faces):
        uv_faces = np.load(path_uv_faces)
    else:
        uv_faces = faces
        warn('uv_faces path does not exist!! (using faces instead.')
        assert uv.shape[0] == points.shape[0]
    edges = np.load(os.path.join(path, patch_file_names.edges))
    bdry_points = np.load(os.path.join(path, patch_file_names.bdry_points))
    edge_t_ind = np.load(os.path.join(path, patch_file_names.edge_t_ind))
    int_edge_ind = np.load(os.path.join(path, patch_file_names.int_edge_ind))
    mesh = Mesh(uv=uv,
                points=points,
                rest_pos=rest_pos,
                faces=faces,
                uv_faces=uv_faces,
                edges=edges,
                bdry_points=bdry_points,
                edge_t_ind=edge_t_ind,
                int_edge_ind=int_edge_ind)

    return mesh


def get_jpgs(root: str, render_path: str):
    import pyvista as pv
    uv_path, aug_path = 'uv', 'mesh'
    if not os.path.exists(render_path):
        os.mkdir(render_path)
        os.mkdir(os.path.join(render_path, uv_path))
        os.mkdir(os.path.join(render_path, aug_path))
    else:
        raise RuntimeError('render path already exists.')
    for dirname in tqdm(os.listdir(root)):
        if not os.path.exists(os.path.join(root, dirname, patch_file_names.points)):
            continue
        mesh = load_patch_mesh(os.path.join(root, dirname))
        # 3d
        p = mesh.points
        tri = mesh.faces
        temp_tri = np.concatenate([np.ones(tri.shape[0], dtype=int)[:, None] * 3, tri], axis=-1).ravel()
        temp_mesh = pv.PolyData(p, faces=temp_tri)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(temp_mesh, show_edges=True)
        plotter.show(screenshot=os.path.join(render_path, aug_path, dirname + '.png'))

        # uv
        uv = mesh.uv
        bdry = mesh.bdry_points
        plt.triplot(uv[:, 0], uv[:, 1], tri)
        plt.plot(uv[:, 0], uv[:, 1], 'o', c='b')
        plt.plot(uv[bdry, 0], uv[bdry, 1], 'o', c='r')
        plt.savefig(os.path.join(render_path, uv_path, dirname + '.png'))
        plt.close()


def add_rest_pos(root: str):
    assert os.path.exists(root)
    c = 0
    for dir_name in tqdm(os.listdir(root)):
        dir_path = os.path.join(root, dir_name)
        src_path = os.path.join(dir_path, patch_file_names.uv)
        dest_path = os.path.join(dir_path, patch_file_names.rest_pos)
        if os.path.exists(dest_path):
            continue
        # shutil.copyfile(dest_path, src_path)
        src_arr = np.load(src_path)
        np.save(dest_path, src_arr)
        c += 1
    print_orig(f'{c} files were copied.')


@hydra.main(version_base=None, config_path='hp', config_name='base_hp')
def main(cfg: DictConfig):
    print_cfg(cfg)
    np.random.seed(0)
    num_points = np.random.randint(cfg.min_num_points, cfg.max_num_points, cfg.n)

    generate_and_save_data_set(cfg.root, cfg.n, cfg.res, num_points, num_harmonics=cfg.aug.num_harmonics,
                               freq_scale=cfg.aug.freq_scale, coef_scale=cfg.aug.coef_scale, append=cfg.append,
                               perform_fps=cfg.perform_fps, num_threads=cfg.num_threads, thread_num=cfg.thread_num,
                               uv_as_rest=cfg.uv_as_rest, discont_uv=cfg.discont_uv)
    # get_jpgs(cfg.root, os.path.join(cfg.root, 'render'))


if __name__ == '__main__':
    # add_rest_pos('../Data/Artificial_Patches_10k')
    # exit()
    Verbose = True
    Illustrate = False
    main()
    # average_edge_length('../Data/Artificial_Patches_shit')
    # root = '../Data/Artificial_Patches/'
    # for dirname in os.listdir(root):
    #     path = os.path.join(root, dirname)
    #     mesh = load_patch_mesh(path)
    #     bdry_points, bdry_edges, edge_t_ind, edges, int_edge_ind = process_edges(mesh.faces)
    #
    #     assert np.allclose(bdry_points, mesh.bdry_points)
    #     assert np.allclose(np.sort(edge_t_ind, axis=-1), np.sort(mesh.edge_t_ind, axis=-1))
    #     assert np.allclose(edges, mesh.edges)
    #     assert np.allclose(int_edge_ind, mesh.int_edge_ind)

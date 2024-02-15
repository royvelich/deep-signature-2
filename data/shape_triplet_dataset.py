import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, radius, radius_graph, fps

from data.non_uniform_sampling import non_uniform_2d_sampling
from utils import random_rotation, random_rotation_numpy, normalize_point_cloud_numpy, \
    normalize_points_translation_and_rotation

# import fpsample
# import os
# import torch_points3d as tp3d
# from torch_points3d.core.data_transform import FPS
from scipy.spatial import cKDTree
# from visualize_pointclouds import visualize_pointclouds2


class ShapeTripletDataset(Dataset):
    def __init__(self, shapes_dataset, transform=None, radius = 0.01, number_of_points_to_sample = 32, max_neighbourhood_size = 256):
        self.shapes_dataset = shapes_dataset
        self.transform = transform
        self.radius = radius # radius for knn graph
        self.number_of_points_to_sample = number_of_points_to_sample # number of points to sample from the point cloud
        self.max_neighbourhood_size = max_neighbourhood_size # max number of points in the neighbourhood


    def __len__(self):
        # assuming all the 3 datasets have the same length
        return len(self.shapes_dataset)




    def __getitem__(self, idx):
       # item will be M sampled point on the shapes surface and their knn graph with radius r
       # anc is the original point cloud, pos is the same point cloud with random rotation, neg is a different sampled points on different point cloud with random rotation
        neg_idx = (idx + torch.randint(1,len(self.shapes_dataset)-1,(1,))).item() % len(self.shapes_dataset)

        shape_anc_pos, shape_neg = copy.deepcopy(self.shapes_dataset[idx]),  copy.deepcopy(self.shapes_dataset[neg_idx])

        # 0.01 of max distance between points
        radius_anc_pos = 0.06 * np.max(np.linalg.norm(shape_anc_pos, axis=1))
        radius_neg = 0.06 * np.max(np.linalg.norm(shape_neg, axis=1))

        # sampling number_of_points_to_sample points using fps detach tensors to numpy
        anc_pos_sampled_points, anc_pos_sampled_indices = self.default_fps_sampling(shape_anc_pos)
        neg_sampled_points, neg_sampled_indices = self.default_fps_sampling(shape_neg)

        anc_pos_sampled_points = torch.tensor(anc_pos_sampled_points)
        neg_sampled_points = torch.tensor(neg_sampled_points)

        # creating the knn neighborhoods for each sampled point w.r.t the original point cloud and radius
        anc_pos_radius_neighborhoods = radius(shape_anc_pos, anc_pos_sampled_points, r=radius_anc_pos, max_num_neighbors=self.max_neighbourhood_size)
        neg_knn_radius_neighborhoods = radius(shape_neg, neg_sampled_points, r=radius_neg, max_num_neighbors=self.max_neighbourhood_size)
        # map the first row of the neighborhoods to the sampled points from 0-number_of_points_to_sample to 0-N image
        anc_pos_radius_neighborhoods[0,:] = anc_pos_sampled_indices[anc_pos_radius_neighborhoods[0,:]]
        neg_knn_radius_neighborhoods[0,:] = neg_sampled_indices[neg_knn_radius_neighborhoods[0,:]]

        anc_v = shape_anc_pos
        pos_v = self.transform_random_rotations(copy.deepcopy(anc_v))
        neg_v = shape_neg

        # remove from anc_v and neg_v the points that are not in the neighborhoods
        # anc_v = anc_v[anc_pos_radius_neighborhoods[1,:]]

        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # pos_v = pos_v[anc_pos_radius_neighborhoods[1,:].to(device=pos_v.device)]
        # pos_v = pos_v.to(device=anc_v.device)
        # neg_v = neg_v[neg_knn_radius_neighborhoods[1,:]]
        # anc_v = torch.tensor(anc_v, dtype=torch.float32)
        # pos_v = torch.tensor(pos_v, dtype=torch.float32)
        # neg_v = torch.tensor(neg_v, dtype=torch.float32)

        item_anc = Data(x=anc_v, pos=anc_v,
             edge_index=anc_pos_radius_neighborhoods)
        item_pos = Data(x=pos_v, pos=pos_v,
                edge_index=anc_pos_radius_neighborhoods)
        item_neg = Data(x=neg_v, pos=neg_v,
                edge_index=neg_knn_radius_neighborhoods)

        item = item_anc, item_pos, item_neg

        # patch_anc = self.transform_random_rotations(copy.deepcopy(patch_anc_pos))
        # patch_pos = self.transform_random_rotations(patch_anc_pos)
        # patch_neg = self.transform_random_rotations(patch_neg)
        #
        # patch_anc = self.default_non_uniform_sampling(patch_anc)
        # patch_pos = self.default_non_uniform_sampling(patch_pos)
        # patch_neg = self.default_non_uniform_sampling(patch_neg)

        # item = patch_anc,patch_pos,patch_neg
        # visualize_pointclouds2(patch_anc.v, patch_pos.v, patch_neg.v)

        return item

    def default_transform(self, patch):
        # non-uniform sampling and translation to origin(suppose to be point N/2,N/2, make sure she is sampled as well) with random rotation
        N = patch.v.shape[0]
        grid_size = int(np.sqrt(N))

        # add 0,0 point to sampled indices - check if it working properly
        mid_point_indice = N // 2 - grid_size//2 - 1

        center_point = patch.v[mid_point_indice]

        # origin = patch.v
        # normalize after sampling
        patch.v = normalize_points_translation_and_rotation(patch.v, center_point=center_point)
        # origin_after_sample_and_normazlize = patch.v
        # visualize_pointclouds(patch.v, origin)
        return patch

    def transform_random_rotations(self, patch):
        N = patch.shape[0]
        grid_size = int(np.sqrt(N))
        # ratio = np.random.uniform(0.01, 0.05)
        mid_point_indice = N // 2 - grid_size // 2 - 1

        # origin= patch.v
        # translate to origin
        # patch = patch - patch[mid_point_indice]
        # origin_after_Trans = patch.v

        # rotate
        patch = random_rotation(patch)
        # origin_after_rot = patch.v

        # visualize_pointclouds(origin, origin_after_Trans, origin_after_rot, origin_after_sample)
        return patch

    def default_fps_sampling(self, v):
            # v = torch.tensor(data=shape.v, dtype=torch.float32)
            # indices = fpsample.fps_sampling(v, self.number_of_points_to_sample)
            # fps sampling via pytorch 3d
            ratio = self.number_of_points_to_sample / v.shape[0]
            indices = fps(v, ratio=ratio)
            indices = indices[:self.number_of_points_to_sample]
            v = v[indices]
            return v, indices


    def default_non_uniform_sampling(self, patch):
        N = patch.v.shape[0]
        grid_size = int(np.sqrt(N))
        ratio = np.random.uniform(0.01, 0.05)
        sampled_indices = non_uniform_2d_sampling(grid_size, ratio=ratio)
        mid_point_indice = N // 2 - grid_size//2 - 1
        if mid_point_indice not in sampled_indices:
            rand_indice = torch.randint(0, len(sampled_indices), (1,))
            sampled_indices[rand_indice] = mid_point_indice

        patch.v = patch.v[sampled_indices]
        return patch

    # Define a custom collate function to handle variable-sized tensors in each triplet
    def padding_collate_fn(self, batch):
        # Find the maximum number of vertices in each triplet
        max_vertices = max(max(patch.size(0) for patch in triplet) for triplet in batch)

        # Pad each tensor to the maximum number of vertices within its triplet
        padded_batch = []
        for triplet in batch:
            padded_triplet = [torch.nn.functional.pad(patch, (0, 0, 0, max_vertices - patch.size(0))) for patch in triplet]
            padded_batch.append(torch.stack(padded_triplet))

        return torch.stack(padded_batch)


    def pack_collate_fn(self, batch):
        patches = []
        for i in range(len(batch)):
            patches.append(pack_sequence(batch[i], enforce_sorted=False))

        return patches


    # Function to compute edges from faces
    def compute_edges_from_faces(self, faces):
        edges = []
        for face in faces:
            edges.extend([(face[i], face[(i + 1) % 3]) for i in range(3)])
        edges = [tuple(sorted(edge)) for edge in edges]  # Remove duplicates
        edges = list(set(edges))  # Ensure uniqueness of edges
        # Convert the list of edges to a 2D tensor with two rows (source and target nodes)
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index


    def batch_collate_fn(self, batch):
        data = []
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                data.append(batch[i][j])

        return Batch.from_data_list(data)

import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

from data.non_uniform_sampling import non_uniform_2d_sampling
from utils import random_rotation, random_rotation_numpy, normalize_point_cloud_numpy, \
    normalize_points_translation_and_rotation

# from visualize_pointclouds import visualize_pointclouds
from visualize_pointclouds import visualize_pointclouds


class DynamicTripletDataset(Dataset):
    def __init__(self, data_spherical_patches, data_hyperbolic_patches, data_parabolic_patches, transform=None, k = 6):
        self.data_spherical_patches = data_spherical_patches
        self.data_hyperbolic_patches = data_hyperbolic_patches
        self.data_parabolic_patches = data_parabolic_patches
        self.transform = transform
        self.k = k # number of neighbors for knn graph

    def __len__(self):
        # assuming all the 3 datasets have the same length
        return len(self.data_spherical_patches)




    def __getitem__(self, idx):
        # case 1: anc,pos spherical, neg hyperbolic
        # case 2: anc,pos hyperbolic, neg spherical
        # case 3: anc,pos parabolic, neg spherical
        # case 4: anc,pos spherical, neg parabolic
        # case 5: anc,pos hyperbolic, neg parabolic
        # case 6: anc,pos parabolic, neg hyperbolic
        rand_case = torch.randint(1,7,(1,))
        if rand_case == 1:
            patch_anc_pos, patch_neg = copy.deepcopy(self.data_spherical_patches[idx]),  copy.deepcopy(self.data_hyperbolic_patches[idx])
        elif rand_case == 2:
            patch_anc_pos, patch_neg = copy.deepcopy(self.data_hyperbolic_patches[idx]),  copy.deepcopy(self.data_spherical_patches[idx])
        elif rand_case == 3:
            patch_anc_pos, patch_neg = copy.deepcopy(self.data_parabolic_patches[idx]), copy.deepcopy(self.data_spherical_patches[idx])
        elif rand_case == 4:
            patch_anc_pos, patch_neg = copy.deepcopy(self.data_spherical_patches[idx]), copy.deepcopy(self.data_parabolic_patches[idx])
        elif rand_case == 5:
            patch_anc_pos, patch_neg = copy.deepcopy(self.data_hyperbolic_patches[idx]), copy.deepcopy(self.data_parabolic_patches[idx])
        elif rand_case == 6:
            patch_anc_pos, patch_neg = copy.deepcopy(self.data_parabolic_patches[idx]),  copy.deepcopy(self.data_hyperbolic_patches[idx])
        else:
            raise ValueError('rand_case is not in range 1-6')

        # if self.transform is not None:
        #     patch_anc_pos = self.transform(patch_anc_pos)
        #     patch2 = self.transform(patch2)
        #     patch_neg = self.transform(patch_neg)
        # else:
        # visualize_pointclouds(patch_anc_pos.v, patch_neg.v)

        patch_anc_pos = self.default_transform(patch_anc_pos)
        patch_neg = self.default_transform(patch_neg)

        patch_anc = self.default_non_uniform_sampling(copy.deepcopy(patch_anc_pos))
        patch_pos = self.default_non_uniform_sampling(patch_anc_pos)
        patch_neg = self.default_non_uniform_sampling(patch_neg)

        item = patch_anc,patch_pos,patch_neg
        # visualize_pointclouds(patch_anc.v, patch_pos.v, patch_neg.v)

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
        N = patch.v.shape[0]
        grid_size = int(np.sqrt(N))
        ratio = np.random.uniform(0.01, 0.05)
        mid_point_indice = N // 2 - grid_size // 2 - 1

        # origin= patch.v
        # translate to origin
        patch.v = patch.v - patch.v[mid_point_indice]
        # origin_after_Trans = patch.v

        # rotate
        patch.v = random_rotation_numpy(patch.v)
        # origin_after_rot = patch.v

        # visualize_pointclouds(origin, origin_after_Trans, origin_after_rot, origin_after_sample)
        return patch

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
        # use Data and Batch from torch_geometric
        data = []
        # @TODO: check face_to_edges in torch geometric
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                # edges = torch.tensor([[0], [0]])
                # data.append(Data(x=batch[i][j].v_second_moments.to(torch.float32), pos=torch.tensor(batch[i][j].v, dtype=torch.float32), edge_index=edges))
                # add face to enable supervised learning, can remove if using just unsup learning to decrease overhead
                data.append(Data(x=torch.tensor(batch[i][j].v, dtype=torch.float32), pos=torch.tensor(batch[i][j].v, dtype=torch.float32), edge_index= knn_graph(torch.tensor(batch[i][j].v), k=self.k, batch=None, loop=False)))
        return Batch.from_data_list(data)

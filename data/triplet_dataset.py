import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


# Define your custom dataset class


class CustomTripletDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     patch1, patch2, patch3 = self.data[idx]
    #     item = patch1.v_second_moments,patch2.v_second_moments,patch3.v_second_moments
    #     return item


    def __getitem__(self, idx):
        patch1, patch2, patch3 = self.data[idx]
        if self.transform is not None:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)
            patch3 = self.transform(patch3)
        item = patch1,patch2,patch3
        return item


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
                edges = self.compute_edges_from_faces(batch[i][j].f)
                # data.append(Data(x=batch[i][j].v_second_moments.to(torch.float32), pos=torch.tensor(batch[i][j].v, dtype=torch.float32), edge_index=edges))
                # add face to enable supervised learning, can remove if using just unsup learning to decrease overhead
                data.append(Data(x=batch[i][j].v_second_moments.to(torch.float32), pos=torch.tensor(batch[i][j].v, dtype=torch.float32), edge_index=edges, face=batch[i][j].f))
        return Batch.from_data_list(data)

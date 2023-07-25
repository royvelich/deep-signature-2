import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define your custom dataset class
class CustomTripletDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        # self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch1, patch2, patch3 = self.data[idx]
        item = patch1.v_second_moments,patch2.v_second_moments,patch3.v_second_moments
        return item

    # Define a custom collate function to handle variable-sized tensors in each triplet
    def custom_collate_fn(self, batch):
        # Find the maximum number of vertices in each triplet
        max_vertices = max(max(patch.size(0) for patch in triplet) for triplet in batch)

        # Pad each tensor to the maximum number of vertices within its triplet
        padded_batch = []
        for triplet in batch:
            padded_triplet = [torch.nn.functional.pad(patch, (0, 0, 0, max_vertices - patch.size(0))) for patch in triplet]
            padded_batch.append(torch.stack(padded_triplet))

        return torch.stack(padded_batch)

# Assuming you have your own data in the following format:
# Replace 'your_data' and 'your_labels' with your actual data and labels

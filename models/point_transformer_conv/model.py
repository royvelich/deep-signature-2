import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch_geometric.nn import PointTransformerConv, radius_graph, global_mean_pool
import pytorch_lightning as pl

from loss import loss_contrastive_plus_codazzi_and_pearson_correlation


# Taken from https://github.com/vsitzmann/siren
class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(input)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()

        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.input_bn(self.input_layer(x)))

        for i in range(self.num_layers - 2):
            x = self.activation(self.hidden_bns[i](self.hidden_layers[i](x)))

        x = self.output_layer(x)
        return x

class PointTransformerConvNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels=8, hidden_channels=64, num_layers=3):
        super(PointTransformerConvNet, self).__init__()

        self.save_hyperparameters()  # Saves all hyperparameters for logging
        self.activation = Sine()

        self.encoder = MLP(input_dim=in_channels, hidden_dim=hidden_channels, output_dim=hidden_channels, num_layers=2, activation=self.activation)
        self.conv_layers = nn.ModuleList([
            PointTransformerConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])

        self.hidden_bns = nn.ModuleList()
        for _ in range(num_layers):  # Last layer doesn't need BN and activation
            self.hidden_bns.append(nn.BatchNorm1d(hidden_channels))

        self.pooling = global_mean_pool  # You can use other pooling functions if needed

        self.decoder = MLP(input_dim=hidden_channels, hidden_dim=hidden_channels, output_dim=out_channels, num_layers=2, activation=self.activation)

        self.loss_func = loss_contrastive_plus_codazzi_and_pearson_correlation

    def forward(self, data):
        x = data.x
        # edge_index = radius_graph(x, r=0.5, batch=None, loop=True, max_num_neighbors=32)

        # Apply initial embedding
        x = self.encoder(x)

        # Apply Point Transformer Convolutional layers
        for i in range(len(self.conv_layers)):
            x = self.activation(self.hidden_bns[i](self.conv_layers[i](x=x,pos=data.pos, edge_index=data.edge_index)))

        # Apply pooling to aggregate information from vertices
        x = self.pooling(x, batch=data.batch)

        # Apply final linear layer
        x = self.decoder(x)

        return x




    def training_step(self, batch, batch_idx):

        output = self.forward(batch)
        device = output.device

        anchor_idx = torch.arange(0, output.size(0), 3, device=device)
        positive_idx = torch.arange(1, output.size(0), 3, device=device)
        negative_idx = torch.arange(2, output.size(0), 3, device=device)

        anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)
        negative_output = torch.index_select(output, 0, negative_idx)

        loss = self.loss_func(a=anchor_output.T, p=positive_output.T, n=negative_output.T)

        self.log('train_loss', loss, on_step=True, on_epoch=True)  # Logging the training loss
        return loss

    def validation_step(self, batch, batch_idx):

        output = self.forward(batch)
        device = output.device

        anchor_idx = torch.arange(0, output.size(0), 3, device=device)
        positive_idx = torch.arange(1, output.size(0), 3, device=device)
        negative_idx = torch.arange(2, output.size(0), 3, device=device)

        anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)
        negative_output = torch.index_select(output, 0, negative_idx)

        loss = self.loss_func(a=anchor_output.T, p=positive_output.T, n=negative_output.T)

        self.log('val_loss', loss, on_step=True, on_epoch=True)  # Logging the validation loss
        return loss


    def unpack_batch(self, batch):
        output_anc = []
        output_pos = []
        output_neg = []
        for i in range(len(batch)):
            # Unpack the packed_patches to get the padded representation
            padded_patches, original_lengths = pad_packed_sequence(batch[i], batch_first=True)

            # Now you can access the individual patches
            for j in range(len(original_lengths)):
                patch_j = padded_patches[j][:original_lengths[j]]
                if j==0:
                    output_anc.append(patch_j)
                elif j==1:
                    output_pos.append(patch_j)
                else:
                    output_neg.append(patch_j)

        # Convert the lists to tensors
        # output_anc = torch.cat(output_anc, dim=0)
        # output_pos = torch.cat(output_pos, dim=0)
        # output_neg = torch.cat(output_neg, dim=0)
        return output_anc, output_pos, output_neg

    def configure_optimizers(self, lr=0.0003,weight_decay=0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return [optimizer], [scheduler]


    def on_train_epoch_end(self):
        # Get the current learning rate from the optimizer
        current_lr = self.optimizers().param_groups[0]['lr']

        # Log the learning rate
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True)


    def normalize_features(self, features):
        mean_values = torch.mean(features, axis=1)
        std_values = torch.std(features, axis=1, unbiased=False)
        # Mask out elements with zero variance
        zero_variance_mask = (std_values == 0.0)
        std_values[zero_variance_mask] = 1.0  # Set to 1.0 to avoid division by zero

        # Step 2: Normalize the features to have zero mean and unit variance
        normalized_features = (features - torch.unsqueeze(mean_values,dim=1)) / torch.unsqueeze(std_values,dim=1)
        return normalized_features
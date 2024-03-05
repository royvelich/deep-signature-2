import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from models.point_transformer_conv.model import PointTransformerConvNetReconstruct
from visualize_pointclouds import visualize_pointclouds2


class MaxPoolLayer(nn.Module):
    def __init__(self, size, stride):
        super(MaxPoolLayer, self).__init__()
        self.maxpool = nn.MaxPool2d(size, stride, padding='same')

    def forward(self, x):
        return self.maxpool(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        return x

class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTransposeLayer, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        return self.conv_transpose(x)

class UNet(pl.LightningModule):
    def __init__(self, num_channels, unet_depth, learning_rate=1e-3):
        super(UNet, self).__init__()

        self.learning_rate = learning_rate

        # Encoder
        input_neurons = 64
        self.levels = nn.ModuleList()
        for level in range(0, unet_depth-1):
            neurons = input_neurons * (2**level)
            self.levels.append(nn.Sequential(
                ConvLayer(num_channels, neurons, (1, 1)),
                ConvLayer(neurons, neurons, (1, 1)),
                MaxPoolLayer((2, 2), (2, 2))
            ))

        # Decoder
        self.decoder = nn.ModuleList()
        for level in range(unet_depth-2, -1, -1):
            neurons = input_neurons * (2**level)
            self.decoder.append(nn.Sequential(
                ConvLayer(neurons * 2, neurons*2, (3, 3)),
                ConvLayer(neurons * 2, neurons, (3, 3)),
                ConvTransposeLayer(neurons, neurons, (2, 2), 2)
            ))

        # Final layers (convolution at input scale and fully connected)
        self.final_layers = nn.Sequential(
            ConvLayer(input_neurons, input_neurons, (3, 3)),
            ConvLayer(input_neurons, input_neurons, (3, 3)),
            ConvLayer(input_neurons, num_channels, (1, 1))  # Reconstruction output
        )

    def forward(self, data):
        x = data.x
        # Encoder
        levels = []
        for level in range(0, len(self.levels)):
            x = self.levels[level](x)
            levels.append(x)

        # Decoder
        for level in range(len(self.decoder)-1, -1, -1):
            x = self.decoder[level](x)
            x = torch.cat((levels[level], x), dim=1)

        # Final layers
        x = self.final_layers(x)

        return x

    def training_step(self, batch, batch_idx):
        x_input = batch.x
        # Unpack the batch
        batch.x = self.append_moments(batch.x)

        output = self.forward(batch)
        device = output.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 1]

        # anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)




        # Compute the loss
        anchor_input = torch.index_select(x_input, 0, anchor_idx)
        loss = self.loss_func(anchor_input, positive_output)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # visualize the point clouds
        # if batch_idx % 1 == 0:
        #     visualize_pointclouds2(anchor_input, positive_output)
        return loss

    def validation_step(self, batch, batch_idx):
        x_input = batch.x
        batch.x = self.append_moments(batch.x)

        output = self.forward(batch)
        device = output.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 1]

        anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)


        # Compute the loss
        anchor_input = torch.index_select(x_input, 0, anchor_idx)

        loss = self.loss_func(anchor_input, positive_output)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        # if batch_idx % 1 == 0:
        #     visualize_pointclouds2(anchor_input, positive_output)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def append_moments(self, x: torch.Tensor) -> torch.Tensor:
        second_order_moments = torch.einsum('bi,bj->bij', x, x)

        # Get the upper triangular indices
        rows, cols = torch.triu_indices(second_order_moments.shape[1], second_order_moments.shape[2])

        # Extract the upper triangular part for each MxM matrix
        upper_triangular_values = second_order_moments[:, rows, cols]

        appended_x = torch.cat((x, upper_triangular_values.view(x.shape[0], -1)), dim=1)
        return appended_x


# # Instantiate the model
# num_channels = 3  # Set to the appropriate number of channels
# unet_depth = 4    # Set to the desired UNet depth
# model = UNet(num_channels, unet_depth)
#
# # Training with PyTorch Lightning Trainer
# trainer = pl.Trainer(max_epochs=10)
# # trainer.fit(model, your_train_dataloader)

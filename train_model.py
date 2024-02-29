import os
import pickle

import torch.cuda
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.dynamic_triplet_dataset import DynamicTripletDataset
from data.human_segmentation_original_dataset import HumanSegOrigDataset
from data.shape_triplet_dataset import ShapeTripletDataset
from models.point_transformer_conv.model import PointTransformerConvNet, PointTransformerConvNetReconstruct, \
    PointCloudReconstruction
from utils import init_wandb, custom_euclidean_transform

from data.triplet_dataset import CustomTripletDataset
from models.pointnet.model import STNkd, PointNet_FC
from vars import LR, WEIGHT_DECAY
from visualize.visualize__results_on_mesh import VisualizerCallback

# Get the current working directory
current_directory = os.getcwd()
print("Current working directory: ", current_directory)


def main_loop():
    max_epochs = 2000
    batch_size = 16
    lr = LR
    weight_decay = WEIGHT_DECAY
    num_workers = 1
    # combine_reg_and_non_reg_patches = False
    server_dir = "/home/gal.yona/deep-signature-2/"
    if torch.cuda.is_available():
        # data_file_name = "triplets_size_300_N_5000_all_monge_patch_non_uniform_sampling_part0.pkl"
        # file_path = server_dir + "triplets_dataset/" + data_file_name
        # file_path = server_dir+"data/spherical_monge_patches_100_N_300.pkl"
        # file_path2 = server_dir+"data/hyperbolic_monge_patches_100_N_300.pkl"
        # file_path3 = server_dir+"data/parabolic_monge_patches_100_N_300.pkl"

        # file_path = "./data/spherical_monge_patches_100_N_10000.pkl"
        # file_path2 = "./data/hyperbolic_monge_patches_100_N_10000.pkl"
        # file_path3 = "./data/parabolic_monge_patches_100_N_10000.pkl"

        # path in hpc
        # dataset_path = "/workspace/Github/deep-signature-2/data/sig17_seg_benchmark/"

        # path in gipdeep
        dataset_path = "/home/gal.yona/diffusion-net2/diffusion-net/experiments/human_segmentation_original/data/sig17_seg_benchmark"
        train_dataset = HumanSegOrigDataset(dataset_path, train=True, use_cache=False, debug=True)
        # os.environ["WANDB_MODE"] = "offline"

        num_workers = 0
        # combine_reg_and_non_reg_patches = True
        train_ratio = 0.9
        batch_size = 1
        devices = -1 # takes the number of available GPUs


    else:
        # file_path = "generated_triplet_data/triplets_data_size_50_N_10_all_monge_patch_normalized_pos_and_rot.pkl"
        os.environ["WANDB_MODE"] = "offline"
        # file_path = "data/spherical_monge_patches_100_N_10.pkl"
        # file_path2 = "data/hyperbolic_monge_patches_100_N_10.pkl"
        # file_path3 = "data/parabolic_monge_patches_100_N_10.pkl"
        dataset_path = "C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\diffusion-net2\diffusion-net\experiments\human_segmentation_original\data\sig17_seg_benchmark"

        train_dataset = HumanSegOrigDataset(dataset_path, train=True, use_cache=False, debug=True)

        batch_size = 1
        train_ratio = 0.8
        devices = 1

        # logger = init_wandb(lr=lr,max_epochs=max_epochs, weight_decay=weight_decay, dataset_path=file_path+" and "+file_path2)




    # Load the triplets from the file
    # with open(file_path, 'rb') as f:
    #     f.seek(0)  # Move the file pointer to the beginning of the file
    #     data_spherical = pickle.load(f)
    # with open(file_path2, 'rb') as f:
    #     f.seek(0)
    #     data_hyperbolic = pickle.load(f)
    # with open(file_path3, 'rb') as f:
    #     f.seek(0)
    #     data_parabolic = pickle.load(f)






    # Create custom dataset
    # custom_dataset = CustomTripletDataset(data)
    # custom_dataset = DynamicTripletDataset(data_spherical, data_hyperbolic, data_parabolic)
    custom_dataset = ShapeTripletDataset(train_dataset)
    # # release memory
    # del data_spherical
    # del data_hyperbolic
    # del data_parabolic

    # Define the ratio for train and validation split (e.g., 80% for training, 20% for validation)

    # Calculate the number of samples for train and validation sets
    num_samples = len(custom_dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_val_samples = num_samples - num_train_samples

    # Use random_split to divide the dataset into train and validation sets
    train_dataset, val_dataset = random_split(custom_dataset, [num_train_samples, num_val_samples])

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_dataset.batch_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_dataset.batch_collate_fn)  # No need to shuffle validation data


    # model - initiallize to recieve input length as 9 for x,y,z,xy,yz,zx,xx,yy,zz
    # model = PointNet_FC(k=9)
    # model = STNkd(k=9)
    num_point_transformer_layers = 4
    num_encoder_decoder_layers = 8
    hidden_channels = 512
    in_channels = 9
    out_channels = 3
    # want to train model from trained weights
    checkpoint = torch.load( "C:/Users\galyo\Downloads\model_reconstruct_uniform_samples_random_rotations_chamfer_and_intra_loss_training-epoch=6851.ckpt", map_location=torch.device('cpu'))

    # Extract the state dictionary from the checkpoint
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('.select', ''): v for k, v in state_dict.items()}

    # model = UNet(num_channels=in_channels, unet_depth=num_encoder_decoder_layers)
    model = PointCloudReconstruction(num_blocks=num_point_transformer_layers, in_channels=in_channels, latent_dim=hidden_channels, num_points_to_reconstruct=64)
    model.load_state_dict(state_dict)
    # model.load_from_checkpoint("C:/Users\galyo\Downloads\model_point_transformer_4_layers_width_512_reconstruct_uniform_samples_random_rotations_just_cont_loss-epoch=998.ckpt", num_blocks=4,
    # in_channels=9,
    # latent_dim=512, num_points_to_reconstruct=1024, map_location=torch.device('cpu'))
    # model = PointCloudReconstruction.load_from_checkpoint("C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2\checkpoints\model_point_transformer_4_layers_width_512_reconstruct_uniform_samples_random_rotations_just_cont_loss-epoch=38.ckpt", map_location=torch.device('cpu'))
    # model = PointTransformerConvNetReconstruct(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_point_transformer_layers=num_point_transformer_layers, num_encoder_decoder_layers=num_encoder_decoder_layers)
    # model_path = "/home/gal.yona/deep-signature-2/trained_models/model_point_transformer_1_layers_width_128_train_non_uniform_samples_also_with_planar_patches-epoch=149.ckpt"
    # model_path = "./checkpoints/model_point_transformer_1_layers_width_512_non_uniform_samples_random_rotations-epoch=116.ckpt"
    # model_path = "C:/Users\galyo\Downloads\model_point_transformer_4_layers_width_512_reconstruct_uniform_samples_random_rotations_just_cont_loss-epoch=704.ckpt"
    # model = PointTransformerConvNetReconstruct.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
    # model.eval()

    # training
    logger = WandbLogger(project="train_on_patches",
               # entity="geometric-dl",
               config={
                   "learning_rate": lr,
                   "architecture": "Point Transformer Net Mean pool",
                   "dataset": "sphere, hyperbolic, parabolic 10000 patches each",
                   "epochs": max_epochs,
                   "weight_decay": weight_decay
               })
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # Directory to save the checkpoints
        filename='model_reconstruct_uniform_samples_random_rotations_chamfer_and_intra_loss_training-{epoch:02d}',
        save_top_k=1,  # Save all checkpoints
        save_on_train_epoch_end=True,
        every_n_epochs=3
    )
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=7,
    #     verbose=False,
    #     mode='min'
    # )
    # visualizer_callback = VisualizerCallback(radius=0.5, sample=data[0][0])
    trainer = Trainer(num_nodes=1,
                      devices=devices,
                      gradient_clip_val=1.0,
                      num_sanity_val_steps=0,
                      # log_every_n_steps=1,
                      accelerator='auto',
                      # overfit_batches=1.0,
                      max_epochs=max_epochs,
                      logger=logger,
                      # callbacks=[visualizer_callback, checkpoint_callback])
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == "__main__":
    main_loop()

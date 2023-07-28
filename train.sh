#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deep-signature-2-ubuntu
export WANDB_CONFIG_DIR=/home/gal.yona/deep-signature-2/wandb_tmp_dir
export WANDB_DIR=/home/gal.yona/deep-signature-2/wandb_tmp_dir
export WANDB_CACHE_DIR=/home/gal.yona/deep-signature-2/wandb_tmp_dir
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python train_model.py
#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deep-signature-2-ubuntu
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python dataset_generation.py
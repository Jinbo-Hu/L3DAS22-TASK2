#!/bin/bash

set -e

CONFIG_FILE='./configs/ein_seld/seld.yaml'
GPU_ID=3

# Extract data
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_data' --dataset_type='train'
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_data' --dataset_type='test'


# Extract frame label
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_frame_label' --dataset_type='train'
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_frame_label' --dataset_type='test'

# Extract track label
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_track_label'

# Extract Scalar
CUDA_VISIBLE_DEVICES=$GPU_ID python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_scalar' 

# Extract salsa feature
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='salsa_extractor' --dataset_type='train'
python code/main.py -c $CONFIG_FILE preprocess --preproc_mode='salsa_extractor' --dataset_type='test'
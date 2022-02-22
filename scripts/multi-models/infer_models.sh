# #!/bin/bash

GPU_ID=2
PORT=12360

CONFIG_FILE='./scripts/multi-models/logmelIV_ConvConformer.yaml'
CUDA_VISIBLE_DEVICES=$GPU_ID python code/main.py --config_file=$CONFIG_FILE infer

CONFIG_FILE='./scripts/multi-models/logmelIV_DenseConformer.yaml'
CUDA_VISIBLE_DEVICES=$GPU_ID python code/main.py --config_file=$CONFIG_FILE infer

CONFIG_FILE='./scripts/multi-models/SALSA_ConvConformer.yaml'
CUDA_VISIBLE_DEVICES=$GPU_ID python code/main.py --config_file=$CONFIG_FILE infer

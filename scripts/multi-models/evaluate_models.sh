#!/bin/bash

CONFIG_FILE='./scripts/multi-models/logmelIV_ConvConformer.yaml'
python code/main.py --config_file=$CONFIG_FILE evaluate

CONFIG_FILE='./scripts/multi-models/logmelIV_DenseConformer.yaml'
python code/main.py --config_file=$CONFIG_FILE evaluate

CONFIG_FILE='./scripts/multi-models/SALSA_ConvConformer.yaml'
python code/main.py --config_file=$CONFIG_FILE evaluate

#!/bin/bash

# download l3dass22 dataset

DOWNLOAD_PATH='./dataset/l3das22'
kaggle datasets download -d l3dasteam/l3das22 -p $DOWNLOAD_PATH --force --unzip



# download l3das21 dataset (Optional)

# DOWNLOAD_PATH='./dataset/l3das21'
# DEV_PATH=$DOWNLOAD_PATH'/L3DAS_Task2_dev.zip'
# TRAIN_PATH=$DOWNLOAD_PATH'/L3DAS_Task2_train.zip'

# wget -P $DOWNLOAD_PATH https://zenodo.org/record/4642005/files/L3DAS_Task2_dev.zip
# wget -P $DOWNLOAD_PATH https://zenodo.org/record/4642005/files/L3DAS_Task2_train.zip

# unzip $DEV_PATH -d $DOWNLOAD_PATH
# unzip $TRAIN_PATH -d DOWNLOAD_PATH

# Test set of l3das21 Task2 is avaliable on 
# https://drive.google.com/file/d/1LvurkN8QyS2fMnEbOa_3BpHnzYijSuYK/view
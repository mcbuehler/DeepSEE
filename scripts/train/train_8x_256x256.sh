#!/bin/bash
PYTHON_ENV=YOUR_PYTHON_PATH/bin

# This should contain the path to the cloned DeepSEE repository
DEEPSEE_PATH=YOUR_DEEPSEE_PATH
cd $DEEPSEE_PATH

IMG_DIR_TRAIN=YOUR_TRAIN_IMG_PATH
LABEL_DIR_TRAIN=YOUR_TRAIN_LABEL_PATH
IMG_DIR_VAL=YOUR_VAL_IMG_PATH
LABEL_DIR_VAL=YOUR_VAL_LABEL_PATH

# This is a file with identity annotations for each image.
# Please check the README to find out where to download it or how to compute it.
IDENTITIES_FILE=YOUR_PATH_TO_IDENTITIES_FILE/identities_all.csv
DATASET=celebamaskhq

# Give your GPU id(s) here or use -1 if none.
GPU_IDS=0


# Choose the model you want to train by uncommenting
NAME=CelebAMaskHQ_8x_independent_256x256
#NAME=CelebAMaskHQ_8x_guided_256x256
$PYTHON_ENV/python $DEEPSEE_PATH/train.py \
    --name $NAME \
    --dataset_mode $DATASET \
    --image_dir $IMG_DIR_TRAIN --label_dir $LABEL_DIR_TRAIN \
    --image_dir_val $IMG_DIR_VAL --label_dir_val $LABEL_DIR_VAL \
    --evaluate_val_set \
    --batchSize 1 \
    --tf_log \
    --niter 50 --niter_decay 25 \
    --identities_file $IDENTITIES_FILE \
    --gpu_ids $GPU_IDS \
    --load_config_from_name

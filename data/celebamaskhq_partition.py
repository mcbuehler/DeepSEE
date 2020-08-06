"""
Partitions the CelebAMask-HQ dataset into train / val / test.
Adjusted script from
https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_partition.py
"""
import os
from shutil import copyfile

import pandas as pd

# This should be the path to the extracted CelebAMask-HQ.zip folder
path_in = "YOURPATH/CelebAMask-HQ"
# We will create subfolders for the dataset splits under this path
path_out = "YOURPATH/CelebAMask-HQ/splits"

if "YOURPATH" in path_in:
    print("Did you set the correct paths?")
    exit()

path_images_in = os.path.join(path_in, 'CelebA-HQ-img')
path_mapping = os.path.join(path_in, 'CelebA-HQ-to-CelebA-mapping.txt')

# Folders for dataset splits
path_train_images_out = os.path.join(path_out, 'train_img')
path_test_images_out = os.path.join(path_out, 'test_img')
path_val_images_out = os.path.join(path_out, 'val_img')

for folder in [path_train_images_out,  path_val_images_out,  path_test_images_out]:
    os.makedirs(folder, exist_ok=True)

image_list = pd.read_csv(path_mapping, delim_whitespace=True, header=0)

for idx, x in zip(image_list.idx, image_list.orig_idx):
    print (idx, x)
    if x >= 162771 and x < 182638:
        copyfile(os.path.join(path_images_in, str(idx) + '.jpg'), os.path.join(path_val_images_out, str(idx) + '.jpg'))

    elif x >= 182638:
        copyfile(os.path.join(path_images_in, str(idx) + '.jpg'), os.path.join(path_test_images_out, str(idx) + '.jpg'))
    else:
        copyfile(os.path.join(path_images_in, str(idx) + '.jpg'), os.path.join(path_train_images_out, str(idx) + '.jpg'))

print("Written to subfolders in {}.".format(path_out))
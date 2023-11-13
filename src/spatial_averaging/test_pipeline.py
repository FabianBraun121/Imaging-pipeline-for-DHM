# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:38:04 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import spatial_averaging as sa
import config
import tifffile
import numpy as np
import filecmp

def are_images_similar(image_path1, image_path2, pixel_threshold=0.01):
    # Open the TIFF images using tifffile
    img1 = tifffile.imread(image_path1)
    img2 = tifffile.imread(image_path2)

    # Check if the shape of the arrays is the same
    if img1.shape != img2.shape:
        return False

    # Calculate the absolute pixel-wise difference
    pixel_diff = np.abs(img1 - img2)

    # Check if the maximum difference is within the threshold
    if np.max(pixel_diff) <= pixel_threshold:
        return True
    else:
        return False

def are_folders_similar(folder1, folder2, pixel_threshold=0.01):
    # Check if the paths are directories
    if not (os.path.isdir(folder1) and os.path.isdir(folder2)):
        return False

    # Use filecmp to compare the directories
    dcmp = filecmp.dircmp(folder1, folder2)

    # Check if common files are similar
    for common_file in dcmp.common_files:
        file_path1 = os.path.join(folder1, common_file)
        file_path2 = os.path.join(folder2, common_file)
        if not are_images_similar(file_path1, file_path2, pixel_threshold):
            return False

    # Recursively check subdirectories
    for common_dir in dcmp.common_dirs:
        subdir1 = os.path.join(folder1, common_dir)
        subdir2 = os.path.join(folder2, common_dir)
        if not are_folders_similar(subdir1, subdir2, pixel_threshold):
            return False

    return True

koala_config_nr = 279
select_recon_rectangle = True
base_dir = os.getcwd() + os.sep + r'..\data\test_data'

config.load_config(koala_config_nrIn=koala_config_nr, save_formatIn='.tif', save_in_same_folderIn=False)
pipe = sa.pipeline.Pipeline(base_dir=base_dir)
if select_recon_rectangle:
    pipe.select_positions_recon_rectangle(same_for_all_pos = True, recon_corners=((100,700),(100,700)))
pipe.process()
[os.remove(os.path.join(pipe.saving_dir, j)) for j in os.listdir(pipe.saving_dir) if j.endswith('.json')]

folder_path1 = pipe.saving_dir
folder_path2 = os.getcwd() + os.sep + r'..\data\test_data reference'

if are_folders_similar(folder_path1, folder_path2, pixel_threshold=0.01):
    print("Test passed")
else:
    print("Test not passed. Wrong output")



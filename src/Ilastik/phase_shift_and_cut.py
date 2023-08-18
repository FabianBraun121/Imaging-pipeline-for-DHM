# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:48:41 2023

@author: SWW-Bc20
"""

import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
from skimage.registration import phase_cross_correlation
from scipy import ndimage
import numpy as np

import tifffile as tiff

def save_image_as_tif(image_array, file_path):
    """
    Save an image array as a TIFF file.

    Args:
        image_array (ndarray): The image array to be saved.
        file_path (str): The file path to save the TIFF file.

    Raises:
        ValueError: If the image_array is not a valid ndarray.
        Exception: If an error occurs while saving the TIFF file.
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a valid ndarray.")

    try:
        tiff.imwrite(file_path, image_array)
    except Exception as e:
        print(f"An error occurred while saving the image: {e}.")

def save_image_as_tif16(image_array, file_path):
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a valid ndarray.")

    try:
        scaled_data = ((image_array + np.pi/2) / np.pi) * 65535
        scaled_data = scaled_data.astype(np.uint16)
        tiff.imwrite(file_path, scaled_data)
    except Exception as e:
        print(f"An error occurred while saving the image: {e}.")


folder_names = r'F:\E10_20230216\2023-04-19 12-27-21 phase averages'
folders = [f for f in os.listdir(folder_names) if os.path.isdir(folder_names + os.sep + f)][-2:-1]
save_folders = r'F:\E10_20230216\2023-04-19 12-27-21 phase averages\cropped'
if not os.path.exists(save_folders):
    os.makedirs(save_folders)
#%%
for folder in folders:
    foldername = folder_names + os.sep + folder
    files = [f for f in os.listdir(foldername) if f.endswith(".bin")]
    save_folder = save_folders + os.sep + folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    image, _ = binkoala.read_mat_bin(foldername + os.sep + files[0])
    fname = save_folder + os.sep + files[0]
    px_size = (float)(_["px_size"])
    hconv = (float)(_["hconv"])
    unit_code = (int)(_["unit_code"])
    binkoala.write_mat_bin(fname, image[10:710,90:790], 700, 700, px_size, hconv, unit_code)
    # save_image_as_tif16(image[10:710,90:790], fname[:-4]+".tif")
    
    for file in files[1:]:
        next_image, _ = binkoala.read_mat_bin(foldername + os.sep + file)
        shift_measured, error, diffphase = phase_cross_correlation(image, next_image, upsample_factor=10, normalization=None)
        shift_vector = (shift_measured[0],shift_measured[1])
        image = ndimage.shift(next_image, shift=shift_vector, mode='constant')
        fname = save_folder + os.sep + file
        binkoala.write_mat_bin(fname, image[10:710,90:790], 700, 700, px_size, hconv, unit_code)
        # save_image_as_tif16(image[10:710,90:790], fname[:-4]+".tif")
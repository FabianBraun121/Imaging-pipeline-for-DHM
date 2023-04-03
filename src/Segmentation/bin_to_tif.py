# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:55:13 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
from PIL import Image
import tifffile
import numpy as np

# Define the TIFFTAG_IMAGEDESCRIPTION constant manually
TIFFTAG_IMAGEDESCRIPTION = 270

path = r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00001'
save_path = r'F:\C11_20230217\test_float'
if not os.path.exists(save_path):
    os.makedirs(save_path)
image_timesteps = os.listdir(path)

for i, image_timestep in enumerate(image_timesteps):
    image, header = binkoala.read_mat_bin(path + '/' + image_timestep)
    width = (int)(header["width"])
    height = (int)(header["height"])
    px_size = (float)(header["px_size"])
    hconv = (float)(header["hconv"])
    unit_code = (int)(header["unit_code"])
    file_path = save_path + f'/pos01_chan01_time{str(i).zfill(3)}.tif'
    # Save as TIFF file with header information
    tifffile.imwrite(file_path, image, photometric='minisblack', metadata={
        'width': width,
        'height': height,
        'unit': 'microns',
        'resolution': (1/px_size, 1/px_size),
        'description': 'Image with negative float32 values, converted to 2D uint8 array',
        'hconv': hconv,
        'unit_code': unit_code
    })
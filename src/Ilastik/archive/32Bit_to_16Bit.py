# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 08:51:43 2023

@author: SWW-Bc20
"""
import os
import numpy as np
import tifffile

input_dir = r"C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\tracking_set\test_tracking"  # Change this to your source directory
output_dir = r"C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\eval_movies\2D\movie_tifs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def save_float16_tiff(array, filepath):
    # Convert the array to float16
    array_float16 = array.astype(np.float16)

    # Create a TIFF file with float16 data type
    with tifffile.TiffWriter(filepath, bigtiff=False) as tiff:
        tiff.write(array_float16, photometric='minisblack')

for filename in os.listdir(input_dir):
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        # Open the image file as a numpy array
        img = tifffile.imread(input_image_path)
        img = img.astype(np.float16)
    
        # Create a TIFF file with float16 data type
        with tifffile.TiffWriter(output_image_path) as tiff:
            tiff.write(img)

print("Conversion of images completed!")
